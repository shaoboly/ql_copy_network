import tensorflow as tf

from model_pools import modeling
from model_pools.base_model_multi_gpu import BaseModelMulti
from model_pools.model_utils.layer import attention_bias
from model_pools.model_utils.module import smooth_cross_entropy, transformer_decoder
from model_pools.modeling import embedding_lookup, embedding_postprocessor
from utils.copy_utils import calculate_final_logits, tf_trunct
import optimization

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign

def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def sum_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class MultiGPUBaselineCopyAddImprove(BaseModelMulti):
    """
    Based on BertSummarizerCopy, change some model settings, 41.9 on CNN/DM
    """

    def __init__(self, bert_config, batcher, hps):
        super(MultiGPUBaselineCopyAddImprove, self).__init__(hps, bert_config, batcher)
        self.gSess_train = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                                      graph=self.graph)

    def build_graph(self):
        with self.graph.as_default():
            self._build_summarization_model()

    def _add_placeholders(self):
        self.batch_size = self.hps.train_batch_size if self.is_training else self.hps.eval_batch_size
        self.input_ids = tf.placeholder(tf.int32, [self.batch_size, None], name='input_ids')  # [b, l_s]
        self.input_len = tf.placeholder(tf.int32, [self.batch_size], name='input_len')  # [b]
        self.segment_ids = tf.placeholder(tf.int32, [self.batch_size, None], name='segment_ids')
        self.output_ids = tf.placeholder(tf.int32, [self.batch_size, None], name='output_ids')  # [b, l_t], not use
        self.output_len = tf.placeholder(tf.int32, [self.batch_size], name='output_len')  # [b]

        # copy related placeholder
        self.output_label = tf.placeholder(tf.int32, [self.batch_size, None], name='output_label')  # [b, l_t], output_ids_oo
        self.max_out_oovs = tf.placeholder(tf.int32, [], name='max_out_oovs')  # []
        self.input_ids_oo = tf.placeholder(tf.int32, [self.batch_size, None], name='input_ids_oo')  # [b, l_s]

        self.input_mask = tf.sequence_mask(self.input_len,
                                           maxlen=tf.shape(self.input_ids)[1],
                                           dtype=tf.float32)  # [b, l_s]
        self.output_mask = tf.sequence_mask(self.output_len,
                                            maxlen=tf.shape(self.output_label)[1],
                                            dtype=tf.float32)  # [b, l_t]
        self.out_segment_ids = tf.zeros_like(self.output_label, dtype=tf.int32, name='out_segment_ids')
        self.tiled_len = tf.shape(self.output_label)[1]
        # encoder output for inference
        self.enc_output = tf.placeholder(tf.float32, [self.batch_size, None, self.hps.hidden_size], name='enc_output')

    def _n_gpu_split_placeholders(self, n):
        self.batch_size_ngpu = self.batch_size // n
        self.input_ids_ngpu = tf.split(self.input_ids, n)
        self.input_len_ngpu = tf.split(self.input_len, n)
        self.segment_ids_ngpu = tf.split(self.segment_ids, n)
        self.output_ids_ngpu = tf.split(self.output_ids, n)
        self.output_len_ngpu = tf.split(self.output_len, n)
        self.output_label_ngpu = tf.split(self.output_label, n)
        self.max_out_oovs_ngpu = [self.max_out_oovs for _ in range(n)]
        self.input_ids_oo_ngpu = tf.split(self.input_ids_oo, n)
        self.input_mask_ngpu = tf.split(self.input_mask, n)
        self.output_mask_ngpu = tf.split(self.output_mask, n)
        self.out_segment_ids_ngpu = tf.split(self.out_segment_ids, n)
        self.tiled_len_ngpu = (tf.shape(output_label)[1] for output_label in self.output_label_ngpu)
        self.enc_output_ngpu = tf.split(self.enc_output, n)


    def _build_summarization_model(self):
        is_training = self.is_training
        config = self.bert_config

        self._add_placeholders()
        self._n_gpu_split_placeholders(self.hps.n_gpu)

        gpu_grads = []
        gpu_ops = []
        zero_ops = []
        accum_gpu_opt = []
        for i, (input_ids, input_len, segment_ids, output_ids, output_len,
                output_label, max_out_oovs, input_ids_oo, input_mask,output_mask,
                out_segment_ids, tiled_len, enc_output) in enumerate(zip(self.input_ids_ngpu,
                                                        self.input_len_ngpu, self.segment_ids_ngpu,
                                                        self.output_ids_ngpu,self.output_len_ngpu,
                                                        self.output_label_ngpu,self.max_out_oovs_ngpu,
                                                        self.input_ids_oo_ngpu,self.input_mask_ngpu,
                                                        self.output_mask_ngpu,self.out_segment_ids_ngpu,
                                                        self.tiled_len_ngpu,self.enc_output_ngpu)):
            do_reuse = True if i > 0 else None
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                '''Creates a classification model.'''
                model = modeling.BertModel(
                    config=self.bert_config,
                    is_training=is_training,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=segment_ids,
                    use_one_hot_embeddings=self.hps.use_tpu)  # use_one_hot_embeddings=Flags.tpu ?

                encoder_output = model.get_sequence_output()  # [b, l_s, h]

                hidden_size = encoder_output.shape[2].value

                enc_attn_bias = attention_bias(input_mask, 'masking')

                out_dict_size = len(self.hps.vocab_out)
                with tf.variable_scope('bert-output'):
                    with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                        # Perform embedding lookup on the target word ids.
                        (out_embed, bert_embeddings) = embedding_lookup(
                            input_ids=output_ids,  # here the embedding input of decoder have to be output_ids
                            vocab_size=out_dict_size,  # decode dictionary modified
                            embedding_size=config.hidden_size,
                            initializer_range=config.initializer_range,
                            word_embedding_name='word_embeddings',
                            use_one_hot_embeddings=False)

                        # Add positional embeddings and token type embeddings, then layer
                        # normalize and perform dropout.
                        out_embed = embedding_postprocessor(
                            input_tensor=out_embed,
                            use_token_type=True,
                            token_type_ids=out_segment_ids,
                            token_type_vocab_size=config.type_vocab_size,
                            token_type_embedding_name='token_type_embeddings',
                            use_position_embeddings=True,
                            position_embedding_name='position_embeddings',
                            initializer_range=config.initializer_range,
                            max_position_embeddings=config.max_position_embeddings,
                            dropout_prob=config.hidden_dropout_prob)

                with tf.variable_scope('decode'):
                    decoder_weights = bert_embeddings
                    masked_out_embed = out_embed * tf.expand_dims(output_mask, -1)
                    dec_attn_bias = attention_bias(tf.shape(masked_out_embed)[1], 'causal')
                    decoder_input = tf.pad(masked_out_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1,
                                         :]  # Shift left
                    all_att_weights, decoder_output = transformer_decoder(decoder_input,
                                                                                    encoder_output,
                                                                                    dec_attn_bias,
                                                                                    enc_attn_bias,
                                                                                    self.hps)
                    # [b, l_t, e] => [b*l_t, v]
                    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
                    vocab_logits = tf.matmul(decoder_output, decoder_weights, False,
                                                  True)  # (b * l_t, v)
                    vocab_probs = tf.nn.softmax(vocab_logits)  # [b * l_t, v]
                    # vocab_size = len(self.hps.vocab)
                    with tf.variable_scope('copy'):
                        logits = calculate_final_logits(decoder_output, all_att_weights,
                                                             vocab_probs,
                                                             input_ids_oo, max_out_oovs, input_mask,
                                                             out_dict_size,
                                                             tiled_len)  # [b * l_t, v + v']
                        pred_ids = tf.reshape(tf.argmax(logits, axis=-1), [self.batch_size_ngpu, -1])

                with tf.variable_scope('loss'):
                    ce = smooth_cross_entropy(
                        logits,
                        output_label,
                        self.hps.label_smoothing)

                    ce = tf.reshape(ce, tf.shape(output_label))  # [b, l_t]

                    loss = tf.reduce_sum(ce * output_mask) / tf.reduce_sum(output_mask)  # scalar
                    params = tf.trainable_variables()

                    if self.hps.accumulate_step>1:
                        # Create a constant with
                        #accumulation_step = tf.constant(self.hps.accumulate_step, dtype=tf.float32)
                        # Retrieve all trainable variables you defined in your graph
                        all_vars = tf.trainable_variables()
                        # Creation of a list of variables with the same shape as the trainable ones
                        accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in
                                      all_vars]
                        # give accumulation tensor initialized value (with 0s)
                        zero_accum_op = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

                        zero_ops.append(zero_accum_op)

                        gvs = tf.gradients(loss, all_vars)
                        (gvs, _) = tf.clip_by_global_norm(gvs, clip_norm=1.0)

                        accum_op = [accum_var.assign_add(gv) for accum_var, gv in zip(accum_vars, gvs) if
                                    gv is not None]
                        accum_gpu_opt.append(accum_op)
                        grads = list(zip(accum_vars, all_vars))
                    else:
                        grads = tf.gradients(loss, params)
                        grads = list(zip(grads, params))

                    gpu_grads.append(grads)
                    gpu_ops.append([loss, logits])

                    if i == 0:
                        self.pred_ids = pred_ids
                        self.loss = loss
                        self.logits = logits
                        self.encoder_output = encoder_output
                        self.out_embed = out_embed
                        self.all_att_weights = all_att_weights
                    '''else:
                        self.pred_ids = tf.concat(self.pred_ids, pred_ids)
                        self.loss = self.loss.append(loss)
                        self.logits = tf.concat(self.logits, logits)
                        self.encoder_output = tf.concat(self.encoder_output, encoder_output)
                        self.out_embed = tf.concat(self.out_embed, out_embed)'''

        grads = sum_grads(gpu_grads)
        grads = [g for g, p in grads]
        self.total_gradient = grads

        if self.hps.accumulate_step > 1:
            self.accum_op, self.zero_accum_op = accum_gpu_opt,zero_ops

        tf.summary.scalar('loss', self.loss)


    def create_opt_op(self):
        self.optimizer, self.cur_lr = optimization.create_optimizer(float(self.hps.learning_rate), self.num_train_steps,
                                                                    self.num_warmup_steps,
                                                                    self.hps.use_tpu)
        self.train_op = optimization.comput_opt_with_gradient_multi(self.total_gradient, self.optimizer,self.hps.accumulate_step)
        if self.hps.accumulate_step > 1:
            '''ret = optimization.create_opt_op_grad_accum(self.loss,
                                                        self.optimizer,
                                                        self.hps.accumulate_step)
            self.train_op, self.accum_op, self.zero_accum_op, self.accum_vars = ret'''
            pass
        else:
            #self.train_op = optimization.create_train_op(self.loss, self.optimizer)
            self.accum_op = None
            self.zero_accum_op = None



    def decode_infer(self, inputs, state):
        # state['enc']: [b * beam, l_s, e]  ,   state['dec']: [b * beam, q', e]
        # q' = previous decode output length
        # during infer, following graph are constructed using beam search
        with self.graph.as_default():
            config = self.bert_config

            target_sequence = inputs['target']  # [b * beam, q']
            vocab_size = len(self.hps.vocab_out)
            # trunct word idx, change those greater than vocab_size to unkId
            shape = target_sequence.shape
            unkid = self.hps.vocab_out[self.hps.unk]
            # target_sequence = tf_trunct(target_sequence, vocab_size, self.hps.unkId)
            target_sequence = tf_trunct(target_sequence, vocab_size, unkid)
            target_sequence.set_shape(shape)

            target_length = inputs['target_length']
            target_seg_ids = tf.zeros_like(target_sequence, dtype=tf.int32, name='target_seg_ids_infer')
            tgt_mask = tf.sequence_mask(target_length,
                                        maxlen=tf.shape(target_sequence)[1],
                                        dtype=tf.float32)  # [b, q']

            # with tf.variable_scope('bert', reuse=True):
            out_dict_size = len(self.hps.vocab_out)
            with tf.variable_scope('bert', reuse=True):
                with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
                    # Perform embedding lookup on the target word ids.
                    (tgt_embed, _) = embedding_lookup(
                        input_ids=target_sequence,
                        vocab_size=out_dict_size,  # out vocab size
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=False)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    tgt_embed = embedding_postprocessor(
                        input_tensor=tgt_embed,
                        use_token_type=True,
                        token_type_ids=target_seg_ids,
                        token_type_vocab_size=config.type_vocab_size,
                        token_type_embedding_name='token_type_embeddings',
                        use_position_embeddings=True,
                        position_embedding_name='position_embeddings',
                        initializer_range=config.initializer_range,
                        max_position_embeddings=config.max_position_embeddings,
                        dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope('decode', reuse=True):
                # [b, q', e]
                masked_tgt_embed = tgt_embed * tf.expand_dims(tgt_mask, -1)
                dec_attn_bias = attention_bias(tf.shape(masked_tgt_embed)[1], "causal")
                decoder_input = tf.pad(masked_tgt_embed, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # Shift left

                infer_decoder_input = decoder_input[:, -1:, :]
                infer_dec_attn_bias = dec_attn_bias[:, :, -1:, :]

                ret = transformer_decoder(infer_decoder_input,
                                          self.enc_output,
                                          infer_dec_attn_bias,
                                          self.enc_attn_bias,
                                          self.hps,
                                          state=state['decoder'])

                all_att_weights, decoder_output, decoder_state = ret
                decoder_output = decoder_output[:, -1, :]  # [b * beam, e]
                vocab_logits = tf.matmul(decoder_output, self.decoder_weights, False, True)  # [b * beam, v]
                vocab_probs = tf.nn.softmax(vocab_logits)
                vocab_size = out_dict_size  # out vocabsize
                # we have tiled source_id_oo before feed, so last argument is set to 1
                with tf.variable_scope('copy'):
                    logits = calculate_final_logits(decoder_output, all_att_weights,
                                                    vocab_probs,
                                                    self.input_ids_oo, self.max_out_oovs, self.input_mask, vocab_size,
                                                    tgt_seq_len=1)
                log_prob = tf.log(logits)  # [b * beam, v + v']
        return log_prob, {'encoder': state['encoder'], 'decoder': decoder_state}

    def _make_input_key(self):
        """The key name should be equal with property name in Batch class"""
        self.tensor_list = {'source_ids': self.input_ids,
                            'source_ids_oo': self.input_ids_oo,
                            'source_len': self.input_len,
                            'source_seg_ids': self.segment_ids,
                            'target_ids': self.output_ids,
                            'target_ids_oo': self.output_label,
                            'max_oov_num': self.max_out_oovs,
                            'target_len': self.output_len,
                            'loss': self.loss,
                            'logits': self.logits,
                            'encoder_output': self.enc_output,
                            'pred_ids': self.pred_ids,
                            # debug
                            'train_encoded': self.encoder_output,
                            'out_embed': self.out_embed,
                            'all_att_weights': self.all_att_weights
                            }
        if self.is_training:
            self.tensor_list.update({
                'train_opt': self.train_op,
                'grad_accum': self.accum_op,
                'summaries': self._summaries
            })
        self.input_keys_infer = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                                 'encoder_output']
        self.input_keys = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                           'target_ids', 'target_ids_oo', 'target_len']
        self.output_keys_train = ['loss', 'train_opt', 'summaries', 'pred_ids', 'train_encoded', 'logits', 'out_embed',
                                  'all_att_weights']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']