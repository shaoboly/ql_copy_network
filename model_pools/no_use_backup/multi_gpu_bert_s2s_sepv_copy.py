import tensorflow as tf

from model_pools import modeling
from model_pools.base_model import BaseModel
from model_pools.model_utils.layer import attention_bias
from model_pools.model_utils.module import transformer_concated_decoder, smooth_cross_entropy, transformer_decoder
from model_pools.modeling import embedding_lookup, embedding_postprocessor
from utils.copy_utils import calculate_final_logits, tf_trunct
import logging
import os

from tensorflow.python.framework.errors_impl import FailedPreconditionError

import optimization
from model_pools import modeling
from utils.utils import gen_sess_config

from data_reading import processors
from data_reading.batcher import Batcher, EvalData
from model_pools import modeling, model_pools
from model_pools.bert_multi_gpu_graph import MultiBuildGraph

# noinspection PyAttributeOutsideInit

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

class MultiGPUTrain():
    """
    Pre-trained bert as encoder + T-decoder(transformer) with copy.
    To implement copy, we need to ->
    Feature feeding:
        Find all UNKs in source sequence and save them.
        Feed unk_num of each sample (max unk number of batch).
        Replace src_ids with src_ids_oo, replace UNKs with extended vocab ids.
        Replace target_ids with target_ids_oo, replace UNKs with extended vocab ids during training.
    Placeholder:
        batch max unk number, for build extended vocabulary.
        src_ids_oo.
        tgt_ids_oo.
    Decode_train:
        The input of decoder should be target_ids.(with unk in it)
        The label of decoder should be target_ids_oo.(with extended vocabulary in it)
        Calculate the copy attention weight during decode.
        Calculate the logits of extended vocabulary using mixture of copy weight and generation weight.
    Decode_infer:
        Input: truncated previous generated target sequence, map all extended vocab word to UNK.
        Use saved UNK str list to decode ids to sequence.
    """

    def __init__(self, bert_config, batcher, hps):
        logging.info('Start build model...')
        # Your model class should contains these properties
        self.input_keys, self.tensor_list, self.output_keys_train, self.output_keys_dev = None, None, None, None
        self.input_keys_infer, self.input_keys_infer_stage_2, self.output_keys_grad_accum = None, None, None
        self.loss = None

        self.graph = tf.Graph()
        self.hps = hps
        self.bert_config = bert_config
        self.is_training = (self.hps.mode == 'train')
        self.batcher = batcher

        if not self.is_training:
            self.hps.residual_dropout = 0.0
            self.hps.attention_dropout = 0.0
            self.hps.relu_dropout = 0.0
            self.hps.label_smoothing = 0.0

        self.num_train_steps = int(
            batcher.samples_number / (hps.train_batch_size * hps.accumulate_step) * hps.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * hps.warmup_proportion)
        self.gSess_train = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                                      graph=self.graph)

        logging.debug('Graph id: {}{}'.format(id(self.graph), self.graph))
        self.graph.as_default()
        with self.graph.as_default():
            self.build_graph()
        self._make_input_key()

    def build_graph(self):
        hps = self.hps
        self._add_placeholders()

        #split feed
        feed_data = self.input_ids, self.input_len, self.segment_ids, self.output_ids, self.output_len, \
                    self.output_label, self.input_ids_oo, self.input_mask, self.output_mask, \
                    self.out_segment_ids
        #tmp = self.max_out_oovs
        n_gpu = [1,3]
        split_feed_data =[]
        for x in feed_data:
            split_feed_data.append(tf.split(x, len(n_gpu), 0))
        #split_feed_data = (tf.split(x, n_gpu, 0) for x in feed_data)

        split_feed_data.append([self.max_out_oovs for i in n_gpu])
        #create
        model = MultiBuildGraph

        bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)

        train_batcher = self.batcher
        # create

        gpu_ops = []
        gpu_grads = []
        for i, xs in enumerate(zip(*split_feed_data)):
            do_reuse = True if i > 0 else None
            with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):
                # create trainning model
                train_model = model(bert_config, train_batcher, hps,xs,self.graph)
                loss = train_model.return_dict["loss"]
                logits = train_model.return_dict["logits"]

                params = tf.trainable_variables()

                grads = tf.gradients(loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)
                gpu_ops.append([loss, logits])

        #ops = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        grads = average_grads(gpu_grads)
        grads = [g for g, p in grads]
        #params = tf.trainable_variables()

        self.loss = (gpu_ops[0][0]+gpu_ops[1][0])/2

        self.optimizer, self.cur_lr = optimization.create_optimizer(float(self.hps.learning_rate), self.num_train_steps,
                                                                    self.num_warmup_steps,
                                                                    self.hps.use_tpu)
        self.train_op = optimization.comput_opt_with_gradient_multi(grads,self.optimizer)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.global_step = tf.train.get_or_create_global_step()

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
                            #'logits': self.logits,
                            'encoder_output': self.enc_output,
                            #'pred_ids': self.pred_ids,

                            }
        if self.is_training:
            self.tensor_list.update({
                'train_opt': self.train_op,
                'grad_accum': None,
                'summaries': None
            })
        self.input_keys_infer = ['source_ids','source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                                 'encoder_output']
        self.input_keys = ['source_ids', 'source_ids_oo', 'source_len', 'source_seg_ids', 'max_oov_num',
                           'target_ids', 'target_ids_oo', 'target_len']
        self.output_keys_train = ['loss', 'train_opt']
        self.output_keys_grad_accum = ['grad_accum']
        self.output_keys_dev = ['loss', 'logits']



    def _make_feed_dict(self, batch):
        feed_dict = {}
        for k in self.input_keys:
            feed_dict[self.tensor_list[k]] = getattr(batch, k)
        return feed_dict

    def make_infer_feed_dict(self, batch):
        feed_dict = {}
        for k in self.input_keys_infer:
            feed_dict[self.tensor_list[k]] = getattr(batch, k)
        return feed_dict

    def make_stage_2_infer_feed_dict(self, batch):
        feed_dict = {}
        for k in self.input_keys_infer_stage_2:
            feed_dict[self.tensor_list[k]] = getattr(batch, k)
        return feed_dict

    def run_train_step(self, batch):
        to_return = {}
        for k in self.output_keys_train:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        res = self.gSess_train.run(to_return, feed_dict)
        if self.hps.accumulate_step > 1:
            self.gSess_train.run(self.zero_accum_op)
        return res

    def run_grad_accum_step(self, batch):
        to_return = {}
        for k in self.output_keys_grad_accum:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        return self.gSess_train.run(to_return, feed_dict)

    def run_dev_step(self, batch):
        to_return = {}
        for k in self.output_keys_dev:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        return self.gSess_train.run(to_return, feed_dict)

    def _load_init_bert_parameter(self):
        init_checkpoint = self.hps.init_checkpoint
        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info('**** Trainable Variables ****')
            for var in tvars:
                init_string = ''
                if var.name in initialized_variable_names:
                    init_string = ', *INIT_FROM_CKPT*'
                tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape, init_string)

    def create_or_load_recent_model(self):
        with self.graph.as_default():
            if not os.path.isdir(self.hps.output_dir):
                os.mkdir(self.hps.output_dir)
            ckpt = tf.train.get_checkpoint_state(self.hps.output_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(self.gSess_train, ckpt.model_checkpoint_path)
            else:
                logging.info('Created model with fresh parameters and bert.')
                self._load_init_bert_parameter()
                self.gSess_train.run(tf.global_variables_initializer())
                #self._load_pretrain_s2s()

    def load_specific_variable(self, v):
        with self.graph.as_default():
            return self.gSess_train.run(v)

    def set_specific_variable(self, v,value):
        with self.graph.as_default():
            return self.gSess_train.run(tf.assign(v,value))

    def save_model(self, checkpoint_basename, with_step=True):
        with self.graph.as_default():
            if with_step:
                global_step = tf.train.get_or_create_global_step()
                try:
                    self.saver.save(self.gSess_train, checkpoint_basename, global_step=global_step)
                except FailedPreconditionError:
                    self.gSess_train.run(tf.initialize_variables([global_step]))
                    self.saver.save(self.gSess_train, checkpoint_basename, global_step=global_step)
            else:
                self.saver.save(self.gSess_train, checkpoint_basename)
            logging.info('model save in {}'.format(checkpoint_basename))

    def load_specific_model(self, best_path):
        with self.graph.as_default():
            self.saver.restore(self.gSess_train, best_path)

    def figure_out_memory_usage(self, batch):
        logging.info('Run figure_out_memory_usage...')
        to_return = {}
        for k in self.output_keys_train:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        run_metadata = tf.RunMetadata()
        self.gSess_train.run(to_return, feed_dict=feed_dict,
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True),
                             run_metadata=run_metadata)
        with open(os.path.join(self.hps.output_dir, 'memory_usage.txt'), 'w', encoding='utf-8') as out:
            out.write(str(run_metadata))