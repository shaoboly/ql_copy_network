import logging
import os

import tensorflow as tf
from tensorflow.python.framework.errors_impl import FailedPreconditionError

import optimization
from model_pools import modeling
from utils.utils import gen_sess_config

import pickle

# noinspection PyAttributeOutsideInit
class BaseModelMulti:
    def __init__(self, hps, bert_config, batcher):
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
            self.bert_config.hidden_dropout_prob = 0.0

        self.num_train_steps = int(
            batcher.samples_number / (hps.train_batch_size * hps.accumulate_step) * hps.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * hps.warmup_proportion)
        self.gSess_train = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                                      graph=self.graph)
        #self.gSess_train = tf.Session(config=gen_sess_config(self.hps), graph=self.graph)
        logging.debug('Graph id: {}{}'.format(id(self.graph), self.graph))
        self.build_graph()
        logging.info('Graph built done...')
        with self.graph.as_default():
            if self.is_training:
                self.create_opt_op()
            self._summaries = tf.summary.merge_all()
            self.global_step = tf.train.get_or_create_global_step()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.summary_writer = tf.summary.FileWriter(hps.output_dir, self.graph)
        self._make_input_key()
        logging.info('Model built done...')

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

    def _load_pretrain_s2s(self):
        if self.hps.pretrain_dir != None:
            ckpt = tf.train.get_checkpoint_state(self.hps.pretrain_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                logging.info("loading from {}".format(ckpt.model_checkpoint_path))
                self.saver.restore(self.gSess_train, ckpt.model_checkpoint_path)
            else:
                print("not found {}".format(self.hps.pretrain_dir))
            self.set_specific_variable(self.global_step,int(0))
            #self.set_specific_variable(self.global_step, int(0))

    def _load_pretrain_s2s_assgin(self):
        if self.hps.pretrain_dir != None:
            self.load_pretrain_s2s_assign(self.hps.pretrain_dir)
            self.set_specific_variable(self.global_step,int(0))

    def save_pretrain_s2s_assign(self,dirname):
        import json
        dirname = os.path.dirname(dirname)
        with self.graph.as_default():
            all_variable = {}
            vars = tf.trainable_variables()
            for v in vars:
                all_variable[v.name] = self.gSess_train.run(v)

            out_f = open(os.path.join(dirname, "para.pkl"),"wb")
            pickle.dump(all_variable,out_f)


    def load_pretrain_s2s_assign(self,dirname):
        import json
        #dirname = os.path.dirname(dirname)
        print("load from {}".format(os.path.join(dirname, "para.pkl")))
        in_f = open(os.path.join(dirname, "para.pkl"),"rb")
        all_variable = pickle.load(in_f)
        with self.graph.as_default():
            vars = tf.trainable_variables()
            for v in vars:
                if v.name in all_variable:
                    self.set_specific_variable(v,all_variable[v.name])

    def create_opt_op(self):
        self.optimizer, self.cur_lr = optimization.create_optimizer(float(self.hps.learning_rate), self.num_train_steps,
                                                                    self.num_warmup_steps,
                                                                    self.hps.use_tpu)
        if self.hps.accumulate_step > 1:
            ret = optimization.create_opt_op_grad_accum(self.loss,
                                                        self.optimizer,
                                                        self.hps.accumulate_step)
            self.train_op, self.accum_op, self.zero_accum_op, self.accum_vars = ret
        else:
            self.train_op = optimization.create_train_op(self.loss, self.optimizer)
            self.accum_op = None
            self.zero_accum_op = None

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
                self._load_pretrain_s2s()

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

    def build_graph(self):
        raise NotImplementedError()

    def _make_input_key(self):
        raise NotImplementedError()
