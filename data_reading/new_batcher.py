import logging
import queue
import random
import time
from random import shuffle
from threading import Thread
from typing import Iterable

import numpy as np

import tokenization
from data_reading import processors
from utils.data_utils import refine_train_summary
from utils.utils import init_sentence_level_info


class Sample(object):
    """Class representing a train/dev/test sample."""

    def __init__(self, sample, params, mode):
        self.params = params
        self.mode = mode
        self.original_data = sample['origin_sample']

        # Process the source sequence
        self.source_ids = sample['article_ids']
        self.source_len = sample['article_lens']
        self.source_ids_oo = sample['article_ids_oo']
        self.source_seg_ids = sample['article_seg_ids']
        self.source_oovs = sample['src_oovs']
        self.source_oov_num = len(sample['src_oovs'])

        # Process the target sequence
        self.target_ids = sample['summary_ids']
        self.target_ids_oo = sample['summary_ids_oo']
        self.target_len = sample['summary_lens']


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class Batch(object):
    """Class representing a minibatch of train samples."""

    def __init__(self, sample_list: Iterable[Sample], true_num, params, mode):
        """Turns the sample_list into a Batch object."""
        self.true_num = true_num
        self.params = params
        self.period_id = self.params.vocab['.']
        self.mode = mode
        self.max_src_len = params.max_seq_length
        self.max_tgt_len = params.max_out_seq_length
        self.pad_id = params.padId
        self.init_encoder_seq(sample_list)  # initialize the input to the encoder
        self.init_decoder_seq(sample_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(sample_list)  # store the original strings

    def init_encoder_seq(self, sample_list):
        # group
        self.source_ids = [ex.source_ids for ex in sample_list]
        self.source_len = [ex.source_len for ex in sample_list]
        self.source_seg_ids = [ex.source_seg_ids for ex in sample_list]
        self.source_ids_oo = [ex.source_ids_oo for ex in sample_list]
        self.source_oov_num = [ex.source_oov_num for ex in sample_list]

        # pad
        max_src_len = min(max(self.source_len), self.max_src_len)
        self.source_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                           for ids in self.source_ids]
        self.source_ids_oo = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                              for ids in self.source_ids_oo]
        self.source_seg_ids = [(ids + [self.pad_id] * (max_src_len - len(ids)))[: max_src_len]
                               for ids in self.source_seg_ids]

        # to numpy array
        self.source_ids = np.array(self.source_ids, dtype=np.int32)
        self.source_ids_oo = np.array(self.source_ids_oo, dtype=np.int32)
        self.source_seg_ids = np.array(self.source_seg_ids, dtype=np.int32)
        self.source_len = np.array(self.source_len, dtype=np.int32)

        # Determine the max number of in-article OOVs in this batch
        self.max_oov_num = max([len(ex.source_oovs) for ex in sample_list])
        # Store the in-article OOVs themselves
        self.source_oovs = [ex.source_oovs for ex in sample_list]
        # Fake encoder output
        self.encoder_output = np.zeros([1, 1, self.params.hidden_size], dtype=np.float32)

    def init_decoder_seq(self, sample_list):
        # group
        self.target_ids = [ex.target_ids for ex in sample_list]
        self.target_len = [ex.target_len for ex in sample_list]
        self.target_ids_oo = [ex.target_ids_oo for ex in sample_list]

        # pad
        max_tgt_len = min(max(self.target_len), self.max_tgt_len)
        self.target_ids = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                           for ids in self.target_ids]
        self.target_ids_oo = [(ids + [self.pad_id] * (max_tgt_len - len(ids)))[: max_tgt_len]
                              for ids in self.target_ids_oo]

        self.sent_level_attn_bias = init_sentence_level_info(self.period_id, self.target_ids)

        # to numpy array
        self.target_ids = np.array(self.target_ids, dtype=np.int32)
        self.target_ids_oo = np.array(self.target_ids_oo, dtype=np.int32)
        self.target_len = np.array(self.target_len, dtype=np.int32)
        self.init_lm_placeholder()

    def init_lm_placeholder(self):
        mask_id = self.params.maskId
        shape = self.target_ids.shape
        batch, length = shape[0], shape[1]
        target_ids = np.expand_dims(self.target_ids, 1)  # (b, 1, l_t)
        target_ids = np.tile(target_ids, [1, length, 1])  # (b, l_t, l_t)
        self.lm_output_ids = np.reshape(target_ids, [-1, length])  # (b * l_t, l_t)
        self.lm_position = np.array([list(range(length)) for _ in range(batch)], dtype=np.int32)  # (b, l_t)
        self.lm_position = np.reshape(self.lm_position, -1)  # (b * l_t)
        lm_position = np.expand_dims(self.lm_position, 1)  # (b * l_t, 1)
        # set i-th word id to MASK_ID
        for i in range(batch * length):
            self.lm_output_ids[i][lm_position[i]] = mask_id

    def store_orig_strings(self, sample_list):
        """Store the original strings in the Batch object"""
        self.original_data = [ex.original_data for ex in sample_list]  # list of lists

    def update_time(self):
        self.time_step += 1


# noinspection PyAttributeOutsideInit
class Batcher(object):
    """A class to generate minibatches of data. Buckets samples together based on length of the encoder sequence."""

    def __init__(self, processor, hps):
        """Initialize the batcher. Start threads that process the data into batches."""
        logging.info('Init data batcher...')
        self.mode = hps.mode
        self.is_train = self.mode == 'train'
        self.processor = processor
        self._config = hps
        self.batch_num = 0
        logging.info('Prepare data features...')

        self.prepare_examples()

        if not self.is_train:
            self._config.batch_size = self._config.eval_batch_size
        else:
            self._config.batch_size = self._config.train_batch_size

        """Read data features"""
        tokenizer = tokenization.FullTokenizer(vocab_file=self._config.vocab_file,
                                               do_lower_case=self._config.do_lower_case)

        self.features = []
        for example in self.examples:
            if not self.is_train:
                feature =  self.processor.convert_example_to_feature(example, tokenizer, self._config)
            else:
                feature = refine_train_summary(example, self._config)
                feature = self.processor.convert_example_to_feature(feature, tokenizer, self._config)

            self.features.append(feature)

        self.c_index = 0
        self.c_epoch = 0

    def prepare_examples(self):
        processor = self.processor
        mode = self.mode
        config = self._config

        if mode == 'train':
            examples = processor.get_train_examples(config.data_dir)
            random.shuffle(examples)
        elif mode == 'dev':
            examples = processor.get_dev_examples(config.data_dir)
        elif mode == 'test':
            examples = processor.get_test_examples(config.data_dir)
        else:
            raise ValueError('Only train dev test modes are supported: %s' % mode)

        self.examples = self.processor.filter_examples(examples)
        self.samples_number = len(examples)
        self.processor.log_statistics(examples)

    def next_batch(self):
        if self.c_index>=self.samples_number:
            self.c_index=0
            self.c_epoch+=1
            return None

        e_index = self.c_index+self._config.train_batch_size

        batch = self.features[self.c_index:e_index]

        self.c_index=e_index
        return batch




class EvalData:
    """Single thread data batcher class"""

    def __init__(self, hps):
        tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file,
                                               do_lower_case=hps.do_lower_case)
        # load custom processer from task name
        task_name = hps.task_name.lower()
        if task_name not in processors:
            raise ValueError('Task not found: %s' % task_name)
        processor = processors[task_name]()
        examples = processor.get_dev_examples(hps.data_dir)
        examples = processor.filter_examples(examples)

        self.features = [Sample(processor.convert_example_to_feature(example, tokenizer, hps), hps, hps.mode)
                         for example in examples]
        self.batches = []
        for i in range(0, len(self.features), hps.eval_batch_size):
            if i + hps.eval_batch_size > len(self.features):
                self.batches.append(Batch(self.features[i:], len(self.features[i:]), hps, hps.mode))
            else:
                self.batches.append(Batch(self.features[i:i + hps.eval_batch_size], hps.eval_batch_size, hps, hps.mode))
        self.cur_batch_num = 0

    def __len__(self):
        return len(self.batches)

    def next_batch(self):
        if self.cur_batch_num < len(self):
            res = self.batches[self.cur_batch_num]
            self.cur_batch_num += 1
        else:
            res = None
        return res

    def restart(self):
        self.cur_batch_num = 0
