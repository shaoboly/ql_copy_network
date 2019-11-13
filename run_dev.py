from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os
from math import inf

import numpy as np
import tensorflow as tf

import config
from data_reading import processors
from data_reading.batcher import Batcher, EvalData
from inference import inference
from model_pools import modeling, model_pools
from utils.metric_utils import calculate_reward,convert_pred_to_token,calc_acc,calc_acc_with_ref
from utils.utils import print_parameters, pprint_params


def create_train_model(hps):
    model = model_pools[hps.model_name]

    bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)
    if hps.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            'Cannot use sequence length %d because the BERT model '
            'was only trained up to sequence length %d' %
            (hps.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(hps.output_dir)

    # load custom processer from task name
    task_name = hps.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    processor = processors[task_name]()
    train_batcher = Batcher(processor, hps)
    # create trainning model
    train_model = model(bert_config, train_batcher, hps)
    train_model.create_or_load_recent_model()
    logging.info('Total train steps: {}, total warm_up steps: {}...'.format(train_model.num_train_steps,
                                                                            train_model.num_warmup_steps))
    logging.info(
        'Init lr: {}, current lr: {}'.format(hps.learning_rate,
                                             train_model.load_specific_variable(train_model.cur_lr)))
    return train_model


def create_eval_model(hps):
    model = model_pools[hps.model_name]

    bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)
    if hps.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            'Cannot use sequence length %d because the BERT model '
            'was only trained up to sequence length %d' %
            (hps.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(hps.output_dir)
    task_name = hps.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)
    processor = processors[task_name]()
    hps.mode = 'dev'  # use dev set to choose best model
    batcher = Batcher(processor, hps)
    # create eval model
    model = model(bert_config, batcher, hps)
    #model.create_or_load_recent_model()
    return model


def train(hps):
    bert_model = create_train_model(hps)

    eval_model = create_eval_model(hps)

    checkpoint_basename = os.path.join(hps.output_dir, hps.model_name)
    logging.info(checkpoint_basename)
    bert_model.save_model(checkpoint_basename)

    start_step = bert_model.load_specific_variable(bert_model.global_step)
    if start_step == 0:
        print_parameters(bert_model.graph, output_detail=True, output_to_logging=True)

    if hps.debug and start_step == 0:
        # figure out how many GPU memory used
        batch = bert_model.batcher.next_batch()
        bert_model.figure_out_memory_usage(batch)

    avg_loss = []
    all_decoded = []
    run_reward = hasattr(bert_model, 'run_reward') and callable(bert_model.run_reward)
    for step in range(start_step, bert_model.num_train_steps):
        # feed draft or ground-truth
        feed_draft = step > hps.start_portion_to_feed_draft * bert_model.num_train_steps \
                     and step % hps.draft_feed_freq == 0
        for i in range(hps.accumulate_step - 1):  # gradient accumulation
            batch = bert_model.batcher.next_batch()
            batch.feed_draft = np.array(feed_draft, dtype=np.bool)
            if run_reward:
                batch_pred_ids = bert_model.run_reward(batch)['pred_ids']  # [b, length]
                batch.reward = calculate_reward(batch_pred_ids, batch, hps.padId)
            bert_model.run_grad_accum_step(batch)
            batch.reward = None

        batch = bert_model.batcher.next_batch()
        batch.feed_draft = np.array(feed_draft, dtype=np.bool)
        if run_reward:
            batch_pred_ids = bert_model.run_reward(batch)['pred_ids']  # [b, length]
            batch.reward = calculate_reward(batch_pred_ids, batch, hps.padId)
        results = bert_model.run_train_step(batch)

        avg_loss.append(results['loss'])
        bert_model.summary_writer.add_summary(results['summaries'], step)

        if step % 100 == 0:
            logging.info('step {}, 100 steps avg loss: {}\n'.format(step, sum(avg_loss) / len(avg_loss)))
            avg_loss = []
            bert_model.summary_writer.flush()

        if step % hps.evaluate_every_n_step == 0:
            bert_model.save_model(checkpoint_basename, with_step=True)
    bert_model.save_model(checkpoint_basename, with_step=True)

def eval_current_model(hps,eval_model):
    with eval_model.graph.as_default():
        t_all = 0
        t_acc = 0
        t_acc_t = 0
        step = 0
        loss_all = []
        all_decoded = []
        all_ref = []

        eval_model.create_or_load_recent_model()
        batcher = eval_model.batcher
        while True:
            batch = batcher.next_batch()
            step += 1
            if batch is None:
                break
            results = eval_model.run_dev_step(batch)
            loss_all.append(results['loss'])

            # decode for train words

            decoded_words = convert_pred_to_token(batch, results["pred_ids"], hps)
            ref_words = convert_pred_to_token(batch, batch["target_ids_oo"],hps)

            all_decoded.extend(decoded_words)
            all_ref.extend(ref_words)

            '''acc, acc_t, total = calc_acc(batch, all_decoded)
            all_decoded = []
            #
            t_all+=total
            t_acc+=acc
            t_acc_t+=acc_t'''

        acc, total = calc_acc_with_ref(batch, all_decoded)
    loss = np.average(loss_all)

    print("acc:{}, acc_T:{}\n".format(t_acc / t_all, t_acc_t / t_all))

    return loss

def eval_best_ckpt(hps):
    checkpoint_basename = os.path.join(hps.output_dir, hps.model_name)
    eval_model, all_ckpt = create_eval_model(hps)
    best_eval_loss, best_ckpt = inf, None
    batcher = EvalData(hps)
    run_reward = hasattr(eval_model, 'run_reward') and callable(eval_model.run_reward)
    for each_ckpt in all_ckpt:
        logging.info('Start evaluate performance of ckpt: {}'.format(each_ckpt))
        batcher.restart()
        eval_model.load_specific_model(each_ckpt)

        loss_all = []
        step = 0
        while True:
            batch = batcher.next_batch()
            step += 1
            if batch is None:
                break
            batch.feed_draft = np.array(True, dtype=np.bool)
            if batch.source_ids.shape[0] != hps.eval_batch_size:
                logging.warning('Purge {} samples, '
                                'as sample num != eval_batch_size ({})'.format(batch.source_ids.shape[0],
                                                                               hps.eval_batch_size))
                continue
            if step % 100 == 0:
                logging.info('step {}, avg loss: {}\n'.format(step, sum(loss_all) / len(loss_all)))
            if run_reward:
                batch_pred_ids = eval_model.run_reward(batch)['pred_ids']  # [b, length]
                batch.reward = calculate_reward(batch_pred_ids, batch, hps.padId)
            results = eval_model.run_dev_step(batch)
            loss_all.append(results['loss'])

        loss = np.average(loss_all)
        if loss < best_eval_loss:
            best_eval_loss = loss
            best_ckpt = each_ckpt
        logging.info('ckpt: {}, dev loss: {}, best loss: {}'.format(each_ckpt, loss, best_eval_loss))
    # load and save best model
    eval_model.load_specific_model(best_ckpt)
    eval_model.save_model(checkpoint_basename, True)


def main(_):
    FLAGS = config.parse_args()
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    if FLAGS.log_file:
        #if not os.path.exists(os.path.dirname(FLAGS.log_file)): os.mkdir(os.path.dirname(FLAGS.log_file))
        handler = logging.FileHandler(FLAGS.log_file, 'a', 'utf-8')
        handler.setFormatter(formatter)
        handlers = [handler]
    else:
        handlers = None
    logging.basicConfig(level=logging.INFO,
                        format=basic_format,
                        handlers=handlers)
    logging.info('-' * 80 + '[' + FLAGS.mode + ']' + '-' * 80)
    logging.info('Starting seq2seq_attention in %s mode...', FLAGS.mode)
    #logging.info(pprint_params(FLAGS))

    if FLAGS.mode == 'train':
        train(FLAGS)
    elif FLAGS.mode == 'eval':
        eval_best_ckpt(FLAGS)
    elif FLAGS.mode == 'dev':
        inference(FLAGS)
    elif FLAGS.mode == 'test':
        inference(FLAGS)
    else:
        raise ValueError('The `mode` flag must be one of train/dev/test')


if __name__ == '__main__':
    tf.app.run()
