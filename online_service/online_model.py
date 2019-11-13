from inference import *

class inference_online():
    # Prepare dir
    def __init__(self):
        FLAGS = config.parse_args_my_flag()
        basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(basic_format)
        if FLAGS.log_file:
            if not os.path.exists(os.path.dirname(FLAGS.log_file)): os.mkdir(os.path.dirname(FLAGS.log_file))
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

        hps = FLAGS
        import tokenization
        self.tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file,
                                               do_lower_case=hps.do_lower_case)

        result_dir = os.path.join(hps.output_dir, hps.mode + '-results/')
        #ref_dir, decode_dir = os.path.join(result_dir, 'ref'+hps.test_file), \
        #                      os.path.join(result_dir, 'pred'+hps.test_file)
        ref_dir, decode_dir = os.path.join(result_dir, 'ref'), os.path.join(result_dir, 'pred')

        dec_dir_stage_1, dec_dir_stage_2 = decode_dir + '_1', decode_dir + '_2'
        trunc_dec_dir = os.path.join(result_dir, 'trunc_pred')

        abs2sents_func = abstract2sents_func(hps)

        # Load configs
        bert_config = modeling.BertConfig.from_json_file(hps.bert_config_file)
        model = model_pools[hps.model_name]
        processor = processors[hps.task_name.lower()]()

        validate_batcher = Batcher(processor, hps)

        # Build model graph
        dev_model = model(bert_config, validate_batcher, hps)
        dev_model.create_or_load_recent_model()

        # Prepare
        results_num = 0
        idx, skipped_num = 0, 0
        infer_type = determine_infer_type(dev_model)

        # build inference graph
        logging.info('Build inference graph...')
        pred_seq, _ = create_infer_op(dev_model, hps)
        fine_tuned_seq = create_infer_op_2(
            dev_model) if infer_type == InferType.two_step or infer_type == InferType.three_step else None
        sent_fine_tune = create_infer_op_sent(dev_model) if infer_type == InferType.three_step else None
        logging.info('Start inference...')
        res_dirs = []

        self.dev_model = dev_model
        self.hps = hps
        self.pred_seq = pred_seq





        '''while True:
            # predict one batch
            batch = dev_model.batcher.next_batch()
            if not batch:
                break
            if all_batch_already_decoded(ref_dir, decode_dir, idx, len(batch.source_ids)):
                idx += len(batch.source_ids)
                skipped_num += len(batch.source_ids)
                continue
            # inference ids seq
            if infer_type == InferType.single:
                ids_results = single_stage_model_inference(dev_model, pred_seq, batch, hps)
                res_dirs = [decode_dir]
            elif infer_type == InferType.three_step:
                ids_results = three_stage_model_inference(dev_model, pred_seq, fine_tuned_seq, sent_fine_tune, batch, hps)
                res_dirs = [dec_dir_stage_1, dec_dir_stage_2, decode_dir]
            else:
                ids_results = two_stage_model_inference(dev_model, pred_seq, fine_tuned_seq, batch, hps)
                res_dirs = [dec_dir_stage_1, decode_dir]
            # convert to string
            decode_result = [decode_target_ids(each_seq_ids, batch, hps) for each_seq_ids in ids_results]
            results_num += batch.true_num
            # save ref and label
            batch_summaries = [[sent.strip() for sent in abs2sents_func(each.summary)] for each in batch.original_data]
            idx = write_batch_for_rouge(batch_summaries, decode_result, idx, ref_dir, res_dirs, trunc_dec_dir,
                                        hps, batch.true_num)
            logging.info("Finished sample %d" % (results_num + skipped_num))

        logging.info('Start calculate ROUGE...')
        # calculate rouge and other metrics
        for i in range(len(res_dirs) - 1):
            results_dict = rouge_eval(ref_dir, res_dirs[i])
            rouge_log(results_dict, res_dirs[i])
        final_pred_dir = trunc_dec_dir if hps.task_name == 'nyt' else decode_dir
        results_dict = rouge_eval(ref_dir, final_pred_dir)
        rouge_log(results_dict, decode_dir)
        logging.info('Start fine tune the predictions...')
        fine_tune(hps)'''

    def infer_one_sentence(self,sentence):
        dev_model = self.dev_model
        pred_seq = self.pred_seq
        hps = self.hps

        #ref_batch = dev_model.batcher.next_batch()
        batch = self.dev_model.batcher.batch_from_sentence(sentence,self.tokenizer)

        ids_results = single_stage_model_inference(dev_model, pred_seq, batch, hps)
        decode_result = [decode_target_ids(each_seq_ids, batch, hps) for each_seq_ids in ids_results]

        out_line = " ".join(decode_result[0][0])

        out_line = out_line.strip().replace(' ##', '')
        return out_line

    def infer_all_candidates(self,sentence):
        dev_model = self.dev_model
        pred_seq = self.pred_seq
        hps = self.hps

        # ref_batch = dev_model.batcher.next_batch()

        batch = self.dev_model.batcher.batch_from_sentence(sentence, self.tokenizer)

        decode_seq,ids_results = single_stage_model_inference_topk(dev_model, pred_seq, batch, hps)
        #decode_result = [decode_target_ids(each_seq_ids, batch, hps) for each_seq_ids in ids_results]

        decode_result = decode_target_ids_beam(ids_results,batch, hps)

        all_lines = []
        out_line = " ".join(decode_result[0][0])

        for i in range(len(decode_result[0])):
            all_lines.append(" ".join(decode_result[0][i]))

        for i in range(len(all_lines)):
            all_lines[i] = all_lines[i].strip().replace(' ##', '')

        #out_line = out_line.strip().replace(' ##', '')
        return all_lines

online_model = inference_online()

