import os
import time
import tensorflow as tf
import beam_search
import data
import json
# import pyrouge
import util
import logging
import numpy as np
import codecs
import matrix

# from beam_search import Hypothesis

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, len, start_decode_Flag = None):
        """Hypothesis constructor.

        Args:
          tokens: List of integers. The ids of the tokens that form the summary so far.
          log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
          coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage
        self.len = len

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, len):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage,
                          len=len)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)

class Candidate_batch():
    def __init__(self,tokens, state , scores ,len):
        self.tokens = tokens
        self.last_states = state
        self.logScores = scores
        self.len = len


class EvalDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
        """
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab
        self._saver = self._model.saver  # we use this to load checkpoints for decoding
        self._sess = self._model.gSess_train

        # Load an initial checkpoint to use for decoding
        # ckpt_path = util.load_ckpt(self._saver, self._sess)

        # if FLAGS.single_pass:
        #  # Make a descriptive decode directory name
        #  ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
        #  self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
        #  if os.path.exists(self._decode_dir):
        #    raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

        # else: # Generic decode dir name
        #  self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        ## Make the decode dir if necessary
        # if not os.path.exists(self._decode_dir): os.mkdir(self._decode_dir)

        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
            FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        self._saver.restore(self._sess, ckpt_path)
        # if FLAGS.single_pass:
        #  # Make the dirs to contain output written in the correct format for pyrouge
        #  self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        #  if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)
        #  self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        #  if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)

    def decode(self):
        """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
        t0 = time.time()
        counter = 0

        f = os.path.join(FLAGS.log_root, "output.txt")
        # print("----------------"+f)
        outputfile = codecs.open(f, "w", "utf8")
        output_result = []
        list_of_reference = []
        while True:
            batch = self._batcher.next_batch()  # 1 example repeated across batch
            if batch is None:  # finished decoding dataset in single_pass mode
                logging.info("eval_finished")
                outputfile.close()
                break
            print(self._batcher.c_index)
            original_article = batch.original_articles[0]  # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

            article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string

            # Run beam search to get best Hypothesis
            result = self.eval_one_batch(self._sess, self._model, self._vocab, batch)


            for i,instance in enumerate(result):
                if i == len(batch.art_oovs):
                    break
                out_words = data.outputids2words(instance, self._model._vocab_out, batch.art_oovs[i])
                if data.STOP_DECODING in out_words:
                    out_words = out_words[:out_words.index(data.STOP_DECODING)]
                    output_now = " ".join(out_words)
                    output_result.append(output_now)
                    # refer = " ".join(refer)

                    refer = batch.original_abstracts[i].strip()
                    list_of_reference.append([refer])

                    outputfile.write(batch.original_articles[i] + '\t' + batch.original_abstracts[i] + '\t' + output_now + '\n')

        bleu = matrix.bleu_score(list_of_reference, output_result)
        acc = matrix.compute_acc(list_of_reference, output_result)

        print("bleu : {}   acc : {}".format(bleu,acc))
        return

    def eval_one_batch(self,sess, model, vocab_out, batch):
        enc_states, dec_in_state = model.run_encoder_eval(sess, batch)

        # Initialize beam_size-many hyptheses

        results = []
        steps = 0
        latest_tokens = [vocab_out.word2id(data.START_DECODING) for i in range(FLAGS.batch_size)]  # latest token produced by each hypothesis

        pre_state = dec_in_state
        prev_coverage = None

        while steps < FLAGS.max_dec_steps:
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
            # Run one step of the decoder to get the new info
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=pre_state,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(steps == 0))

            topk_ids = np.reshape(topk_ids,[-1])
            results.append(topk_ids)

            pre_state = new_states
            latest_tokens = topk_ids

            steps+=1

        results = np.array(results).T
        return results

    def beam_search_eval(self,sess, model, vocab_out, batch):
        enc_states, dec_in_state = model.run_encoder_eval(sess, batch)

        # Initialize beam_size-many hyptheses

        results = []
        steps = 0


        latest_tokens = [vocab_out.word2id(data.START_DECODING) for i in
                         range(FLAGS.batch_size)]  # latest token produced by each hypothesis


        pre_state = dec_in_state
        prev_coverage = None

        while steps < FLAGS.max_dec_steps:
            latest_tokens = [t if t in range(vocab_out.size()) else vocab_out.word2id(data.UNKNOWN_TOKEN) for t in
                             latest_tokens]
            # Run one step of the decoder to get the new info
            (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                            batch=batch,
                                                                                                            latest_tokens=latest_tokens,
                                                                                                            enc_states=enc_states,
                                                                                                            dec_init_states=pre_state,
                                                                                                            prev_coverage=prev_coverage,
                                                                                                            first=(steps == 0))

            topk_ids = np.reshape(topk_ids, [-1])
            results.append(topk_ids)

            pre_state = new_states
            latest_tokens = topk_ids

            steps += 1

        results = np.array(results).T
        return results