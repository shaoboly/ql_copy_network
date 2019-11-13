import json
import os
import logging

from data_reading.summarization_read import SummarizeProcessor
from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess

class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class KeysProcessor(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.test'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.dev'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.train'))

    @staticmethod
    def abstract2sents(abstract: str):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        return [abstract.strip()]

    @staticmethod
    def _create_examples(file_path):
        print(file_path)
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            line = json.loads(line)
            try:
                src,tgt = line["title"],line["body"]
            except:
                continue

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples

class KeysProcessorMaskTest(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.test'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.dev'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'larger.txt.train'))

    @staticmethod
    def abstract2sents(abstract: str):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        return [abstract.strip()]

    @staticmethod
    def _create_examples(file_path):
        print(file_path)
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            line = json.loads(line)
            try:
                src,tgt = line["title"],line["body"]
            except:
                continue

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        article_pieces = tokenizer.tokenize(example.article)
        article_tokens = []
        segment_ids = []
        article_tokens.append(config.cls)
        segment_ids.append(0)
        for token in article_pieces:
            article_tokens.append(token)
            segment_ids.append(0)
        article_tokens.append(config.sep)
        segment_ids.append(0)

        # summary
        summary_tokens = tokenizer.tokenize(example.true_summary)
        summary_tokens.append(config.pad)

        input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
        summary_len = len(summary_ids)
        input_len = len(input_ids)

        input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                           config, tokenizer.vocab)
        assert len(input_ids) == len(input_ids_oo)
        assert len(summary_ids) == len(summary_ids_oo)

        """logging.info('*** Example ***')
        logging.info('guid: %s' % example.guid)
        logging.info('article: %s' % (' '.join(article_tokens)))
        logging.info('summary: %s' % (' '.join(summary_tokens)))
        logging.info('input_ids: %s' % (' '.join([str(x) for x in input_ids])))
        logging.info('input_ids_oo: %s' % (' '.join([str(x) for x in input_ids_oo])))
        logging.info('summary_ids: %s' % (' '.join([str(x) for x in summary_ids])))
        logging.info('summary_ids_oo: %s' % (' '.join([str(x) for x in summary_ids_oo])))
        logging.info('src oovs: %s' % (' '.join([str(x) for x in src_oovs])))
        logging.info('input_len: %d' % input_len)
        logging.info('summary_len: %d' % summary_len)
        logging.info('segment_ids: %s' % (' '.join([str(x) for x in segment_ids])))"""

        feature = {
            'origin_sample': example,
            'article_ids': input_ids,
            'article_ids_oo': input_ids_oo,
            'article_lens': input_len,
            'article_seg_ids': segment_ids,
            'summary_ids': summary_ids,
            'summary_ids_oo': summary_ids_oo,
            'summary_lens': summary_len,
            'src_oovs': src_oovs
        }
        return feature

    @staticmethod
    def filter_examples(examples):
        """# See https://github.com/abisee/pointer-generator/issues/1"""
        return [example for example in examples if len(example.article) != 0]

    @staticmethod
    def log_statistics(examples):
        logging.info('Data Samples: {}'.format(len(examples)))
        max_target_len = max(len(sample.true_summary.split()) for sample in examples)
        max_source_len = max(len(sample.article.split()) for sample in examples)
        mean_target_len = sum(len(sample.true_summary.split()) for sample in examples) / len(examples)
        mean_source_len = sum(len(sample.article.split()) for sample in examples) / len(examples)
        logging.info('Max article length is {}'.format(max_source_len))
        logging.info('Max summary length is {}'.format(max_target_len))
        logging.info('Mean article length is {}'.format(mean_source_len))
        logging.info('Mean summary length is {}'.format(mean_target_len))