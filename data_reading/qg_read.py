import json
import os
import logging
from data_reading.summarization_read import SummarizeProcessor
from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess



class InputExample(object):
    def __init__(self, guid, article, summary,original=None):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary
        self.original = original


class SQUADProcessor(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

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
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = json.loads(line.strip())
            examples.append(InputExample(guid="1", article=data['article'], summary=data['abstract']))
        return examples


class BingQG(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

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
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = line.split('\t')
            try:
                data[0] = data[0].strip()
                data[1] = data[1].strip()
                examples.append(InputExample(guid="1", article=data[1], summary=data[0]))
            except:
                print(line)
        return examples


class SQUADProcessorAnswer():

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev-v1.1.json.refine'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev-v1.1.json.refine'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train-v1.1.json.refine'))

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
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = json.loads(line.strip())
            examples.append(InputExample(guid="1", article=data['answer']+" [ANS_SEQ] "+data['article'], summary=data['abstract']))
        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        context,answer = example.article.split('[ANS_SEQ]')
        context_pieces = tokenizer.tokenize(context)
        answer_pieces = tokenizer.tokenize(answer)
        answer_pieces.append(config.sep)
        article_pieces = answer_pieces+context_pieces

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


import copy
class NQQGProcessorAnswer():

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev-v1.1.json.refine'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev-v1.1.json.refine'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train-v1.1.json.refine'))

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
        examples = []

        for i,line in enumerate(open(file_path, 'r', encoding='utf-8', errors='ignore')):
            data = json.loads(line.strip())
            try:
                q = data['question']
            except:
                data['question']="no reference"
                q="no reference"
            answers = copy.deepcopy(data['answers'])
            d = data['document']
            for answer in answers:
                s, e = answer
                data["answers"] = [answer]
                answer = ' '.join(d.split()[s:e])
                examples.append(InputExample(guid=str(i), article=answer+" [ANS_SEQ] "+d, summary=q, original=data))
        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        context,answer = example.article.split('[ANS_SEQ]')
        context_pieces = tokenizer.tokenize(context)
        answer_pieces = tokenizer.tokenize(answer)
        answer_pieces.append(config.sep)
        article_pieces = answer_pieces+context_pieces

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



