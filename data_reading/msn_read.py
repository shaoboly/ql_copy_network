import json
import os

from data_reading.summarization_read import SummarizeProcessor
from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess
import logging

class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class MSNProcessor(SummarizeProcessor):

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
                src,tgt = line["body"],line["title"]
            except:
                continue

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples





class M2MKeyProcessor(SummarizeProcessor):

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
            line = line.split('\t')
            if len(line)==2:
                examples.append(InputExample(guid="0", article=line[0], summary=line[1]))
            else:
                continue

        return examples



class Paraphrase(SummarizeProcessor):

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
            line = line.split('\t')
            if len(line)>=2:
                examples.append(InputExample(guid="0", article=line[0], summary=line[1]))


        return examples


class ParaphraseTagsProcessor(SummarizeProcessor):

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
            line = line.strip().split('\t')
            input_all = " | ".join([line[0],line[2],line[3]])
            examples.append(InputExample(guid="0", article=input_all, summary=line[1]))

        return examples





class ParaphrasesrcTagsProcessor(SummarizeProcessor):

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
            line = line.strip().split('\t')
            input_all = " | ".join([line[0],line[2]])
            examples.append(InputExample(guid="0", article=input_all, summary=line[1]))

        return examples


class ParaphrasesrcTag2Tag(SummarizeProcessor):

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
            line = line.strip().split('\t')
            #input_all = " | ".join([line[0],line[2]])
            examples.append(InputExample(guid="0", article=line[2], summary=line[3]))

        return examples


class ParaphrasesrcTag2Tag_newvocab(SummarizeProcessor):

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
            line = line.strip().split('\t')
            #input_all = " | ".join([line[0],line[2]])
            examples.append(InputExample(guid="0", article=line[2], summary=line[3]))

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        article_pieces = example.article.strip().lower().split()
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
        summary_tokens = example.true_summary.strip().lower().split()
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


class ParaphraseSemanticTag2Tag(SummarizeProcessor):

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
            line = line.strip().split('\t')
            input_all = " | ".join([line[0],line[2]])
            examples.append(InputExample(guid="0", article=input_all, summary=line[3]))

        return examples



def parsing_tree_mask(ptree):
    ptree = ptree.replace('(',' ( ').replace(')',' ) ')
    ptree = ptree.split()

    mask_words = []
    terminal_words = ['(', ')', 'NP', 'VP', 'NN', 'ROOT', '.', 'SQ', 'IN', 'PP', 'SBARQ', 'DT', 'VB', 'NNP', 'S', 'JJ', 'NNS', 'WHNP', 'VBP', 'PRP', 'WRB', 'VBZ', 'WHADVP', 'WP', 'MD', 'SBAR', 'TO', 'RB',
                      'CC', 'PRP$', 'ADJP', 'CD', 'JJS', 'VBG', 'ADVP', 'VBN', 'WDT', 'VBD', ',', 'FRAG', 'POS', 'JJR', 'RBS', 'NNPS', 'EX', 'WHADJP', 'PRT', 'RP', 'PRN', '-RRB-', '-LRB-', ':', 'NP-TMP',
                      'RBR', 'SINV', 'QP', 'FW', 'X', '$', 'UCP', 'WHPP', 'PDT', 'NX', 'SYM', 'CONJP', 'LS', 'WP$', 'NAC', 'UH', 'RRC', 'INTJ', '#', 'LST', '-LRB-800-RRB-','``',"''"]
    for w in ptree:
        if w not in terminal_words:
            mask_words.append("[MASK]")
        else:
            mask_words.append(w)



    return " ".join(mask_words)

class ParaphraseParsingTreeProcessor(SummarizeProcessor):

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
            line = line.strip().split('\t')
            input_all = " | ".join([line[0],line[2],line[3]])
            examples.append(InputExample(guid="0", article=input_all, summary=line[1]))

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












class QIProcessor(SummarizeProcessor):

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
                src,tgt = line["article"],line["title"]
            except:
                continue

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples



class MSNProcessorQG():

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'nq_ra_an.tsv.test'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'nq_ra_an.tsv.test'))

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
                src,tgt,answer = line["body"],line["question"],line["answer"]

            except:
                continue

            examples.append(InputExample(guid="0", article=answer+' [ANS_SEQ] '+src, summary=tgt))

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        context, answer = example.article.split('[ANS_SEQ]')
        context_pieces = tokenizer.tokenize(context)
        answer_pieces = tokenizer.tokenize(answer)

        '''
        context_tokenized = " ".join(context_pieces)
        answer_tokenizaed = " ".join(answer_pieces)
        '''



        answer_pieces.append(config.sep)
        article_pieces = answer_pieces + context_pieces

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



class NewKeyMultiInput():

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
        for i,line in enumerate(open(file_path, 'r', encoding='utf-8', errors='ignore')):
            line = line.strip()
            line = line.split("\t")
            try:
                examples.append(InputExample(guid=str(i), article=line[0], summary=line[1]))
            except:
                examples.append(InputExample(guid=str(i), article=line[0], summary="."))

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        query_all = example.article.split('|')
        article_pieces = []
        for i in range(len(query_all)):
            if i!=0:
                article_pieces.append(config.sep)
            article_pieces+=tokenizer.tokenize(query_all[i])


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



class CalloutRead():

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
        for i,line in enumerate(open(file_path, 'r', encoding='utf-8', errors='ignore')):
            line = json.loads(line)

            input = line["adtitle"]+"[SEP]"+line["desc1"]+"[SEP]"+line["desc2"]+"[SEP]"+line["lfcontent"]
            examples.append(InputExample(guid=str(i), article=input, summary=line["call_out"]))


        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # article
        query_all = example.article.split('[SEP]')
        article_pieces = []
        for i in range(len(query_all)):
            if i!=0:
                article_pieces.append(config.sep)
            article_pieces+=tokenizer.tokenize(query_all[i])


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