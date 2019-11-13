import json
import os
import logging

from data_reading.summarization_read import SummarizeProcessor
from data_reading.wiki_features import WikiFeatures,WikiFeaturesEncoderDecdoer,WikiFeaturesEncoderDecdoerNoise,WikiFeaturesMulti
from data_reading.wiki_features import *

class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class WikiProcessor(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=cur_line, summary=cur_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples

class WikiProcessorRevsese(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))
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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    init_curline = cur_line
                    cur_line = cur_line.split()
                    cur_line.reverse()
                    cur_line = " ".join(cur_line)
                    examples.append(InputExample(guid="0", article=init_curline, summary=cur_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples

import random


def refine_sentence(all_line):
    total_length = 0
    new_all_line = []
    for line in all_line:
        total_length += len(line.split())
        if total_length > 512:
            break
        new_all_line.append(line)
    return " ".join(new_all_line)

class WikiProcessorNgramMask(WikiFeatures):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))


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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                cur_line = refine_sentence(all_line)
                #cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    new_line = cur_line
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=new_line, summary=cur_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples


class WikiProcessorNgramMaskEncoderDecoder(WikiFeaturesEncoderDecdoer):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))


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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                cur_line = refine_sentence(all_line)
                #cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    new_line = cur_line
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=new_line, summary=cur_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples


class WikiProcessorNgramMaskEncoderDecoderDenoise(WikiFeaturesEncoderDecdoerNoise):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                cur_line = refine_sentence(all_line)
                #cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    new_line = cur_line
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=new_line, summary=cur_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples


from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess
import random

def refine_sentence_and_sample_one(all_line):
    total_length = 0
    new_all_line = []
    for line in all_line:
        total_length += len(line.split())
        if total_length > 512:
            break
        new_all_line.append(line)
    mask_sent_idx = random.sample(range(len(new_all_line)),1)[0]
    return new_all_line,mask_sent_idx


class WikiPassage2Sent(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                if len(all_line)==0:
                    continue
                all_line,sent_idx = refine_sentence_and_sample_one(all_line)
                target_sent = all_line[sent_idx]
                all_line[sent_idx] = "[masked_sentence]"
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=cur_line, summary=target_sent))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # summary
        summary_tokens = tokenizer.tokenize(example.true_summary)
        masked_summary = ['[MASK]' for i in range(len(summary_tokens))]
        summary_tokens.append(config.pad)

        # article
        split_article = example.article.split("[masked_sentence]")
        article_pieces = tokenizer.tokenize(split_article[0])+masked_summary\
                         +tokenizer.tokenize(split_article[1])

        article_tokens = []
        segment_ids = []
        article_tokens.append(config.cls)
        segment_ids.append(0)
        for token in article_pieces:
            article_tokens.append(token)
            segment_ids.append(0)
        article_tokens.append(config.sep)
        segment_ids.append(0)


        input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
        summary_len = len(summary_ids)
        input_len = len(input_ids)

        input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                           config, tokenizer.vocab)
        assert len(input_ids) == len(input_ids_oo)
        assert len(summary_ids) == len(summary_ids_oo)

        # logging.info('*** Example ***')
        # logging.info('guid: %s' % example.guid)
        # logging.info('article: %s' % (' '.join(article_tokens)))
        # logging.info('summary: %s' % (' '.join(summary_tokens)))
        # logging.info('input_ids: %s' % (' '.join([str(x) for x in input_ids])))
        # logging.info('input_ids_oo: %s' % (' '.join([str(x) for x in input_ids_oo])))
        # logging.info('summary_ids: %s' % (' '.join([str(x) for x in summary_ids])))
        # logging.info('summary_ids_oo: %s' % (' '.join([str(x) for x in summary_ids_oo])))
        # logging.info('src oovs: %s' % (' '.join([str(x) for x in src_oovs])))
        # logging.info('input_len: %d' % input_len)
        # logging.info('summary_len: %d' % summary_len)
        # logging.info('segment_ids: %s' % (' '.join([str(x) for x in segment_ids])))

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


class WikiMultiTask(WikiFeaturesMulti):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                if len(all_line)==0:
                    continue

                #passage denoising
                cur_line = refine_sentence(all_line)
                # cur_line = " ".join(all_line)
                if len(cur_line.split()) > 0:
                    new_line = cur_line
                    # print(cur_line)
                    examples.append(InputExample(guid="0", article=new_line, summary=cur_line))


                #passage2sentence
                all_line,sent_idx = refine_sentence_and_sample_one(all_line)
                target_line = " ".join(all_line)
                #target_sent = all_line[sent_idx]
                all_line[sent_idx] = "[masked_sentence]"
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    examples.append(InputExample(guid="1", article=cur_line, summary=target_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples


class WikiMultiTaskP2S(WikiFeaturesMulti):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                if len(all_line)==0:
                    continue

                #passage denoising
                cur_line = refine_sentence(all_line)
                # cur_line = " ".join(all_line)
                if len(cur_line.split()) > 0:
                    new_line = cur_line
                    # print(cur_line)
                    examples.append(InputExample(guid="0", article=new_line, summary=cur_line))


                #passage2sentence
                all_line,sent_idx = refine_sentence_and_sample_one(all_line)
                target_line = all_line[sent_idx]
                #target_sent = all_line[sent_idx]
                all_line[sent_idx] = "[masked_sentence]"
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    examples.append(InputExample(guid="1", article=cur_line, summary=target_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""

        if example.guid == "1":
            # summary
            summary_tokens = tokenizer.tokenize(example.true_summary)
            summary_tokens = ["[unused11]"] + summary_tokens
            masked_summary = ['[MASK]' for i in range(len(summary_tokens))]
            summary_tokens.append(config.pad)

            # article
            split_article = example.article.split("[masked_sentence]")
            article_pieces = tokenizer.tokenize(split_article[0]) + masked_summary \
                             + tokenizer.tokenize(split_article[1])

            article_tokens = []
            segment_ids = []
            article_tokens.append(config.cls)
            segment_ids.append(0)
            for token in article_pieces:
                article_tokens.append(token)
                segment_ids.append(0)
            article_tokens.append(config.sep)
            segment_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
            summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
            summary_len = len(summary_ids)
            input_len = len(input_ids)

            input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                               config, tokenizer.vocab)
            assert len(input_ids) == len(input_ids_oo)
            assert len(summary_ids) == len(summary_ids_oo)

            # logging.info('*** Example ***')
            # logging.info('guid: %s' % example.guid)
            # logging.info('article: %s' % (' '.join(article_tokens)))
            # logging.info('summary: %s' % (' '.join(summary_tokens)))
            # logging.info('input_ids: %s' % (' '.join([str(x) for x in input_ids])))
            # logging.info('input_ids_oo: %s' % (' '.join([str(x) for x in input_ids_oo])))
            # logging.info('summary_ids: %s' % (' '.join([str(x) for x in summary_ids])))
            # logging.info('summary_ids_oo: %s' % (' '.join([str(x) for x in summary_ids_oo])))
            # logging.info('src oovs: %s' % (' '.join([str(x) for x in src_oovs])))
            # logging.info('input_len: %d' % input_len)
            # logging.info('summary_len: %d' % summary_len)
            # logging.info('segment_ids: %s' % (' '.join([str(x) for x in segment_ids])))

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

        if example.guid == "0":
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

            article_tokens = mask_ngram(article_tokens, tokenizer)
            article_tokens = denoising_tokens(article_tokens)
            article_tokens.append(config.sep)
            segment_ids.append(0)

            # summary
            summary_tokens = tokenizer.tokenize(example.true_summary)
            summary_tokens = ["[unused10]"] + summary_tokens
            summary_tokens.append(config.pad)

            input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
            summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
            summary_len = len(summary_ids)
            input_len = len(input_ids)

            input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                               config, tokenizer.vocab)

            # mask input
            summary_ids = mask_target_id(summary_ids, tokenizer)

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


class WikiPassageMS2Sent(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip()
            if len(line.split())==0:
                if len(all_line)==0:
                    continue
                all_line,sent_idx = refine_sentence_and_sample_one(all_line)
                target_line = " ".join(all_line)
                #target_sent = all_line[sent_idx]
                all_line[sent_idx] = "[masked_sentence]"
                cur_line = " ".join(all_line)
                if len(cur_line.split())>0:
                    #print(cur_line)
                    examples.append(InputExample(guid="0", article=cur_line, summary=target_line))
                all_line = []
                continue
            line = " ".join(line.split())
            all_line.append(line)

        return examples

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config, lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""
        # summary
        summary_tokens = tokenizer.tokenize(example.true_summary)
        masked_summary = ['[MASK]' for i in range(len(summary_tokens))]
        summary_tokens.append(config.pad)

        # article
        split_article = example.article.split("[masked_sentence]")
        article_pieces = tokenizer.tokenize(split_article[0])+masked_summary\
                         +tokenizer.tokenize(split_article[1])

        article_tokens = []
        segment_ids = []
        article_tokens.append(config.cls)
        segment_ids.append(0)
        for token in article_pieces:
            article_tokens.append(token)
            segment_ids.append(0)
        article_tokens.append(config.sep)
        segment_ids.append(0)


        input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
        summary_len = len(summary_ids)
        input_len = len(input_ids)

        input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                           config, tokenizer.vocab)
        assert len(input_ids) == len(input_ids_oo)
        assert len(summary_ids) == len(summary_ids_oo)

        # logging.info('*** Example ***')
        # logging.info('guid: %s' % example.guid)
        # logging.info('article: %s' % (' '.join(article_tokens)))
        # logging.info('summary: %s' % (' '.join(summary_tokens)))
        # logging.info('input_ids: %s' % (' '.join([str(x) for x in input_ids])))
        # logging.info('input_ids_oo: %s' % (' '.join([str(x) for x in input_ids_oo])))
        # logging.info('summary_ids: %s' % (' '.join([str(x) for x in summary_ids])))
        # logging.info('summary_ids_oo: %s' % (' '.join([str(x) for x in summary_ids_oo])))
        # logging.info('src oovs: %s' % (' '.join([str(x) for x in src_oovs])))
        # logging.info('input_len: %d' % input_len)
        # logging.info('summary_len: %d' % summary_len)
        # logging.info('segment_ids: %s' % (' '.join([str(x) for x in segment_ids])))

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


class WikiProcessorOnlyTarget(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'wikipedia_split_0101'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'merge_10p'))

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
        all_line = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip().split('\t')[1]

            examples.append(InputExample(guid="0", article="mask .", summary=line))

        return examples