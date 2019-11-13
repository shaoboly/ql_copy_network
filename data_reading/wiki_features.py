import logging

from model_pools.model_utils.copy_mechanism import copy_mechanism_preprocess
import random

def mask_ngram_one_line(cur_line):
    tokens = cur_line.split()
    all_length = range(1, 5)
    i = 0
    new_tokens = []
    mask = '[MASK]'
    while i < len(tokens):
        if random.random() < 0.1:
            num_n = random.sample(all_length, 1)[0]
            new_tokens.extend([mask for j in range(num_n)])
            i += num_n
        else:
            new_tokens.append(tokens[i])
            i += 1
    new_line = " ".join(new_tokens)
    return new_line

import random
def mask_ngram(article_tokens,tokenizer):
    all_length = range(1, 5)
    i = 0
    new_tokens = []
    mask = '[MASK]'
    while i < len(article_tokens):
        if random.random() < 0.1 and i+10<len(article_tokens):
            num_n = random.sample(all_length, 1)[0]
            new_tokens.extend([mask for j in range(num_n)])
            i += num_n
        else:
            new_tokens.append(article_tokens[i])
            i += 1
    assert len(new_tokens)==len(article_tokens)
    return new_tokens

def denoising_tokens(article_tokens):
    all_length = range(1, 5)
    i = 0
    new_tokens = []
    mask = '[MASK]'
    while i < len(article_tokens):
        rand_score = random.random()
        if rand_score < 0.05:
            num_n = random.sample(all_length, 1)[0]
            new_tokens.extend([article_tokens[i] for j in range(num_n)])
        elif rand_score <0.1:
            pass
        else:
            new_tokens.append(article_tokens[i])
        i += 1
    return new_tokens

class WikiFeatures(object):
    def get_train_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
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


def mask_target_id(input_ids,tokenizer):
    all_length = range(1, 5)
    i = 0
    new_tokens = []
    mask = tokenizer.vocab['[MASK]']
    while i < len(input_ids):
        if random.random() < 0.1 and i + 10 < len(input_ids):
            num_n = random.sample(all_length, 1)[0]
            new_tokens.extend([mask for j in range(num_n)])
            i += num_n
        else:
            new_tokens.append(input_ids[i])
            i += 1
    assert len(new_tokens) == len(input_ids)
    return new_tokens


class WikiFeaturesEncoderDecdoer(object):
    def get_train_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
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

        #mask input
        summary_ids = mask_target_id(summary_ids,tokenizer)

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




class WikiFeaturesEncoderDecdoerNoise(object):
    def get_train_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
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
        summary_tokens.append(config.pad)

        input_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)
        summary_len = len(summary_ids)
        input_len = len(input_ids)

        input_ids_oo, src_oovs, summary_ids_oo = copy_mechanism_preprocess(article_tokens, summary_tokens,
                                                                           config, tokenizer.vocab)

        #mask input
        summary_ids = mask_target_id(summary_ids,tokenizer)

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



class WikiFeaturesMulti(object):
    def get_train_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,fname=None):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def convert_example_to_feature(example, tokenizer, config,lf_tokenizer=None):
        """Turn data to binary format, takes about one hour on CNN/DM train set."""

        if example.guid=="1":
            # summary
            summary_tokens = tokenizer.tokenize(example.true_summary)
            summary_tokens = ["[unused11]"] + summary_tokens
            summary_tokens.append(config.pad)

            # article
            split_article = example.article.split("[masked_sentence]")
            article_pieces = tokenizer.tokenize(split_article[0])+["[unused20]"]\
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

        if example.guid=="0":
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