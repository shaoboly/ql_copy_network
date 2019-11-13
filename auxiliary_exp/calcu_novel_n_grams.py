import glob
import json
import os
from collections import defaultdict

import tensorflow as tf
from nltk import ngrams
from tqdm import tqdm

import config
import tokenization
from data_reading import processors


def main(_):
    os.chdir('..')
    hps = config.parse_args()

    n = 4
    test_set_num = 11490

    total_result_pred = {i: 0 for i in range(1, n + 1)}
    total_result_ref = {i: 0 for i in range(1, n + 1)}
    result_dir = os.path.join(hps.output_dir, hps.mode + '-results/')
    ref_dir, pred_dir = os.path.join(result_dir, 'ref'), os.path.join(result_dir, 'pred')

    processor = processors[hps.task_name.lower()]()
    examples = processor.get_test_examples(hps.data_dir)
    file_num = 0

    print('Start read data.')

    preds = read_summary(pred_dir)
    refs = read_summary(ref_dir)

    total_pred_len = 0
    total_ref_len = 0

    print('Read data done.')
    tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file, do_lower_case=hps.do_lower_case)
    for file_num in tqdm(range(test_set_num)):
        example = examples[file_num]
        article = example.article
        article = process_article(article, tokenizer)
        total_pred_len += len(preds[file_num].split())
        total_ref_len += len(refs[file_num].split())
        article_n_grams = calcu_contain_n_gram(article, n)
        ref_n_grams = calcu_contain_n_gram(refs[file_num], n, keep_same=True)
        pred_n_grams = calcu_contain_n_gram(preds[file_num], n, keep_same=True)

        res_ref = calcu_novel_n_gram(article_n_grams, ref_n_grams, n)
        res_pred = calcu_novel_n_gram(article_n_grams, pred_n_grams, n)

        for i in range(1, n + 1):
            total_result_pred[i] += res_pred[i]
            total_result_ref[i] += res_ref[i]
        file_num += 1

    for i in range(1, n + 1):
        total_result_ref[i] /= file_num
        total_result_ref[i] *= 100
        total_result_pred[i] /= file_num
        total_result_pred[i] *= 100

    print('Total {} samples.'.format(file_num))
    print('ref NN:\n', total_result_ref)
    print('pred NN:\n', total_result_pred)


def process_article(article: str, tokenizer):
    article = ' '.join(tokenizer.tokenize(article)).replace(' ##', '')
    return article


# noinspection PyBroadException
def read_summary(directory):
    cache_file = os.path.join(directory, 'cache_summary.sum')
    if os.path.exists(cache_file):
        res = json.load(open(cache_file, 'r', encoding='utf-8'))
        return {int(k): v for k, v in res.items()}
    else:
        res = {}
        for file in glob.glob(directory + '/*.txt'):
            summary_sents = [line.strip() for line in open(file, 'r', encoding='utf-8')]
            try:
                guid = int(os.path.basename(file).split('_')[0])
                res[guid] = ' '.join(summary_sents)
            except:
                continue
        with open(cache_file, 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(res))
        return res


def calcu_contain_n_gram(article: str, n, keep_same=False):
    func = set if not keep_same else list
    return {i: func(ngrams(article.split(), i)) for i in range(1, n + 1)}


def calcu_novel_n_gram(article_n_grams, summary_n_grams, n):
    res = {}
    for i in range(1, n + 1):
        novel_i_gram = 0
        article_i_grams = article_n_grams[i]
        s_i_grams = summary_n_grams[i]
        times = defaultdict(int)
        for each_gram in s_i_grams:  # 计算summary中每个i_gram出现了多少次
            times[each_gram] += 1
        novel_i_grams = set(s_i_grams).difference(article_i_grams)  # i_gram in summary but not in article
        for novel_gram in novel_i_grams:
            novel_i_gram += times[novel_gram]
        res[i] = novel_i_gram / len(s_i_grams)  # percentage
    return res


if __name__ == '__main__':
    tf.app.run()
