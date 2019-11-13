import glob
import json
import os
import re
from collections import defaultdict, OrderedDict

import rouge
from matplotlib import ticker
from tqdm import tqdm
import tensorflow as tf

import config
import tokenization
from data_reading import processors

pg_output_dir = 'data/pg_test_output'
our_output_dir = 'data/test_results'
cache_res_file_ours = 'data/cache_res.sum'
cache_res_file_pg = 'data/cache_res_pg.sum'
cache_res_file_lead_3 = 'data/cache_res_lead_3.sum'

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                        max_n=2,
                        limit_length=False,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=True,
                        apply_best=False,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


def main(_):
    os.chdir('..')
    hps = config.parse_args()

    processor = processors[hps.task_name.lower()]()
    examples = processor.get_test_examples(hps.data_dir)

    test_set_num = 11490

    our_pred_dir, our_ref_dir = os.path.join(our_output_dir, 'pred'), os.path.join(our_output_dir, 'ref')
    pg_pred_dir, pg_ref_dir = os.path.join(pg_output_dir, 'pointer-gen-cov'), os.path.join(pg_output_dir, 'reference')

    file_num = 0

    print('Start read data.')

    our_preds = read_summary(our_pred_dir)
    our_refs = read_summary(our_ref_dir)
    pg_preds = read_summary(pg_pred_dir)
    pg_refs = read_summary(pg_ref_dir)

    print('Read data done.')
    tokenizer = tokenization.FullTokenizer(vocab_file=hps.vocab_file, do_lower_case=hps.do_lower_case)

    our_rouge_res, pg_rouge_res, lead_3_rouge_res = read_res_from_cache()
    if not (our_rouge_res and pg_rouge_res):
        our_rouge_res = defaultdict(list)
        pg_rouge_res = defaultdict(list)
        lead_3_rouge_res = defaultdict(list)
        for file_num in tqdm(range(test_set_num)):
            example = examples[file_num]
            article = example.article
            article = process_article(article, tokenizer)
            pred_lead_3 = find_lead_3_sents(article)

            n_th_rouge_lead_3 = calculate_rouge(pred_lead_3, our_refs[file_num][0])
            n_th_rouge_our = calculate_rouge(our_preds[file_num][0], our_refs[file_num][0])
            n_th_rouge_pg = calculate_rouge(pg_preds[file_num][0], pg_refs[file_num][0])
            ref_len_our = len(' '.join(our_refs[file_num][0]).split())
            ref_len_pg = len(' '.join(pg_refs[file_num][0]).split())
            our_rouge_res[ref_len_our].append(n_th_rouge_our)
            pg_rouge_res[ref_len_pg].append(n_th_rouge_pg)
            lead_3_rouge_res[ref_len_our].append(n_th_rouge_lead_3)

        write_cache_res(our_rouge_res, pg_rouge_res, lead_3_rouge_res)
        print('Total {} samples.'.format(file_num))

    statistic(our_rouge_res, pg_rouge_res, lead_3_rouge_res)


def find_lead_3_sents(article: str):
    sents = re.split('[.?!]', article)
    return [sent.strip() + ' .' for sent in sents[:3]]


def process_article(article: str, tokenizer):
    article = ' '.join(tokenizer.tokenize(article)).replace(' ##', '')
    return article


def calculate_rouge(pred, ref):
    res = evaluator.get_scores([' '.join(pred)], [' '.join(ref)])
    return {
        'rouge-1': res['rouge-1']['f'],
        'rouge-2': res['rouge-2']['f'],
        'rouge-l': res['rouge-l']['f']
    }


def write_cache_res(our_rouge_res, pg_rouge_res, lead_3_rouge_res):
    with open(cache_res_file_ours, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(our_rouge_res))
    with open(cache_res_file_pg, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(pg_rouge_res))
    with open(cache_res_file_lead_3, 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(lead_3_rouge_res))


def read_res_from_cache():
    if os.path.exists(cache_res_file_ours) and os.path.exists(cache_res_file_pg) and os.path.exists(
            cache_res_file_lead_3):
        res = json.load(open(cache_res_file_ours, 'r', encoding='utf-8'))
        res_pg = json.load(open(cache_res_file_pg, 'r', encoding='utf-8'))
        res_lead_3 = json.load(open(cache_res_file_lead_3, 'r', encoding='utf-8'))
        return {int(k): v for k, v in res.items()}, {int(k): v for k, v in res_pg.items()}, {int(k): v for k, v in
                                                                                             res_lead_3.items()}
    return None, None, None


def statistic(our_rouge_res, pg_rouge_res, lead_3_rouge_res, res_metric='rouge-l'):
    our_res = {}
    pg_res = {}
    lead_3_res = {}
    metrices = ['rouge-1', 'rouge-2', 'rouge-l']
    for each_len, all_sample_res in our_rouge_res.items():
        res = defaultdict(int)
        for each_sample_res in all_sample_res:
            for metric in metrices:
                res[metric] += each_sample_res[metric]
        our_res[each_len] = [res[res_metric], len(all_sample_res)]

    for each_len, all_sample_res in lead_3_rouge_res.items():
        res = defaultdict(int)
        for each_sample_res in all_sample_res:
            for metric in metrices:
                res[metric] += each_sample_res[metric]
        lead_3_res[each_len] = [res[res_metric], len(all_sample_res)]

    for each_len, all_sample_res in pg_rouge_res.items():
        res = defaultdict(int)
        for each_sample_res in all_sample_res:
            for metric in metrices:
                res[metric] += each_sample_res[metric]
        pg_res[each_len] = [res[res_metric], len(all_sample_res)]
    import matplotlib.pyplot as plt

    step = 10
    our_res = merge_to_bins(our_res, step)
    pg_res = merge_to_bins(pg_res, step)
    lead_3_res = merge_to_bins(lead_3_res, step)

    diff = show_diff(our_res, pg_res)
    diff_2 = show_diff(our_res, lead_3_res)

    # fig = plt.gcf()
    # width = 7
    # plt.bar(diff.keys(), diff.values(), width, color='green')
    # plt.bar(diff_2.keys(), diff_2.values(), width, color='red')
    # plt.bar(our_res.keys(), our_res.values(), width, color='green')
    # plt.bar(pg_res.keys(), pg_res.values(), width, color='blue')
    # plt.ylim(0.2, 0.5)
    # plt.show()

    to_pd_and_draw(diff, diff_2, step)


def to_pd_and_draw(diff, diff_2, step):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    lens, np_diff, np_diff_2 = [], [], []
    colors = []
    ori_lens = []
    for length in sorted(diff.keys()):
        if length == 15:
            lens.append('<20')
        elif length == 105:
            lens.append(' >=100')
        else:
            lens.append(str(length - step // 2) + '-' + str(length + step // 2))
        ori_lens.append(length)
        colors.append('light_blue')
        np_diff.append(diff[length])
        np_diff_2.append(diff_2[length])
    np_lens = np.array(lens, dtype=np.str)
    np_origin_lens = np.array(ori_lens, dtype=np.int32)
    np_diff = np.array(np_diff, dtype=np.float32)
    np_diff_2 = np.array(np_diff_2, dtype=np.float32)
    table = pd.DataFrame()
    table['length'] = np_lens
    table['ori_len'] = np_origin_lens
    table['Improvement_on_PG'] = np_diff
    table['Improvement_on_Lead_3'] = np_diff_2

    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    col_list_1 = [np.array([1, 1, 1, 1]) for x in [0.80, 0.72, 0.54, 0.45, 0.32, 0.32, 0.45, 0.54, 0.72, 0.80]]
    col_list_2 = [np.array([1, 1, 1, 1]) for x in [0.80, 0.72, 0.54, 0.45, 0.32, 0.32, 0.45, 0.54, 0.72, 0.80]]
    # col = sns.light_palette('red')
    # col = col_list
    bar = sns.barplot(y='Improvement_on_Lead_3', x='length', facecolor='white', data=table, ax=ax1, edgecolor='black')
    hatch_1 = '\\' * 2
    for this_bar in bar.patches:
        this_bar.set_width(this_bar.get_width() * 2 / 3)
        this_bar.set_hatch(hatch_1)

    for spine in ax1.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax1.set(xlabel='Golden length', ylabel='Performance against PG')

    bar = sns.barplot(y='Improvement_on_PG', x='length', facecolor='white', data=table, ax=ax2, edgecolor='black')
    hatch_2 = 'x' * 2
    for this_bar in bar.patches:
        this_bar.set_width(this_bar.get_width() * 2 / 3)
        this_bar.set_hatch(hatch_2)
    for spine in ax2.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax2.set(xlabel='Golden length', ylabel='Performance against Lead-3')

    sns.set_context("poster")

    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontsize(12) for label in labels]
    # [label.set_horizontalalignment('center') for label in labels]
    ax1.yaxis.label.set_size(12)
    ax1.xaxis.label.set_size(12)
    ax2.yaxis.label.set_size(12)
    ax2.xaxis.label.set_size(12)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    plt.subplots_adjust(wspace=0, hspace=0.3)
    # plt.show()
    figure_fig = plt.gcf()  # 'get current figure'
    figure_fig.savefig('res_on_diff_len.png', format='png', dpi=1000)


def show_diff(our_res, pg_res):
    res = OrderedDict()
    for i in our_res.keys():
        res[i] = our_res[i] - pg_res[i] if our_res[i] != 0 else 0
    return res


def merge_to_bins(res, step=10):
    new_res = defaultdict(list)
    r = {}
    import numpy as np
    bins = np.arange(10, 110, step)
    # bins = [0, 75, 100, 125]
    for lens, v in res.items():
        for i, bin_start in enumerate(bins):
            if bin_start <= lens and (len(bins) <= i + 1 or bins[i + 1] > lens):
                new_res[bin_start + step // 2].append(v)

    for k, v in new_res.items():
        total_num, total_rouge = 0, 0
        for each in v:
            total_num += each[1]
            total_rouge += each[0]
        r[k] = total_rouge / total_num
    return r


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
                res[guid] = [summary_sents, file]
            except:
                continue
        with open(cache_file, 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(res))
        return res


if __name__ == '__main__':
    tf.app.run()
