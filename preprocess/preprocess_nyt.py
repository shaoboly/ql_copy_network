"""
Train and test sample ids file are generated from:
http://nlp.cs.berkeley.edu/downloads/train_abstracts_standoff.tgz
http://nlp.cs.berkeley.edu/downloads/eval_abstracts_standoff.tgz
"""
import argparse
import json
import os
import random
import re
import xml.etree.ElementTree as ETree

from tqdm import tqdm

data_path = '../data/nyt/original/'
train_file = '../data/nyt/train.txt'
valid_file = '../data/nyt/dev.txt'
test_file = '../data/nyt/test.txt'

train_ids_file = '../data/nyt/train_ids'
test_ids_file = '../data/nyt/test_ids'

article_xpath = './/block[@class="full_text"]'
abstract_xpath = './/abstract'
guid_xpath = './/doc-id'
date_xpath = './/pubdata'
test_set_date_time = '20070101'

test_sample_num = 3452

parser = argparse.ArgumentParser()
parser.add_argument('--from_txt', action='store_true',
                    help='preprocess from txt (instead of preprocessed original xml files)')
args = parser.parse_args()


class Sample(object):
    def __init__(self, guid, article, abstract, date):
        self.guid = guid
        self.article = article
        self.abstract = abstract
        self.date = date

    def __str__(self):
        return json.dumps({
            'guid': self.guid,
            'article': self.article,
            'abstract': self.abstract,
            'date': self.date
        })


def read_ids():
    train_ids, test_ids = set(), set()
    for line in open(train_ids_file, 'r', encoding='utf-8'):
        train_ids.update([line.strip()])
    for line in open(test_ids_file, 'r', encoding='utf-8'):
        test_ids.update([line.strip()])
    return train_ids, test_ids


def list_files(directory, extension):
    for (dir_path, dir_names, f_names) in os.walk(directory):
        for each_dir in dir_names:
            list_files(each_dir, extension)
        for f in f_names:
            if f.endswith('.' + extension):
                yield os.path.join(dir_path, f)


def read_from_json_files(directory):
    for f in ['train.txt', 'dev.txt', 'test.txt']:
        for line in open(os.path.join(directory, f), 'r', encoding='utf-8'):
            sample = json.loads(line.strip())
            yield Sample(abstract=sample['abstract'], guid=sample['guid'], article=sample['article'],
                         date=sample['date'])


def train_dev_split(train_data, valid_num):
    random.shuffle(train_data)
    return train_data[valid_num:], train_data[:valid_num]


def write_to_files(train_data, test_data):
    test_data = sorted(test_data, key=lambda a: len(a.article.split()))
    print('Write data to files...')
    with open(test_file, 'w', encoding='utf-8') as f_out:
        for i, each in enumerate(test_data):
            if i < test_sample_num:
                f_out.write(str(each))
                f_out.write('\n')
    valid_num = test_sample_num
    real_train_data, valid_data = train_dev_split(train_data, valid_num)
    print('Splitted train num: {}, valid num: {}'.format(len(real_train_data), len(valid_data)))
    with open(train_file, 'w', encoding='utf-8') as f_out:
        for each in real_train_data:
            f_out.write(str(each))
            f_out.write('\n')
    with open(valid_file, 'w', encoding='utf-8') as f_out:
        for each in valid_data:
            f_out.write(str(each))
            f_out.write('\n')


def change_abs(abstract: str):
    all_words = abstract.strip().split()
    all_words = ['0' if word.isdigit() else word for word in all_words]
    new_abs = ' '.join(all_words)
    new_abs = re.sub('\([s|m]\)', '', new_abs)
    new_abs = re.sub('; (photo|graph|chart|map|table|drawing)[s]?[ ]?$', '', new_abs)
    new_abs = new_abs.strip()
    return new_abs


def process():
    without_art, more_than_one_art, without_abs, without_guid_date = 0, 0, 0, 0
    all_num, abs_too_short = 0, 0
    train_data, test_data = [], []
    train_ids, test_ids = read_ids()
    print('Start pre-processing...')
    for i, data in tqdm(enumerate(read_from_json_files(data_path))):
        all_num += 1
        # 1. find abstract
        abstract = data.abstract
        abstract = change_abs(abstract)
        min_len = 50 if data.guid in train_ids else 50
        if len(abstract.split()) < min_len:
            abs_too_short += 1
            continue
        # 4. find date_time
        date_time = data.date
        # append
        if data.guid in train_ids:
            train_data.append(Sample(guid=data.guid, article=data.article, abstract=abstract, date=date_time))
        if data.guid in test_ids:
            test_data.append(Sample(guid=data.guid, article=data.article, abstract=abstract, date=date_time))
    # statistic
    print('-' * 50 + 'Statistics Of Data' + '-' * 50)
    print('Used sample num: {}, train num: {}, test num: {}, All num: {}'.format(len(train_data) + len(test_data),
                                                                                 len(train_data),
                                                                                 len(test_data),
                                                                                 all_num))
    print('Filtered sample: '
          '{} without abstract, '
          '{} with more than one article node, '
          '{} abstract short than 50 words, '
          '{} without article, '
          '{} without guid or datetime'.format(without_abs, more_than_one_art, abs_too_short, without_art,
                                               without_guid_date))
    write_to_files(train_data, test_data)
    print('Done.')


def process_from_xml():
    without_art, more_than_one_art, without_abs, without_guid_date = 0, 0, 0, 0
    all_num, abs_too_short = 0, 0
    train_ids, test_ids = read_ids()
    gen_train_ids, gen_test_ids = set(), set()
    train_data, test_data = [], []
    print('Start pre-processing...')
    for i, file in tqdm(enumerate(list_files(data_path, 'xml'))):
        all_num += 1
        data = ETree.parse(source=open(file, 'r', encoding='utf-8'))
        # 3. find guid
        guid_node = data.findall(guid_xpath)
        if not guid_node or len(guid_node) != 1:
            without_guid_date += 1
            continue
        guid = guid_node[0].get('id-string')
        if guid in train_ids:
            gen_train_ids.update([guid])
            min_len = 50
        elif guid in test_ids:
            gen_test_ids.update([guid])
            min_len = 50
        else:
            continue
        # find abstract
        abs_node = data.findall(abstract_xpath)
        abstract = ' '.join([each_p.text.lower() for each_p in abs_node[0]])
        abstract = change_abs(abstract)
        if len(abstract.split()) < min_len:
            abs_too_short += 1
            continue
        # find article
        article_node = data.findall(article_xpath)
        article = ' '.join([each_para.text.lower() for each_para in article_node[0]])
        # find date_time
        date_node = data.findall(date_xpath)
        date_time = date_node[0].get('date.publication')
        # append
        if guid in train_ids:
            train_data.append(Sample(guid=guid, article=article, abstract=abstract, date=date_time))
        if guid in test_ids:
            test_data.append(Sample(guid=guid, article=article, abstract=abstract, date=date_time))
    if train_ids != gen_train_ids:
        print('train_ids: {}, generated train_ids: {}'.format(len(train_ids), len(gen_train_ids)))
    if test_ids != gen_test_ids:
        print('test_ids: {}, generated test_ids: {}'.format(len(test_ids), len(gen_test_ids)))
    # statistic
    print('-' * 50 + 'Statistics Of Data' + '-' * 50)
    print('Used sample num: {}, train num: {}, test num: {}, All num: {}'.format(len(train_data) + len(test_data),
                                                                                 len(train_data),
                                                                                 len(test_data),
                                                                                 all_num))
    print('Filtered sample: '
          '{} without abstract, '
          '{} with more than one article node, '
          '{} abstract short than 50 words, '
          '{} without article, '
          '{} without guid or datetime'.format(without_abs, more_than_one_art, abs_too_short, without_art,
                                               without_guid_date))
    write_to_files(train_data, test_data)
    print('Done.')


if __name__ == '__main__':
    if not args.from_txt:
        process_from_xml()
    else:
        process()
