import argparse
import os
import re

import spacy

nlp = spacy.load('en')

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help='input wiki corpus directory')
parser.add_argument("--output", type=str, default='../data/en-corpus.txt', help='output file')
parser.add_argument("--percent", type=float, default=100, help='percent of corpus that will used by pre-train')
args = parser.parse_args()

ARTICLE_SEGMENT = '<SEG_ARTICLE>'
ALL_ARTICLE_NUM = 5542451
space = ' '


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path, 'r', encoding='utf-8'):
                    sline = line.strip()
                    if sline == "":
                        continue
                    if sline.startswith('<doc'):
                        yield ARTICLE_SEGMENT
                        continue
                    rline = cleanhtml(sline)
                    is_alpha_word_line = rline.lower()
                    yield is_alpha_word_line


def seg_paragraph(paragraph: str) -> list:
    return [s.text for s in nlp(paragraph).sents]


def transfer_data(f):
    fout = open(f, 'w', encoding='utf-8')
    data = []
    articles = 0
    next_is_title = True
    can_print = False
    all_corpus = MySentences(args.input)
    for paragraph_or_title in all_corpus:
        if articles % 10 == 0 and can_print:
            print('Read {} articles.'.format(articles))
            can_print = False
        if articles / ALL_ARTICLE_NUM * 100 > args.percent:
            break
        if paragraph_or_title == ARTICLE_SEGMENT:
            fout.write('\n')
            articles += 1
            can_print = True
            next_is_title = True
            continue
        if next_is_title:
            if paragraph_or_title.strip():
                # print(paragraph_or_title.strip())
                next_is_title = False
            continue
        for sent in seg_paragraph(paragraph_or_title.strip()):
            fout.write(sent + '\n')

    fout.close()
    return data


print('Transfer wiki data...')
data = transfer_data(args.output)
print('Done.')
