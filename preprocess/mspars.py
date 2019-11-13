import hashlib
import json
import os
import struct
import sys
import time
import collections

class DataProcesser():
    def __init__(self):
        pass

    def extract_sample(self,inname):
        pass

    def extract_sample_pattern(self,inname):
        in_f = open(inname,encoding="utf-8").readlines()
        out_f = open(inname+".seq2seq","w",encoding="utf-8")
        i = 0
        while i<len(in_f):
            q = in_f[i].strip().lower().split('\t')[1]
            lf = in_f[i + 1].strip().lower().split('\t')[1]
            entity = in_f[i + 2].strip().lower().split('\t')[1]
            type = in_f[i + 3].strip().lower().split('\t')[1]

            lf_splits = lf.split()

            for en in entity.split('|||'):
                if len(en.split()) > 2:
                    print(en)
                    continue
                enn, ind = en.split()
                for k,w in enumerate(lf_splits):
                    if w==enn:
                        lf_splits[k]="entity"

            lf_p = " ".join(lf_splits)
            out_f.write("{}\t{}\n".format(q,lf_p))
            i+=5

def replace_lf2pattern(line):
    tokens = line.strip().split()

    st = "<es>"
    et = "<ee>"
    new_tokens = []
    jump = False
    for w in tokens:
        if w ==st:
            jump=True
            new_tokens.append("entity")
            continue
        if w ==et:
            jump=False
            continue
        if not jump:
            new_tokens.append(w)

    return " ".join(new_tokens)

def input2json(inname):
    SENTENCE_START = '<s>'
    SENTENCE_END = '</s>'
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".json","w",encoding="utf-8")
    for i,line in enumerate(in_f):
        q,l = line.strip().split('\t')[:2]

        l = replace_lf2pattern(l)

        l = "{} {} {}".format(SENTENCE_START,l,SENTENCE_END)

        out_line = json.dumps({
            'guid': i,
            'article': q,
            'abstract':l
        })
        out_f.write(out_line+'\n')

def convert_vocab(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".bert","w",encoding="utf-8")
    special_words = {
        'pad': '[PAD]',  # pad, as well as eos
        'unk': '[UNK]',
        'cls': '[CLS]',  # cls, as well as bos
        'sep': '[SEP]',  # to separate the sentence part
        'mask': '[MASK]'  # to mask the word during LM train
    }

    for k,w in special_words.items():
        out_f.write("{}\n".format(w))

    for i,line in enumerate(in_f):
        v,num = line.strip().split('\t')
        out_f.write(v+'\n')

from tokenization import LFTokenizer
def generate_vocab(innames):
    samples = []
    for inname in innames:
        in_f = open(inname,encoding="utf-8").readlines()
        for line in in_f:
            jsample = json.loads(line)
            samples.append(jsample)

    lf_tokenizer = LFTokenizer(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\vocab.txt.original.bert",True)
    counter = collections.Counter()
    for sample in samples:
        lf = sample["abstract"]
        #lf = lf.replace('<s>',"")
        #lf = lf.replace('</s>', "")
        lf_tokens = lf_tokenizer.tokenize(lf)
        for token in lf_tokens:
            counter[token]+=1

    counter = counter.most_common(10000)

    out_f = open(os.path.join(os.path.dirname(innames[0]),"vocab.out.bert"), "w", encoding="utf-8")
    special_words = {
        'pad': '[PAD]',  # pad, as well as eos
        'unk': '[UNK]',
        'cls': '[CLS]',  # cls, as well as bos
        'sep': '[SEP]',  # to separate the sentence part
        'mask': '[MASK]'  # to mask the word during LM train
    }

    for k, w in special_words.items():
        out_f.write("{}\n".format(w))
    for w in counter:
        out_f.write("{}\n".format(w[0]))


if __name__=="__main__":
    input2json(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Dev.seq2seq")
    input2json(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Test.seq2seq")
    input2json(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Train.seq2seq")
    #convert_vocab(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\vocab.out")

    generate_vocab([r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Train.seq2seq.json",
                    r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Dev.seq2seq.json",
                    r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS\seq2seq_pattern\MSParS.Test.seq2seq.json"])

    '''dp =DataProcesser()
    dp.extract_sample_pattern(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS.Train")
    dp.extract_sample_pattern(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS.Dev")
    dp.extract_sample_pattern(r"D:\data\research\final-v3.release\final-v3.release\final-v3.release\MSParS.Test")'''