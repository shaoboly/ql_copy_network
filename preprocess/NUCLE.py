#!/usr/bin/env python

import sys
import re


def preprocess_m2():
    input_path = r"D:\data\research\grammar_correct\release3.3\release3.3\data\conll14st-preprocessed.m2"
    output_src_path = r"D:\data\research\grammar_correct\release3.3\release3.3\data\src"
    output_tgt_path = r"D:\data\research\grammar_correct\release3.3\release3.3\data\tgt"

    words = []
    corrected = []
    sid = eid = 0
    prev_sid = prev_eid = -1
    pos = 0


    with open(input_path) as input_file, open(output_src_path, 'w') as output_src_file, open(output_tgt_path, 'w') as output_tgt_file:
        for line in input_file:
            line = line.strip()
            if line.startswith('S'):
                line = line[2:]
                words = line.split()
                corrected = ['<S>'] + words[:]
                output_src_file.write(line+'\t')
            elif line.startswith('A'):
                line = line[2:]
                info = line.split("|||")
                sid, eid = info[0].split()
                sid = int(sid) + 1
                eid = int(eid) + 1
                error_type = info[1]
                if error_type == "Um":
                    continue
                for idx in range(sid, eid):
                    corrected[idx] = ""
                if sid == eid:
                    if sid == 0: continue	# Originally index was -1, indicating no op
                    if sid != prev_sid or eid != prev_eid:
                        pos = len(corrected[sid-1].split())
                    cur_words = corrected[sid-1].split()
                    cur_words.insert(pos, info[2])
                    pos += len(info[2].split())
                    corrected[sid-1] = " ".join(cur_words)
                else:
                    corrected[sid] = info[2]
                    pos = 0
                prev_sid = sid
                prev_eid = eid
            else:
                target_sentence = ' '.join([word for word in corrected if word != ""])
                assert target_sentence.startswith('<S>'), '(' + target_sentence + ')'
                target_sentence = target_sentence[4:]
                output_src_file.write(target_sentence + '\n')
                prev_sid = -1
                prev_eid = -1
                pos = 0


#preprocess_m2()

def swap(sentence):
    sentence = sentence.strip().split()

    if len(sentence)<5:
        return " ".join(sentence)

    swap_index = random.sample(range(len(sentence)-3),1)[0]

    t = sentence[swap_index]
    sentence[swap_index] = sentence[swap_index+1]
    sentence[swap_index+1]=t

    return " ".join(sentence)


def drop(sentence):
    sentence = sentence.strip().split()
    if len(sentence)<5:
        return " ".join(sentence)

    drop_index = random.sample(range(len(sentence) - 2),1)[0]

    sentence = sentence[:drop_index]+sentence[drop_index+1:]

    return " ".join(sentence)


def overlap(sentence):
    sentence = sentence.strip().split()
    if len(sentence)<5:
        return " ".join(sentence)

    drop_index = random.sample(range(len(sentence) - 2),1)[0]

    sentence = sentence[:drop_index] + [sentence[drop_index]]+ sentence[drop_index:]

    return " ".join(sentence)

def special_check(sentence):
    spcial = {"are":["is","am"],
              "is": ["am","are"],
              "am": ["are","is"]}

    sentence = sentence.strip().split()
    for i,w in enumerate(sentence):
        if w in spcial:
            rw = random.sample(spcial[w],1)[0]
            sentence[i] = rw
            break
    return " ".join(sentence)

import random
def preprocess_wiki(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".noise_check","w",encoding="utf-8")
    for line in in_f:
        line = line.strip()
        if line=="":
            continue

        rv = random.random()
        if rv<0.2:
            newline = swap(line)
        elif rv<0.4:
            newline =drop(line)
        elif rv<0.6:
            newline =overlap(line)
        elif rv<0.8:
            newline =special_check(line)
        else:
            newline = line
        out_f.write('{}\t{}\n'.format(newline,line))


preprocess_wiki(r"/data/yegong/boshao/bert_corpus/wikipedia_sentences.tokenized/merge_10p")