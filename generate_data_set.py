import collections
import random

datadir= r"D:\data\seq2seq\lcquad\detect_lower_split"

train_f = open(datadir+r"\lc.entity_detect.train",encoding="utf-8").readlines()
valid_f = open(datadir+r"\lc.entity_detect.test",encoding="utf-8").readlines()
test_f = open(datadir+r"\lc.entity_detect.test",encoding="utf-8").readlines()

train_out_f = open("train.txt","w",encoding="utf-8")
valid_out_f = open("validation.txt","w",encoding="utf-8")
test_out_f = open("input.txt","w",encoding="utf-8")

source_min =10
target_min = 1
vocat_fenge = "\t"
import re
def generate_vocab(train_f,word_size):
    counter = collections.Counter()

    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[0].split()+line[1].split()

        for w in line:
            w = w.strip()
            counter[w] += 1

            #if re.match("ns:.*?\..*?\.(.)+", w):
            if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
                predict[w] =1

    counter = counter.most_common(word_size)

    vocab = open("vocab","w",encoding="utf-8")

    for i, w in enumerate(counter):
        if w[0] in predict:
            vocab.write("{} {}\n".format(w[0],w[1]))
        else:
            if w[1]>=source_min:
                vocab.write("{} {}\n".format(w[0], w[1]))
    vocab.close()


def generate_predicate_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        predicate = line[1].split()

        line = line[0].split()


        for w in line:
            w = w.strip()
            counter[w] += 1

        for w in predicate:
            if re.match("(r-mso|mso):.*?\..*?\.(.)+", w):
                w =w.split(':')[1]
                w_all = w.split('.')
                for tmp in w_all:
                    counter[tmp] += target_min

    counter = counter.most_common(word_size)

    vocab = open("vocab.in", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= target_min:
            vocab.write("{}{}{}\n".format(w[0],vocat_fenge, w[1]))
    vocab.close()


import re
def generate_target_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[1].split()

        for w in line:
            w = w.strip()
            counter[w] += 1

            #if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
            if "http://dbpedia.org" in w:
                predict[w] = 1

    counter = counter.most_common(word_size)

    vocab = open("vocab.out", "w", encoding="utf-8")

    print(len(predict))
    for i, w in enumerate(counter):
        if w[0] in predict:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
        else:
            if w[1] >= target_min:
                vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()

def generate_target_vocab_pre_only(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[1].split()

        for w in line:
            w = w.strip()
            if not re.match("(r-mso|mso):.*?\..*?\.(.)+", w):
                counter[w] += 1

            if re.match("(r-mso|mso):.*?\..*?\.(.)+",w):
                predict[w] = 1

    counter = counter.most_common(word_size)

    vocab = open("vocab.out", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= target_min:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))

    for w in predict:
        vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()

def generate_source_vocab(train_f,word_size):
    counter = collections.Counter()
    predict = {}
    for line in train_f:
        #line = str(line).strip().lower().split('\t')
        line = str(line).strip().split('\t')
        line = line[0].split()

        for w in line:
            w = w.strip()
            counter[w] += 1


    counter = counter.most_common(word_size)

    vocab = open("vocab.in", "w", encoding="utf-8")

    for i, w in enumerate(counter):
        if w[1] >= source_min:
            vocab.write("{}{}{}\n".format(w[0], vocat_fenge, w[1]))
    vocab.close()


def generate_training_file(train_f,train_out_f):
    random.shuffle(train_f)

    for line in train_f:
        #line =  str(line).strip().lower().split("\t")
        line = str(line).strip().split("\t")
        line[1] = ' '.join(line[1].strip().split())
        train_out_f.write("{}\t{}\n".format(line[0],line[1]))


#generate_training_file(train_f,train_out_f)
#generate_training_file(valid_f,valid_out_f)
#generate_training_file(test_f,test_out_f)
import re
_SPLIT = re.compile("([,.])")
def generate_fresh_data(train_f,train_out_f):
    predicate_pattern = "(r-mso|mso):.*?\..*?\.(.)+"
    for j,line in enumerate(train_f):
        q,l = str(line).strip().lower().split("\t")[:2]
        q = ' '.join(_SPLIT.split(q))

        q = q.replace("’s", " 's")
        q = q.replace("'s", " 's")

        q = ' '.join(q.strip().split())

        l = l.strip().split()
        l_words = []
        for i,w in enumerate(l):
            if re.match(predicate_pattern,w) or w =="’s" or w =="'s":
                l_words.append(w)
            else:
                w = " ".join(_SPLIT.split(w))
                w = " _||_ ".join(w.split())
                w = w.replace("’s"," _||_ 's")
                w = w.replace("'s", " _||_ 's")
                l_words.append(w)

        l = ' '.join(l_words)
        train_out_f.write("{}\t{}\n".format(q, l))

generate_training_file(train_f,train_out_f)
generate_training_file(valid_f,valid_out_f)
generate_training_file(test_f,test_out_f)


generate_vocab(train_f,50000)
generate_target_vocab(train_f,10000)
generate_source_vocab(train_f,10000)


#generate_predicate_vocab(train_f,10000)
#generate_predicate_vocab(train_f,50000)