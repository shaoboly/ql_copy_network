import collections
import random

datadir= r"D:\data\seq2seq\MSPaD.Merge\MSPaD"

train_f = open(datadir+"\MSPaD.Merge.train.SP.split.shuffle",encoding="utf-8").readlines()
valid_f = open(datadir+"\MSPaD.Merge.dev.SP.split.shuffle",encoding="utf-8").readlines()
test_f = open(datadir+"\MSPaD.Merge.test.SP.split.shuffle",encoding="utf-8").readlines()

train_out_f = open("train.txt","w",encoding="utf-8")
valid_out_f = open("validation.txt","w",encoding="utf-8")
test_out_f = open("input.txt","w",encoding="utf-8")


def generate_vocab(train_f,word_size):
    counter = collections.Counter()

    predict = {}
    for line in train_f:
        line = str(line).strip().lower().split('\t')
        line = line[0].split()+line[1].split()

        for w in line:
            w = w.strip()
            counter[w] += 1
            if len(w.split('.'))==3:
                predict[w] =1

    counter = counter.most_common(word_size)

    vocab = open("vocab","w",encoding="utf-8")

    for i, w in enumerate(counter):
        if w[0] in predict:
            vocab.write("{} {}\n".format(w[0],w[1]))
        else:
            if w[1]>=4:
                vocab.write("{} {}\n".format(w[0], w[1]))
    vocab.close()

def generate_training_file(train_f,train_out_f):
    random.shuffle(train_f)

    for line in train_f:
        line =  str(line).strip().lower().split("\t")

        train_out_f.write("{}\t{}\n".format(line[0],line[1]))

generate_vocab(train_f,50000)

generate_training_file(train_f,train_out_f)
generate_training_file(valid_f,valid_out_f)
generate_training_file(test_f,test_out_f)
