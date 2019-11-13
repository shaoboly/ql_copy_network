# from subtokenizer import SubTokenizer


def generate_word_count(inname):
    words_count = {}
    in_f= open(inname,encoding="utf-8")
    for i,line in enumerate(in_f):
        line = line.lower()
        line = line.strip().split('\t')[0]
        line = line.split()
        for w in line:
            try:
                words_count[w]+=1
            except:
                words_count[w]=1
        if i>10000:
            break
    return words_count

# words_count = generate_word_count(r"D:\data\research\MT\europarl-v9.de-en.tsv\europarl-v9.de-en.tsv")
#
# tokenizer = SubTokenizer.learn(words_count,size=1000)
# tokenizer.save("test")

# from bpe import Encoder

def generate_vocab():
    def generate_lines(inname):
        all_line = []
        in_f= open(inname,encoding="utf-8")
        for i,line in enumerate(in_f):
            line = line.lower()
            line = line.strip().split('\t')[0]
            all_line.append(line)

        return all_line

    all_line = generate_lines(r"D:\data\research\MT\europarl-v9.de-en.tsv\europarl-v9.de-en.tsv")

    print(len(all_line))

    encoder = Encoder(50000, pct_bpe=0.2)  # params chosen for demonstration purposes
    encoder.fit(all_line)


    out_f = open("vocab.bpe","w",encoding="utf-8")

    for w in encoder.word_vocab.keys():
        out_f.write(w+'\n')

    for w in encoder.bpe_vocab.keys():
        out_f.write("##"+w+'\n')

    print("done")


def read_and_split(inname):
    in_f = open(inname, encoding="utf-8")
    out_f_train = open(inname + ".train", "w", encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")
    out_f_test = open(inname + ".test", "w", encoding="utf-8")


    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')
        if len(new_line)<2:
            continue
        article,title = new_line[1],new_line[0]

        if i%100000==0:
            print(i)

        if i>20000:
            out_f=out_f_train
        if i < 20000:
            out_f = out_f_test
        if i < 10000:
            out_f = out_f_dev

        out_f.write("{}\t{}\n".format(article,title))


# read_and_split(r"D:\data\research\MT\europarl-v9.de-en.tsv\europarl-v9.de-en.tsv")

import random
def split(inname):
    in_f = open(inname, encoding="utf-8").readlines()
    out_f_train = open(inname + ".train", "w", encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")

    random.shuffle(in_f)

    in_f_dev = in_f[:3000]
    in_f = in_f[3000:]
    out_f_train.writelines(in_f)
    out_f_dev.writelines(in_f_dev)
# split(r"/home/v-boshao/data/mt/europarl-v7.fr-en.fr.pair")

def merge_data(src_name,tgt):
    src = open(src_name,encoding="utf-8")
    tgt = open(tgt,encoding="utf-8")

    out_f = open(src_name+".pair","w",encoding="utf-8")
    for line1,line2 in zip(src,tgt):
        out_f.write("{}\t{}\n".format(line1.strip(),line2.strip()))


# merge_data(r"D:\data\production\trie_experiment_data\cross_lingual_human_label_pairs\human_translated_pairs_en_src.q",
#            r"D:\data\production\trie_experiment_data\cross_lingual_human_label_pairs\human_translated_pairs_fr_src.q")

