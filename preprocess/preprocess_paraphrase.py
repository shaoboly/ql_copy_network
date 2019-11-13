import random
import csv

def read_refine_split_new_2017_quaro(inname):
    def out_data(f,data):
        for line in data:
            f.write("{}\t{}\n".format(line[0],line[1]))

    in_f = open(inname, encoding="utf-8")
    in_f.readline()

    in_f = in_f.readlines()
    collect_all = []
    for line in in_f:
        line = line.strip().split('\t')
        if len(line)<6:
            print(line)
            continue
        if line[5].strip() == "1":
            collect_all.append([line[3], line[4]])

    random.shuffle(collect_all)

    out_f = open(inname + ".test", "w", encoding="utf-8")
    test_set = collect_all[:1000]
    out_data(out_f,test_set)

    collect_all = collect_all[1000:]

    random.shuffle(collect_all)
    out_data(open(inname + ".train50", "w", encoding="utf-8"), collect_all[:50000])
    out_data(open(inname + ".train100", "w", encoding="utf-8"), collect_all[:100000])
    out_data(open(inname + ".train", "w", encoding="utf-8"), collect_all)

# read_refine_split_new_2017_quaro(r"D:\data\research\paraphrase\quora-question-pairs\2017_quaro\quora_duplicate_questions.tsv")

def read_refine_split_quaro(inname):
    in_f = open(inname, encoding="utf-8")
    in_f.readline()
    in_f = csv.reader(in_f)
    #in_f = in_f[1:]

    def out_data(f,data):
        for line in data:
            f.write("{}\t{}\n".format(line[0],line[1]))

    collect_all = []
    for line in in_f:
        if len(line)<6:
            continue
        if line[5].strip()=="1":
            collect_all.append([line[3],line[4]])

    random.shuffle(collect_all)

    out_f = open(inname + ".test", "w", encoding="utf-8")
    test_set = collect_all[:1000]
    out_data(out_f,test_set)

    collect_all = collect_all[1000:]

    random.shuffle(collect_all)

    # out_data(open(inname + ".train50", "w", encoding="utf-8"), collect_all[:50000])
    # out_data(open(inname + ".train", "w", encoding="utf-8"), collect_all)

# read_refine_split_quaro(r"D:\data\research\paraphrase\quora-question-pairs\train.csv")







def read_generate_pos(inname=None):
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(r'/home/v-boshao/service/stanford-corenlp-full-2018-10-05')

    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname + ".with_tag", "w", encoding="utf-8")
    for i,line in enumerate(in_f):
        if i%1000==0:
            print(i)
        input,output = line.strip().split('\t')

        input_tag = nlp.pos_tag(input)
        output_tag = nlp.pos_tag(output)

        tokenize1 = " ".join([c[0] for c in input_tag])
        tokenize2 = " ".join([c[0] for c in output_tag])

        tags1 = " ".join([c[1] for c in input_tag])
        tags2 = " ".join([c[1] for c in output_tag])

        out_f.write("{}\t{}\t{}\t{}\n".format(tokenize1,tokenize2,tags1,tags2))


    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    # print('Tokenize:', nlp.word_tokenize(sentence))
    # print('Part of Speech:', nlp.pos_tag(sentence))
    # t = nlp.pos_tag(sentence)
    # print("done")

# read_generate_pos(r"/home/v-boshao/data/paraphrase/wikianswers/test_source.txt.sample")
# read_generate_pos(r"/home/v-boshao/data/paraphrase/wikianswers/train_source.txt.test2")
# read_generate_pos(r"/home/v-boshao/data/paraphrase/wikianswers/train_source.txt.500k")



def read_generate_tree_pos(inname=None):
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(r'/home/v-boshao/service/stanford-corenlp-full-2018-10-05')

    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname + ".with_parsing_tree", "w", encoding="utf-8")
    for i,line in enumerate(in_f):
        if i%1000==0:
            print(i)
        input,output = line.strip().split('\t')

        try:
            input_tag = nlp.parse(input).split('\n')
            output_tag = nlp.parse(output).split('\n')
        except:
            print("Input {}, Output {}".format(input,output))
            continue


        tags1 = " ".join([' '.join(c.strip().split()) for c in input_tag])
        tags2 = " ".join([' '.join(c.strip().split()) for c in output_tag])

        input_tag = nlp.pos_tag(input)
        output_tag = nlp.pos_tag(output)

        tokenize1 = " ".join([c[0] for c in input_tag])
        tokenize2 = " ".join([c[0] for c in output_tag])

        out_f.write("{}\t{}\t{}\t{}\n".format(tokenize1,tokenize2,tags1,tags2))


# read_generate_tree_pos(r"/home/v-boshao/data/paraphrase/quora/train.csv.train")
# read_generate_tree_pos(r"/home/v-boshao/data/paraphrase/quora/train.csv.test")
# read_generate_pos(r"/home/v-boshao/data/paraphrase/quora/train.csv.train")


def find_tag_vocab(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname + ".tag_vocab", "w", encoding="utf-8")

    tags = {}
    for line in in_f:
        line = line.strip().split('\t')

        for tag in line[2].split():
            tags[tag.lower()]=1
        for tag in line[3].split():
            tags[tag.lower()]=1

    for k in tags.keys():
        out_f.write(k+"\n")


#find_tag_vocab(r"/home/v-boshao/data/paraphrase/quora/train.csv.train.with_tag")


def merge_to_bert_vocab(inname,bert_name=r"/home/v-boshao/code/bert-intent/model/uncased_L-12_H-768_A-12/vocab.txt"):
    bert_vocab = open(bert_name,encoding="utf-8").readlines()
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname + ".bert_tag_vocab", "w", encoding="utf-8")

    cnt_token = 0
    for i,line in enumerate(bert_vocab):
        if line.strip()[:7]=='[unused':
            if in_f[cnt_token] not in bert_vocab:
                bert_vocab[i]=in_f[cnt_token]
            else:
                print(in_f[cnt_token])
            cnt_token+=1
            if len(in_f)==cnt_token:
                break

    out_f.writelines(bert_vocab)

# merge_to_bert_vocab(r"/home/v-boshao/data/paraphrase/quora/train.csv.train.with_tag.tag_vocab")

import json
def preprocess_mscoco(inname):
    in_f = open(inname,encoding="utf-8").readline()
    out_f = open(inname+".pair","w",encoding="utf-8")

    dataset = json.loads(in_f)
    print("done")
    annotations = dataset["annotations"]
    captions_all = {}
    for item in annotations:
        image_id = item["image_id"]
        try:
            captions_all[image_id].append(item["caption"].strip())
        except:
            captions_all[image_id]=[item["caption"].strip()]

    for item in captions_all.items():
        out_line = "\t".join(item[1])
        out_f.write(out_line+"\n")



# preprocess_mscoco(r"D:\data\research\paraphrase\annotations_trainval2014\annotations\captions_train2014.json")

def merge_shuffle_split_coco(inlist):
    all_samples = []
    for inname in inlist:
        in_f = open(inname, encoding="utf-8").readlines()
        all_samples+=in_f
    random.shuffle(all_samples)
    pair_candidate = []
    for sample in all_samples:
        sample = sample.strip().split('\t')
        random.shuffle(sample)
        if len(sample)<5:
            print(sample)
            continue
        for i in range(4):
            pair_candidate.append([sample[i],sample[4]])

    random.shuffle(pair_candidate)
    random.shuffle(pair_candidate)

    out_f = open(inlist[0]+".pairwise.test","w",encoding="utf-8")
    for line in pair_candidate[:2000]:
        out_line = "{}\t{}\n".format(line[0],line[1])
        out_f.write(out_line)

    out_f = open(inlist[0] + ".pairwise.dev", "w", encoding="utf-8")
    for line in pair_candidate[2000:4000]:
        out_line = "{}\t{}\n".format(line[0], line[1])
        out_f.write(out_line)

    out_f = open(inlist[0] + ".pairwise.train", "w", encoding="utf-8")
    for line in pair_candidate[4000:]:
        out_line = "{}\t{}\n".format(line[0], line[1])
        out_f.write(out_line)

#merge_shuffle_split_coco([r"D:\data\research\paraphrase\annotations_trainval2014\annotations\captions_train2014.json.pair",r"D:\data\research\paraphrase\annotations_trainval2014\annotations\captions_val2014.json.pair"])





def replace_with_generate(test_set,tagset):
    in_f = open(test_set, encoding="utf-8").readlines()
    tag_inf = open(tagset, encoding="utf-8").readlines()

    out_f = open(test_set+".semantic_pred_tag","w",encoding="utf-8")

    for line,tagline in zip(in_f,tag_inf):
        line = line.strip().split('\t')
        tagline = json.loads(tagline)

        out_taget = tagline["generated_beam"][0].strip()
        line.append(line[3])
        line[3] = out_taget
        out_f.write("\t".join(line)+"\n")

# replace_with_generate(r"/home/v-boshao/data/paraphrase/quora/train_part1",r"/home/v-boshao/data/paraphrase/quora/semantic_tag/train_part1-results/beam_result.txt")
#
# replace_with_generate(r"/home/v-boshao/data/paraphrase/quora/train_part2",r"/home/v-boshao/data/paraphrase/quora/semantic_tag/train_part2-results/beam_result.txt")
#
# replace_with_generate(r"/home/v-boshao/data/paraphrase/quora/train.csv.test.with_tag",r"/home/v-boshao/data/paraphrase/quora/semantic_tag/train.csv.test.with_tag-results/beam_result.txt")

def merge_two_part(part_list):
    out_f = open(part_list[0]+".merge","w",encoding="utf-8")
    for inname in part_list:
        out_f.writelines(open(inname,encoding="utf-8").readlines())

# merge_two_part([r"/home/v-boshao/data/paraphrase/quora/train_part1.semantic_pred_tag",r"/home/v-boshao/data/paraphrase/quora/train_part2.semantic_pred_tag"])



def merge_alignment(srcname,tgtname):
    srcin_f= open(srcname,encoding="utf-8").readlines()
    tgtin_f= open(tgtname,encoding="utf-8").readlines()
    out_f = open(srcname+".500k","w",encoding="utf-8")

    merge_all = list(zip(srcin_f,tgtin_f))
    random.shuffle(merge_all)
    cnt=500000
    for src,tgt in merge_all:
        src = src.strip().split()
        tgt = tgt.strip().split()

        if len(src)==0 or len(src)>15:
            continue
        if len(tgt)==0 or len(tgt)>15:
            continue
        out_f.write("{}\t{}\n".format(" ".join(src)," ".join(tgt)))
        cnt-=1
        if cnt==0:
            break


# merge_alignment(r"D:\data\research\paraphrase\neural-paraphrase-generation-dev\neural-paraphrase-generation-dev\data\mscoco\train_source.txt",
#                 r"D:\data\research\paraphrase\neural-paraphrase-generation-dev\neural-paraphrase-generation-dev\data\mscoco\train_target.txt")
# merge_alignment(r"D:\data\research\paraphrase\neural-paraphrase-generation-dev\neural-paraphrase-generation-dev\data\wikianswers\train_source_1.txt",
#                 r"D:\data\research\paraphrase\neural-paraphrase-generation-dev\neural-paraphrase-generation-dev\data\wikianswers\train_target_1.txt")


def add_and_sample(inname1,inname2):
    in_f= open(inname1,encoding="utf-8").readlines()
    in_f2 = open(inname2,encoding="utf-8").readlines()

    out_f = open(inname1+".refine","w",encoding="utf-8")

    random.shuffle(in_f)
    random.shuffle(in_f2)
    total=[]
    total+=in_f[:300]
    total += in_f2[:1700]
    random.shuffle(total)

    out_f.writelines(total)



# add_and_sample(r"/data/yegong/boshao/paraphrase/mscoco_paraphrase/test_source.txt.sptest2",r"/data/yegong/boshao/paraphrase/mscoco_paraphrase/train_source.txt.merge")


def find_vocab(inname):
    import collections
    in_f = open(inname, encoding="utf-8").readlines()
    out_f = open(inname+".tag_vocab","w",encoding="utf-8")

    counter = collections.Counter()
    for line in in_f:
        line = line.strip().split('\t')
        line[2] = line[2].replace('(',' ( ').replace(')',' ) ').strip().split()
        line[3] = line[3].replace('(', ' ( ').replace(')', ' ) ').strip().split()
        for i,w in enumerate(line[2]):
            if i<len(line[2])-1:
                if line[2][i+1]==')':
                    continue

            if w.upper()==w:
                counter[w]+=1
        for i,w in enumerate(line[3]):
            if i<len(line[3])-1:
                if line[3][i+1]==')':
                    continue
            if w.upper() == w:
                counter[w]+=1

    counter = counter.most_common(1000)
    for i, w in enumerate(counter):
        out_f.write("'{}', ".format(w[0]))


find_vocab(r"/home/v-boshao/data/paraphrase/quora/train.csv.train.with_parsing_tree")