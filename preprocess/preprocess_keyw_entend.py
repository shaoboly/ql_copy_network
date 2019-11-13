import os
import json

import random

def read_refine_split(inname):
    in_f = open(inname, encoding="utf-8")

    #random.shuffle(in_f)

    out_f_train = open(inname + ".train", "w", encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")
    out_f_test = open(inname + ".test", "w", encoding="utf-8")
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')

        # if (i+1)%16000000==0:
        #     now_part = int(i / 16000000)
        #     out_f_train = open(inname + ".train{}".format(now_part), "w", encoding="utf-8")
        #     out_f = out_f_train

        if len(new_line)<2:
            continue

        article,title = new_line[0],new_line[1]

        cur_line = {
            "body": article,
            "title": title
        }

        if i%10000==0:
            print(i)

        if i>10000:
            out_f=out_f_train
        if i < 10000:
            out_f = out_f_test
        if i < 5000:
            out_f = out_f_dev


        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")


#read_refine_split(r"/data/yegong/boshao/keywords/EEMv507_6_2018-08-21.txt")


def read_shuffle_split(inname = r"/data/yegong/boshao/keywords/EEMv507_6_2018-08-21.txt"):
    in_f = open(inname, encoding="utf-8")

    out_f_list = []
    for i in range(10):
        out_f_list.append(open(inname+".train{}".format(i),"w",encoding="utf-8"))
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')
        if len(new_line)<2:
            continue

        if i%100000==0:
            print(i)

        article,title = new_line[0],new_line[1]

        cur_line = {
            "body": article,
            "title": title
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f = out_f_list[i%10]
        out_f.write(cur_line + "\n")


#read_shuffle_split(r"/home/v-boshao/data/keywords/EEMv507_6_2018-08-21.txt")

def read_refine(inname):
    in_f = open(inname, encoding="utf-8")
    in_f.readline()
    # random.shuffle(in_f)

    out_f = open(inname + ".jsonl", "w", encoding="utf-8")
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')

        # if (i+1)%16000000==0:
        #     now_part = int(i / 16000000)
        #     out_f_train = open(inname + ".train{}".format(now_part), "w", encoding="utf-8")
        #     out_f = out_f_train

        if len(new_line)<3:
            continue

        guid,article,title = new_line[0],new_line[1],new_line[2]

        cur_line = {
            "body": article,
            "title": title
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

# read_refine(r"D:\data\production\keywords\0710data\data.tsv")

def merge_all(inname,piece):
    out_f = open(os.path.join(os.path.dirname(inname),"train.all"),"w",encoding="utf-8")
    for i in range(piece):
        print(inname+str(i))
        in_f = open(inname+str(i),encoding="utf-8").readlines()
        out_f.writelines(in_f)

# merge_all(r"/home/v-boshao/data/keywords/EEMv507_6_2018-08-21.txt.train",16)

def refine_new_test(inname):
    out_f = open(inname + ".test", "w", encoding="utf-8")

    in_f = open(inname, encoding="utf-8")
    for i, line in enumerate(in_f):
        line = line.strip()
        cur_line = {
            "body": line,
            "title": line
        }

        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

def refine_new_train(inname):
    out_f = open(inname + ".train", "w", encoding="utf-8")

    in_f = open(inname, encoding="utf-8")
    for i, line in enumerate(in_f):
        if i%100000==0:
            print(i)
        line = line
        line = line.strip().split('\t')
        if len(line)<2:
            continue
        cur_line = {
            "body": line[0],
            "title": line[1]
        }

        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

# refine_new_train(r"/home/v-boshao/data/yeyun_9975.tsv")


from data_reading import processors
import os
reader = processors["msn"]()

def merge_result(inname,test_dir):
    examples = reader.get_test_examples(r"/data/yegong/boshao/keywords","real.test")
    examples = reader.filter_examples(examples)
    out_f = open(os.path.join(test_dir, "result"),"w",encoding="utf-8")
    print(len(examples))
    for i,ex in enumerate(examples):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i
        pred = open(os.path.join(test_dir,"pred", filename)).readlines()

        refs = open(os.path.join(test_dir,"ref", refname)).readlines()
        pred = [s.strip() for s in pred]
        refs = [s.strip() for s in refs]

        if len(pred) > 1:
            print(pred)

        pred = " ".join(pred)
        refs = " ".join(refs)

        out_f.write("=================\n\n")
        out_f.write("Query:{}\n\n".format(ex.article))
        out_f.write("Pred:{}\n\n".format(pred))

        out_f.write("=================\n\n\n")



#merge_result("",r"/data/yegong/boshao//multi_gpu_bert_s2l_copy_add-msn-04-30-1-keywords/test-results")


def convert_title_key(inname):
    out_f = open(inname + ".test", "w", encoding="utf-8")

    in_f = open(inname, encoding="utf-8")
    for i, line in enumerate(in_f):
        line = line.strip().split('\t')
        cur_line = {
            "guid": line[0],
            "article": line[5],
            "abstract": ""
        }

        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

convert_title_key(r"D:\data\production\keywords\Seed_1_sample_0.01.ss")