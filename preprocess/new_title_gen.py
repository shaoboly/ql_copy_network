import os
import json

def read_refine_split(inname):
    in_f = open(inname, encoding="utf-8").readlines()
    out_f_train = open(inname + ".train", "w", encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")
    out_f_test = open(inname + ".test", "w", encoding="utf-8")

    for i,line in enumerate(in_f):

        new_line = line.split('\t')
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


read_refine_split(r"/home/v-boshao/data/msn/0.1part_data.tsv")


def refine_new_test(inname):
    in_f = open(inname, encoding="utf-8").readlines()
    out_f = open(inname+".refine","w",encoding="utf-8")
    for line in in_f:
        line = line.split('\t')
        id,title,context,abstract,start_p,end_p = line

        context = context.replace("<sep>"," ")

        cur_line = {
            "body": context,
            "title": title
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")





#refine_new_test(r"D:\data\production\msn\test.txt.500")


def merge(inname,pred_name):
    in_f = open(inname, encoding="utf-8").readlines()
    pred_all = open(pred_name, encoding="utf-8").readlines()

    out_f = open(inname+".merge","w",encoding="utf-8")
    for line,pred in zip(in_f,pred_all):
        line = line.strip()
        pred = pred.strip()

        out_f.write( "{}\t{}\n".format(line,pred))


# merge(r"D:\data\production\msn\test.txt.500",r"D:\data\production\msn\pred.all")