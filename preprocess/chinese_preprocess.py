import json
def clean_and_sample(inname):
    in_f = open(inname,encoding="utf-8")
    out_f = open(inname+".train","w",encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")
    out_f_test = open(inname + ".test", "w", encoding="utf-8")
    for i,line in enumerate(in_f):
        if i%10000==0:
            print(i)
        if i > 1000000:
            out_f = out_f_dev
        if i > 1010000:
            out_f = out_f_test
        if i> 1020000:
            break

        out_f.write(line)
#clean_and_sample("/data/yegong/boshao/articles.txt")

def clean_all_and_sample(inname):
    in_f = open(inname, encoding="utf-8")
    out_f = open(inname + ".train.all", "w", encoding="utf-8")
    for i, line in enumerate(in_f):
        init_line = json.loads(line)
        try:
            article = init_line["body"]
            title = init_line["title"]
        except:
            continue

        article = article.split()[:512]
        article = " ".join(article)

        cur_line = {
            "body": article,
            "title": title
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line+"\n")

# clean_all_and_sample("/data/yegong/boshao/articles.txt")

def sample_more(inname):
    in_f = open(inname, encoding="utf-8")
    out_f = open(inname + ".train300w", "w", encoding="utf-8")

    for i, line in enumerate(in_f):
        if i < 3000000:
            continue
        if i > 6000000:
            break
        if i%100000==0:
            print(i)
        out_f.write(line)
#sample_more("/data/yegong/boshao/articles.txt")

from data_reading import processors
import os
reader = processors["chineseice"]()

def merge_result(inname,test_dir):
    examples = reader.get_test_examples(inname,"title.testset.reviewed.txt.json")
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

        out_f.write("=================")
        out_f.write("Article:{}\n\n".format(ex.article))
        out_f.write("Title:{}\n\n".format(ex.true_summary))
        out_f.write("Pred:{}\n\n".format(pred))

        out_f.write("=================\n\n\n")

#merge_result(r"/data/yegong/boshao/xiaoice",r"/data/yegong/boshao/bert_s2l_copy-chineseice-04-09-1/test-results")


def generate_new_test_set(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".json","w",encoding="utf-8")
    for line in in_f:
        title,article = line.strip().split('\t')
        cur_line = {
            "body":article,
            "title":title
        }
        cur_line = json.dumps(cur_line,ensure_ascii=False)

        out_f.write(cur_line+'\n')


#generate_new_test_set(r"D:\data\production\xiaoice\title.testset.reviewed.txt")

