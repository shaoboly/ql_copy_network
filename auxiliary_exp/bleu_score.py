
import os

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
def bleu_score(list_of_references,predictions):
    return corpus_bleu(list_of_references, predictions,smoothing_function=SmoothingFunction(epsilon=0.01).method1,weights=(0.5, 0.5, 0.0, 0.0)) * 100



def compute_acc(dev_dir, ref_dir,test_number=2000):

    total=0
    acc=0
    list_of_references = []
    predictions = []
    for i in range(test_number):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i
        pred = open(os.path.join(dev_dir,filename)).readlines()

        refs = open(os.path.join(ref_dir, refname)).readlines()

        pred = [s.strip() for s in pred]
        refs = [s.strip() for s in refs]

        all_pred_words = []
        all_ref_words = []
        for s in pred:
            all_pred_words+=s.split()

        for s in refs:
            all_ref_words+=s.split()

        if all_pred_words[-1]!='.':
            all_pred_words.append('.')
        if all_ref_words[-1]!='.':
            all_ref_words.append('.')

        list_of_references.append([all_ref_words])
        predictions.append(all_pred_words)

    s = bleu_score(list_of_references,predictions)
    print("bleu:",)
    print(s)




data_dir = "/home/v-boshao/data//bert_s2l_copy-m2m-11-05-mscoco-full/train_source.txt.merge-results"
compute_acc(data_dir+"/pred",
            data_dir+"/ref")
#
# compute_acc(r"/data/yegong/boshao/summarize_data/summarize_bert_baseline-cnn_dm-03-15-1/test-results/pred",
#            r"/data/yegong/boshao/summarize_data/summarize_bert_baseline-cnn_dm-03-15-1/test-results/ref")



def merge_for_normal(dev_dir, ref_dir,test_number=1000):

    total=0
    acc=0
    list_of_references = []
    predictions = []
    for i in range(test_number):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i
        pred = open(os.path.join(dev_dir,filename)).readlines()

        refs = open(os.path.join(ref_dir, refname)).readlines()

        pred = [s.strip() for s in pred]
        refs = [s.strip() for s in refs]

        all_pred_words = []
        all_ref_words = []
        for s in pred:
            all_pred_words+=s.split()

        for s in refs:
            all_ref_words+=s.split()

        list_of_references.append(all_ref_words)
        predictions.append(all_pred_words)

    out_f = open(os.path.dirname(dev_dir)+r"/ref0","w",encoding="utf-8")
    for line in list_of_references:
        refer = " ".join(line)
        out_f.writelines(refer+"\n")

    out_f = open(os.path.dirname(dev_dir) + r"/pred0", "w", encoding="utf-8")
    for line in predictions:
        refer = " ".join(line)
        out_f.writelines(refer + "\n")

merge_for_normal(data_dir+"/pred",
            data_dir+"/ref")


def compute_f(dev_dir, ref_dir,test_number=1000, beta=0.5):
    total=0
    acc=0
    list_of_references = []
    predictions = []
    for i in range(test_number):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i
        pred = open(os.path.join(dev_dir,filename)).readlines()

        refs = open(os.path.join(ref_dir, refname)).readlines()

        pred = [s.strip() for s in pred]
        refs = [s.strip() for s in refs]


        pred = " ".join(pred)
        refs = " ".join(refs)
        if pred==refs:
            acc+=1

        if pred!=refs:
            print(pred)
            print(refs)
            print("================")

        total+=1

    print("accuracy:{}\n".format(acc/total))

# compute_f(r"/home/v-boshao/data//bert_s2l_copy-disconfuse-04-10-1/test-results/pred",
#            r"/home/v-boshao/data//bert_s2l_copy-disconfuse-04-10-1/test-results/ref")

import json
def compute_recall(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    total = 0
    hit = 0
    for i,line in enumerate(in_f):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        refs = open(os.path.join(os.path.dirname(inname),"ref", refname)).readlines()

        refs = [s.strip() for s in refs]

        all_ref_words = []

        for s in refs:
            all_ref_words += s.split()

        refs = " ".join(refs)

        line = json.loads(line)
        golden = line["golden"]
        candidate = line["generated_beam"]
        if refs in candidate:
            hit+=1
        total+=1
    print(hit/total)
    return hit/total

# compute_recall(r"/home/v-boshao/data//bert_s2l_copy-msnqg-05-24-a/test-results/beam_result.txt")

