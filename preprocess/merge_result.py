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

        out_f.write("=================\n\n")
        out_f.write("Context:{}\n\n".format(ex.article))
        out_f.write("Title:{}\n\n".format(ex.true_summary))
        out_f.write("Pred:{}\n\n".format(pred))

        out_f.write("=================\n\n\n")


merge_result(r"/data/yegong/boshao/xiaoice",r"/data/yegong/boshao//bert_s2l_copy-chineseice-04-09-1/test-results")

def out_result(inname,test_dir):
    examples = reader.get_test_examples(inname, "nq_ra_an.tsv.test")
    examples = reader.filter_examples(examples)
    out_f = open(os.path.join(test_dir, "pred.all"),"w",encoding="utf-8")
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

        out_f.write(pred+"\n")

#out_result(r"/data/yegong/boshao/msn",r"/data/yegong/boshao/multi_gpu_bert_s2l_copy_add-msn-04-26-1/test-results")