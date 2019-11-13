import os
import tokenization
import json

tokenizer = tokenization.LFTokenizer(vocab_file=r"/home/v-boshao/data/mspars/seq2seq_pattern/vocab.out.bert", do_lower_case=True)

def compute_acc(dev_dir, ref_name):
    refers = open(ref_name,encoding="utf-8").readlines()

    total=0
    acc=0
    for i,ref in enumerate(refers):
        filename = "%06d_decoded.txt" % i
        pred = open(os.path.join(dev_dir,filename)).readline()
        pred = pred.strip()

        lf = json.loads(ref)["abstract"]
        #lf = tokenizer.tokenize(lf)
        #lf = " ".join(lf)

        if lf==pred:
            acc+=1
        total+=1

    print(acc/total)

compute_acc(r"/home/v-boshao/data/mspars/seq2seq_pattern/bert_s2l_copy-mspars-03-15-1/test-results/pred",
            r"/home/v-boshao/data/mspars/seq2seq_pattern/MSParS.Test.seq2seq.json")