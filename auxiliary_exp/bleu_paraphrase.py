import os

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


def bleu_score(list_of_references, predictions):
    return corpus_bleu(list_of_references, predictions, smoothing_function=SmoothingFunction(epsilon=0.01).method1,
                       weights=(0.5, 0.5, 0.0, 0.0)) * 100



def token_check_fresh(tokens,vocab):
    for i,t in enumerate(tokens):
        if t not in vocab:
            tokens[i]="unk"

    return tokens

def compute_acc(dev_dir, ref_dir,vocab_name, test_number=200):
    vocabline = open(vocab_name,encoding="utf-8").readlines()
    vocab = {}
    for line in vocabline:
        vocab[line.strip()]=1

    total = 0
    acc = 0
    list_of_references = []
    predictions = []
    for i in range(test_number):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i
        pred = open(os.path.join(dev_dir, filename)).readlines()

        refs = open(os.path.join(ref_dir, refname)).readlines()

        pred = [s.strip() for s in pred]
        refs = [s.strip() for s in refs]

        all_pred_words = []
        all_ref_words = []
        for s in pred:
            all_pred_words += s.split()

        for s in refs:
            all_ref_words += s.split()

        if all_pred_words[-1] != '.':
            all_pred_words.append('.')
        if all_ref_words[-1] != '.':
            all_ref_words.append('.')

        all_pred_words = token_check_fresh(all_pred_words,vocab)
        all_ref_words = token_check_fresh(all_ref_words,vocab)

        list_of_references.append([all_ref_words])
        predictions.append(all_pred_words)

    s = bleu_score(list_of_references, predictions)
    print("bleu:", )
    print(s)


data_dir = "/data/yegong/boshao//multi_gpu_bert_s2l_copy_add-paraphrase-11-03-newcoco/train_source.txt.merge-results"
compute_acc(data_dir+"/pred",
            data_dir+"/ref",r"/data/yegong/boshao/paraphrase/mscoco_paraphrase/train_vocab.txt")