import json
import os

def read_and_refine(src,tgt):
    out_f = open(src + ".all",'w',encoding="utf-8")
    src = open(src,encoding="utf-8").readlines()
    tgt = open(tgt, encoding="utf-8").readlines()

    i=0
    for line1,line2 in zip(src,tgt):
        result = {'article':line1,
                  'abstract':line2}

        our_line = json.dumps(result,ensure_ascii=False)
        out_f.write(our_line+'\n')


def read_from_init(inname):
    out_f = open(inname + ".refine", 'w', encoding="utf-8")
    in_f = open(inname, encoding="utf-8").readline()
    data_all=json.loads(in_f)

    data_all = data_all["data"]

    number= 0

    for para in data_all:
        title = para['title']
        now_paragraph = para['paragraphs']
        for context_list in now_paragraph:
            context = context_list['context']
            qas_all = context_list['qas']
            for qa in qas_all:
                question = qa['question']
                answer = qa['answers'][0]['text']
                one_result = {'article':  context,
                              'abstract': question,
                              'id':number,
                              'title':title,
                              'answer':answer}
                number+=1
                our_line = json.dumps(one_result, ensure_ascii=False)
                out_f.write(our_line + '\n')

    print("done")


from data_reading import processors
import os
reader = processors["new_key_multi"]()

def merge_result(inname,test_dir):
    examples = reader.get_test_examples(inname,"AdInsight_AdCreative_SeqPairTrainingData_20190702.tsv.dev")
    examples = reader.filter_examples(examples)
    out_f = open(os.path.join(test_dir, "result"),"w",encoding="utf-8")
    print(len(examples))
    em=0
    total=0
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


        if ex.true_summary.lower().strip()==pred.strip().lower():
            em+=1
        total+=1

        if total>600:
            print(em/total)
            break

        out_f.write("=================\n\n")
        out_f.write("Input:{}\n\n".format(ex.article))
        out_f.write("Golden:{}\n\n".format(ex.true_summary))
        out_f.write("Pred:{}\n\n".format(pred))

        out_f.write("=================\n\n\n")




if __name__=="__main__":
    #read_from_init(r"D:\data\research\mrc\SQuAD-explorer-master\SQuAD-explorer-master\dataset\dev-v1.1.json")
    #read_from_init(r"D:\data\research\mrc\SQuAD-explorer-master\SQuAD-explorer-master\dataset\train-v1.1.json")
    # read_and_refine(r"/data/yegong/boshao/qg/train/train.txt.source.txt.shuf",
    #                 r"/data/yegong/boshao/qg/train/train.txt.target.txt.shuf")
    # read_and_refine(r"/data/yegong/boshao/qg/dev/dev.txt.shuffle.dev.source.txt",
    #                 r"/data/yegong/boshao/qg/dev/dev.txt.shuffle.dev.target.txt")
    # read_and_refine(r"/data/yegong/boshao/qg/test/dev.txt.shuffle.test.source.txt",
    #                 r"/data/yegong/boshao/qg/test/dev.txt.shuffle.test.target.txt")

    merge_result(r"/data/yegong/boshao/new_key_multi", r"/data/yegong/boshao/multi_gpu_bert_s2l_copy_add-new_key_multi-07-01-1/test-results")
