import json
def read_refine_split(inname):
    in_f = open(inname, encoding="utf-8")
    out_f_train = open(inname + ".train", "w", encoding="utf-8")
    out_f_dev = open(inname + ".dev", "w", encoding="utf-8")
    out_f_test = open(inname + ".test", "w", encoding="utf-8")

    for i,line in enumerate(in_f):
        if i%100000==0:
            print(i)

        new_line = line.split('\t')
        question,raw_answer,meta = new_line[0],new_line[1],new_line[2]

        meta = json.loads(meta)
        answer = meta['DirectAnswer.Text']

        cur_line = {
            "body": raw_answer,
            "question": question,
            "answer":answer
        }
        if i>10000:
            out_f=out_f_train
        if i < 10000:
            out_f = out_f_test
        if i < 5000:
            out_f = out_f_dev


        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")



#read_refine_split(r"D:\data\production\msn\QG\nq_ra_an.tsv")

def refine_test2(inname):
    in_f = open(inname, encoding="utf-8")
    out_f = open(inname + ".1000jsonl", "w", encoding="utf-8")
    cnt=0
    for i,line in enumerate(in_f):
        all_item = line.split('\t')
        ans_qu = all_item[5]

        guid = all_item[0]+"-"+all_item[1]

        all_samples = ans_qu.split('<end>')
        for sample in all_samples:
            answer,body = sample.split("[ANS_SEQ]")
            cur_line = {
                "guid":guid,
                "body": body,
                "question": "null",
                "answer": answer
            }

            cur_line = json.dumps(cur_line, ensure_ascii=False)
            out_f.write(cur_line + "\n")
            cnt+=1
            if cnt>1000:
                return

# refine_test2(r"D:\data\production\msn\QG\test_data\tvCtxInBodyNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\tvCtxInAbsNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\sportsCtxInBodyNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\sportsCtxInAbsNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\kidsCtxInBodyNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\kidsCtxInAbsNews.tsv")

# refine_test2(r"D:\data\production\msn\QG\test_data\travelCtxInBodyNews.tsv")
# refine_test2(r"D:\data\production\msn\QG\test_data\travelCtxInAbsNews.tsv")

# from data_reading import processors
import os


def merge_result(inname,test_dir,test_name):
    reader = processors["msnqg"]()
    examples = reader.get_test_examples(inname,test_name)
    examples = reader.filter_examples(examples)
    last = test_dir.split("/")[-1]
    out_f = open(os.path.join(test_dir, last+".result"),"w",encoding="utf-8")
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
        #out_f.write("Title:{}\n\n".format(ex.true_summary))
        out_f.write("Pred:{}\n\n".format(pred))

        out_f.write("=================\n\n\n")



#merge_result(r"/data/weizhen/c_trie_generator/data/msnqg",r"/home/v-boshao/data//bert_s2l_copy-msnqg-05-24-a/kidsCtxInBodyNews.tsv.1000jsonl-results","nq_ra_an.tsv.test")
# merge_result(r"/home/v-boshao/data/msn_qg",r"/home/v-boshao/data//bert_s2l_copy-msnqg-05-24-a/travelCtxInBodyNews.tsv.1000jsonl-results","travelCtxInBodyNews.tsv.1000jsonl")

def merge_result_for_generate(inname):
    in_f = open(inname,encoding="utf-8")
    out_f= open(inname+".refine","w",encoding="utf-8")

    pre_context = None
    beam_samples = []
    for i, line in enumerate(in_f):
        if i>100000:
            break
        line = json.loads(line)
        context = line["input"]
        golden = line["golden"]
        sample = line["generated"]
        if pre_context!=None and context!=pre_context:
            out_f.write("=================\n\n")
            out_f.write("Context:{}\n\n".format(context))
            out_f.write("Golden:{}\n\n".format(golden))
            out_f.write("Pred:{}\n\n")
            for c in beam_samples:
                out_f.write(c+"\n")
            beam_samples=[]
            out_f.write("=================\n\n\n")

        pre_context=context
        beam_samples.append(sample)

#merge_result_for_generate(r"/data/weizhen/c_trie_generator/data/bert_s2l_copy-msnqg-05-24-a/nq_ra_an.tsv.test-results/beam_results.txt")


def merge_result_with_score(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".beam","w",encoding="utf-8")

    pre_text=None
    cnt = 0
    candidate = []
    for line in in_f:
        line = json.loads(line)
        cur = str(line['guid'])+line["input"]
        if pre_text!=None and pre_text!=cur:
            out_f.write("=========================\n\n\n")
            out_f.write("Input:{}\n".format(line["input"]))
            out_f.write("Beam:\n")
            candidate = candidate[:5]
            for sample in candidate:
                out_f.write(sample+"\n")
            out_f.write("\n\n==========================\n\n\n")
            candidate=[]

        pre_text = cur
        sample_c = line["generated"]
        candidate.append(sample_c)


# merge_result_with_score(r"D:\data\production\msn\QG\trie_result\kid_abs_beam_results.txt")
# merge_result_with_score(r"D:\data\production\msn\QG\trie_result\kid_ctx_beam_results.txt")
# merge_result_with_score(r"D:\data\production\msn\QG\trie_result\sports_abs_beam_results.txt")
# merge_result_with_score(r"D:\data\production\msn\QG\trie_result\sports_ctx_beam_results.txt")



def merge_pos_neg(pos_name,neg_name):
    pos_namein_f = open(pos_name, encoding="utf-8")
    neg_namein_f = open(neg_name, encoding="utf-8")
    out_f = open(pos_name + ".small", "w", encoding="utf-8")
    cnt=2000

    for pos_line,neg_line in zip(pos_namein_f,neg_namein_f):
        pos_line = json.loads(pos_line)
        neg_line = json.loads(neg_line)

        pos_line["is_impossible"] = False
        neg_line["is_impossible"] = True

        pos_line = json.dumps(pos_line)
        neg_line = json.dumps(neg_line)

        out_f.write(pos_line + "\n")
        out_f.write(neg_line + "\n")

        cnt-=1
        if cnt==0:
            break


merge_pos_neg(r"D:\data\research\NQ_gen\nq_pretrain_data\nq_pretrain.100w",r"D:\data\research\NQ_gen\nq_pretrain_data\unanswerable_qd.json")