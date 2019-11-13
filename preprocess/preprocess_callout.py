import json

def read_callout(inname):
    in_f = open(inname,encoding="utf-8")
    out_f= open(inname+".all_new_train","w",encoding="utf-8")
    out_f_test =  open(inname+".all_new_test","w",encoding="utf-8")
    for i,line in enumerate(in_f):
        if i%10000==0:
            print(i)
        if i>2000000:
            break

        keywords,adtitle,desc1,desc2,domain,url,callout,sentsum,qisnippet,_,_,lpcontent=line.split('\t')

        callout = callout.split('|')[:5]

        for c_call in callout:
            cur_line =  {
                "keywords":keywords,
                "url":url,
                "adtitle":adtitle,
                "desc1":desc1,
                "desc2":desc2,
                "lfcontent":lpcontent,
                "call_out":c_call
            }
            cur_line = json.dumps(cur_line)
            if i %10==0:
                out_f_test.write(cur_line+"\n")
            else:
                out_f.write(cur_line + "\n")


# read_callout(r"D:\data\production\call_out\AdLPInfo_CalloutText.v3.tsv")



def merge_result_with_score(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".beam","w",encoding="utf-8")

    pre_text=None
    cnt = 0
    candidate = []
    beams = ""
    for line in in_f:
        line = json.loads(line)
        cur = line["input"]
        if pre_text!=None and pre_text!=cur:
            out_f.write("=========================\n\n\n")
            out_f.write("Input:{}\n".format(line["src"]))
            out_f.write("Golden:\n")
            candidate = candidate[:5]

            out_f.write(" | ".join(candidate)+"\n")

            out_f.write("Pred:\n")
            out_f.write(" | ".join(beams) + "\n")
            out_f.write("\n\n==========================\n\n\n")
            candidate=[]

        pre_text = cur
        sample_c = line["golden"]
        beams = line["generated_beam"]
        candidate.append(sample_c)


#merge_result_with_score(r"/home/v-boshao/code/trie_generator_init/test.jsonl.outAdLPInfo_CalloutText.v3.tsv.new_test")

import random
def merge_result_with_trie_score(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".beam","w",encoding="utf-8")
    print(len(in_f))
    pre_text=None
    cnt = 0
    candidate = []
    pre_line = None
    pre_golden = []
    for i,line in enumerate(in_f):
        line = json.loads(line)
        cur = str(line['guid'])+line["input"]
        if pre_text!=None and pre_text!=cur:
            pre_golden = list(set(pre_golden))
            candidate = candidate[:5]
            out_json = {
                'input':pre_line,
                'pred':candidate,
                "golden":pre_golden
            }
            out_json= json.dumps(out_json,ensure_ascii=False)
            out_f.write(out_json+"\n")
            candidate=[]
            pre_golden = []

        pre_text = cur
        sample_c = line["generated"]
        sample_c = " ".join(sample_c)
        sample_c = sample_c.replace(" ##","")
        candidate.append(sample_c)
        pre_line = line['input']
        pre_golden.append(line["golden"])
        if i==len(in_f)-1:
            pre_golden = list(set(pre_golden))
            candidate = candidate[:5]

            out_json = {
                'input': pre_line,
                'pred': candidate,
                "golden": pre_golden
            }
            # out_f.write("=========================\n\n\n")
            # out_f.write("Input:{}\n".format(line["input"]))
            # out_f.write("Beam:{}\n")
            # for sample in candidate:
            #     out_f.write(sample+"\n")
            # out_f.write("\n\n==========================\n\n\n")
            out_json = json.dumps(out_json, ensure_ascii=False)
            out_f.write(out_json + "\n")
            candidate = []

merge_result_with_trie_score(r"/home/v-boshao/code/trie_generator_init/test.jsonl.outAdLPInfo_CalloutText.v4.tsv.v4_test")


def calc_acc(inname):
    in_f= open(inname,encoding="utf-8").readlines()
    total = len(in_f)

    trigger = 0
    for line in in_f:
        line = json.loads(line)
        if line["pred"][0]==line["golden"][0]:
            trigger+=1

    print(trigger/total)

#calc_acc(r"/home/v-boshao/code/trie_generator_init/test.jsonl.outhuman_translated_pairs_en_src.q.pair.beam")


def read_split_v4(inname):
    in_f = open(inname,encoding="utf-8")

    out_f = open(inname+".v4_train","w",encoding="utf-8")
    out_f_dev = open(inname+".v4_test", "w", encoding="utf-8")

    for i,line in enumerate(in_f):
        ads_content,LPcontent,Calloutitems = line.strip().split('\t')
        content = "|".join([ads_content,LPcontent])
        Calloutitems = Calloutitems.split("|")
        cur_line = {
            "body": content,
            "callout": Calloutitems
        }



        cur_line = json.dumps(cur_line, ensure_ascii=False)
        if i % 100 == 1:
            out_f_dev.write(cur_line + "\n")
        else:
            out_f.write(cur_line + "\n")

#read_split_v4(r"D:\data\production\call_out\AdLPInfo_CalloutText.v4.tsv")