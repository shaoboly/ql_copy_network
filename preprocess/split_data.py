def split_data(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    cnt = 4
    total_length = len(in_f)
    part = int(total_length/cnt)+1

    st = 0
    for i in range(cnt):
        out_f = open(inname + '.split{}'.format(i), "w", encoding="utf-8")

        out_f.writelines(in_f[st:st+part])
        st = st+part

#split_data(r"/data/yegong/boshao/summarize_data/cnn_dm/train.txt")

import random
def sample(inname,part=0.1):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+"sample{}".format(part),"w",encoding="utf-8")

    sample_number = int(len(in_f)*part)
    sample_results = list(random.sample(in_f,sample_number))
    out_f.writelines(sample_results)

# sample(r"/data/yegong/boshao/cnn_dm/train.txt")
# sample(r"/data/yegong/boshao/cnn_dm/dev.txt")
# sample(r"/data/yegong/boshao/cnn_dm/test.txt")


import json
def refine(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+'.jsonl',"w",encoding="utf-8")
    for line in in_f:
        line = json.loads(line)

        out_lines = line["tgt"]
        out_result = []
        for out in out_lines:
            out = out.replace(" ##", "")
            out_result.append(out)

        line["tgt"] = out_result
        cur_line = json.dumps(line,ensure_ascii=False)
        out_f.write(cur_line+"\n")

refine(r"D:\data\production\keywords\new_key\beam_result.txt")