import json

def read_refine(inname):
    in_f = open(inname, encoding="utf-8")
    in_f.readline()
    # random.shuffle(in_f)

    out_f = open(inname + ".jsonl", "w", encoding="utf-8")
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')

        if i>30000000:
            break

        # if (i+1)%16000000==0:
        #     now_part = int(i / 16000000)
        #     out_f_train = open(inname + ".train{}".format(now_part), "w", encoding="utf-8")
        #     out_f = out_f_train

        if len(new_line)<3:
            continue

        query,ke, = new_line[0],new_line[1]

        cur_line = {
            "body": query,
            "title": ke
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

read_refine(r"D:\data\production\m2m\1_100_training.tsv")