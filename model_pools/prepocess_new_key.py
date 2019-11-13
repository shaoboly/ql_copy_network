import random

def split_new_key(inname):
    in_f = open(inname,encoding="utf-8").readlines()

    random.shuffle(in_f)
    dev_in_f = in_f[:2000]
    in_f = in_f[2000:]

    out_f = open(inname+".train","w",encoding="utf-8")
    for i,line in enumerate(in_f):
        out_f.write(line)

    out_f = open(inname + ".dev", "w", encoding="utf-8")
    for i, line in enumerate(dev_in_f):
        out_f.write(line)


split_new_key(r"D:\data\production\keywords\new_key\AdInsight_AdCreative_SeqPairTrainingData_20190702.tsv")