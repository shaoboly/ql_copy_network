def build_index(refer):
    in_f = open(refer,encoding="utf-8")

    seen = {}
    for i,line in enumerate(in_f):
        try:
            q,num1,num2 = line.strip().split('\t')
        except:
            print(line)
            continue
        seen[q] = 1

    return seen

import json
def read_filter(inname,refername):
    seen = build_index(refername)

    in_f = open(inname,encoding="utf-8")

    out_f = open(inname+".filter","w",encoding="utf-8")
    out_f_long = open(inname + "._long", "w", encoding="utf-8")

    for i,line in enumerate(in_f):
        if i%100000==0:
            print(i)
        try:
            q,num1,num2 = line.strip().split('\t')
        except:
            print(line)
            continue

        if q in seen:
            continue
        else:
            cur_line = {
                "guid":"1",
                "article": q,
                "abstract": "no"
            }

            cur_line = json.dumps(cur_line, ensure_ascii=False)

            if len(q.split())>6:
                out_f_long.write(cur_line+'\n')
            else:
                out_f.write(cur_line + "\n")


# read_filter(r"D:\data\production\keywords\query.10m.tsv","D:\data\production\keywords\query.5m.tsv")


def three_part_split(inname,part=3):
    in_f = open(inname,encoding="utf-8").readlines()

    total = len(in_f)

    first_part = int(total/part)
    sec_part = int(total/part*2)

    out_f1 = open(inname+".1","w",encoding="utf-8")
    out_f2 = open(inname + ".2", "w", encoding="utf-8")
    out_f3 = open(inname + ".3", "w", encoding="utf-8")

    out_f1.writelines(in_f[:first_part])
    out_f2.writelines(in_f[first_part:sec_part])
    out_f3.writelines(in_f[sec_part:])


# three_part_split(r"D:\data\production\keywords\query.10m.tsv.filter")

def merge_three_part(innames,part=3):
    out_f = open(innames[0]+".merge","w",encoding="utf-8")
    for inname in innames:
        in_f = open(inname,encoding="utf-8")
        for i,line in enumerate(in_f):
            if i%100000==0:
                print(i)
            out_f.write(line)

# merge_three_part([r"D:\data\production\keywords\new_5m\good.txt",
#                   r"D:\data\production\keywords\new_5m\good2.txt",
#                   r"D:\data\production\keywords\new_5m\good3.txt"])
#
# merge_three_part([r"D:\data\production\keywords\new_5m\bad.txt",
#                   r"D:\data\production\keywords\new_5m\bad2.txt",
#                   r"D:\data\production\keywords\new_5m\bad3.txt"])
