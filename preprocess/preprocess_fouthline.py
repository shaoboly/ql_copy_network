import json

def read_shuffle_split(inname = r"D:\data\production\m2m\MSMLabeling.txt"):
    in_f = open(inname, encoding="utf-8")

    out_f= open(inname+".train","w",encoding="utf-8")
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')
        if len(new_line)<2:
            continue

        if i%100000==0:
            print(i)

        article,title = new_line[0],new_line[1]

        cur_line = {
            "body": article,
            "title": title
        }
        cur_line = json.dumps(cur_line, ensure_ascii=False)
        out_f.write(cur_line + "\n")

read_shuffle_split()