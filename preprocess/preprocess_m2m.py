import json
def combine_refine_label(inname,refname):
    in_f= open(inname,encoding="utf-8").readlines()
    ref_f = open(refname,encoding="utf-8").readlines()
    out_f = open(inname+".refine","w",encoding="utf-8")
    for line,ref in zip(in_f,ref_f):
        line = json.loads(line)
        label = ref.strip().split('\t')[-1]

        generate = []
        for item in line["generated_beam"]:
            if "[CLS]" in item or "[SEP]" in item:
                continue
            generate.append(item)

        out_line = {
            "query":line["src"],
            "kw":line["golden"],
            "label":label,
            "predicate":generate
        }
        out_line= json.dumps(out_line)
        out_f.write(out_line+"\n")


#combine_refine_label(r"D:\data\production\m2m\beam_result.txt",r"D:\data\production\m2m\MSMLabeling.txt")

def combine_refine(inname):
    in_f= open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".refine","w",encoding="utf-8")
    for line in in_f:
        line = json.loads(line)

        generate = []
        for item in line["generated_beam"]:
            if "[CLS]" in item or "[SEP]" in item:
                continue
            generate.append(item)

        out_line = {
            "query":line["src"],
            "generated_beam":generate
        }
        out_line= json.dumps(out_line)
        out_f.write(out_line+"\n")
# combine_refine(r"D:\data\production\m2m\query_decile_2-8_2019-07-03_2019-10-03.txt.short.beam")

def cnt_number(inname):
    in_f = open(inname, encoding="utf-8")
    cnt=0
    out_f = open(inname + ".short_part0_length7", "w", encoding="utf-8")
    part=0
    for i,line in enumerate(in_f):
        origin_line = line
        line = line.split('\t')
        if i%100000==0:
            print("i:{}, cnt:{}\n".format(i,cnt))
        if len(line) == 2:
            #if int(line[1]) > 6 or int(line[1]) < 2:
            if int(line[1]) != 7:
                # print("skip : {}  num:{}".format(line[0],line[1]))
                continue
            else:
                out_f.write(origin_line)
                cnt+=1

                if cnt%300000==0:
                    part+=1
                    out_f = open(inname + ".short_part{}_length7".format(part), "w", encoding="utf-8")





cnt_number(r"D:\data\production\m2m\query_decile_2-8_2019-07-03_2019-10-03.txt")


def merge_by_dirname(indir):
    out_f = open("total.txt","w",encoding="utf-8")
    for i in range(69):
        inname = indir+"/{}_result/beam_result.txt".format(i)
        print(inname)
        in_f = open(inname,encoding="utf-8").readlines()
        out_f.writelines(in_f)
