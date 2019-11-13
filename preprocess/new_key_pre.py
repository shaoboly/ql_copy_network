import json

def merge_result_with_score(inname):
    in_f = open(inname,encoding="utf-8").readlines()
    out_f = open(inname+".beam","w",encoding="utf-8")

    pre_text=None
    cnt = 0
    candidate = []
    pre_line = None
    for i,line in enumerate(in_f):
        line = json.loads(line)
        cur = str(line['guid'])+line["input"]
        if pre_text!=None and pre_text!=cur:
            out_json = {
                'input':pre_line,
                'pred':candidate
            }
            # out_f.write("=========================\n\n\n")
            # out_f.write("Input:{}\n".format(line["input"]))
            # out_f.write("Beam:{}\n")
            # for sample in candidate:
            #     out_f.write(sample+"\n")
            # out_f.write("\n\n==========================\n\n\n")
            out_json= json.dumps(out_json,ensure_ascii=False)
            out_f.write(out_json+"\n")
            candidate=[]

        pre_text = cur
        sample_c = line["generated"]
        sample_c = " ".join(sample_c)
        sample_c = sample_c.replace(" ##","")
        candidate.append(sample_c)
        pre_line = line['input']
        if i==len(in_f)-1:
            out_json = {
                'input': line["input"],
                'pred': candidate
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

merge_result_with_score(r"D:\data\production\ads_title2\part1_part2_adstitle2_aether.tsv")

def remove_long_title(inname):
    in_f = open(inname, encoding="utf-8").readlines()
    out_f = open(inname + ".short", "w", encoding="utf-8")

    for line in in_f:
        column =  line.strip().split('|')[1].split()
        if len(column)>1 and len(column)<10:
            out_f.writelines(line)

# remove_long_title(r"D:\data\production\ads_title2\SampleData_Source_DestinationURL_2019-10-08.tsv")
