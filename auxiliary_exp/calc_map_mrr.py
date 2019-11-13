import json

def read_label_dict(inname=r"D:\data\production\keywords\to20190416.tsv"):
    in_f = open(inname, encoding="utf-8")
    in_f.readline()
    # random.shuffle(in_f)

    #out_f = open(inname + ".0617_test", "w", encoding="utf-8")
    label_dict = {}
    for i,line in enumerate(in_f):

        new_line = line.strip().split('\t')

        # if (i+1)%16000000==0:
        #     now_part = int(i / 16000000)
        #     out_f_train = open(inname + ".train{}".format(now_part), "w", encoding="utf-8")
        #     out_f = out_f_train

        if len(new_line)<2:
            continue

        article,title,label = new_line[0],new_line[1],new_line[2]
        label_dict[article+"[seq]"+ title]=label
    return label_dict
label_dict = read_label_dict()

def add_intrie_or_not( inname=r"D:\data\production\keywords\test\trie_check_result.txt"):
    in_f = open(inname, encoding="utf-8")
    # random.shuffle(in_f)

    # out_f = open(inname + ".0617_test", "w", encoding="utf-8")
    in_trie = {}
    for i, line in enumerate(in_f):
        line = json.loads(line)
        article = line["article"]
        title = line["abstract"][0]
        trie_label = line["abstract"][1]
        in_trie[article+"[seq]"+ title] = trie_label
    return in_trie

in_trie = add_intrie_or_not()


def filter_rule(query):
    words = query.split()
    judge_di = False
    if len(words)<=1:
        judge_di = words[0].isdigit()

    return judge_di

def calc_map(all_results,topk=50):
    map_score = 0
    total = 0
    trigger= 0

    for guid,result in all_results.items():
        labels = result["pred"]
        labels = labels[:topk]
        gd_num = 0.0
        avg_score = 0
        for i,c_label in enumerate(labels):
            if c_label==1:
                gd_num += 1
                avg_score += gd_num / (i + 1)

        if gd_num==0:
            avg_score=0
        else:
            avg_score /= gd_num
            trigger+=1
        map_score += avg_score
        total += 1
    return map_score/total, trigger/total

def collect_bad(all_results,filename):
    out_f = open(filename,"w",encoding="utf-8")
    out_f_json = open(filename+".json","w",encoding="utf-8")
    all_collection = []
    for guid, result in all_results.items():
        labels = result["pred"][:20]
        label = True

        if 1 not in labels:

            label=False

        all_collection.append(result)
        out_f.write("==========================\n\n")
        out_f.write("Query:{}\n".format(result["query"]))
        out_f.write("Golden:{}\n".format(result["golden"]))
        out_f.write("Trigger:{}\n".format(label))
        out_f.write("Trie_label:{}\n".format(result["trie_label"]))
        out_f.write("Pred:\n")
        for pred in result["generate_result"]:
            out_f.write(pred + "\n")


        dict_now = {
            "query":result["query"],
            "golden":result["golden"],
            "pred":result["generate_result"],
            "trie_label":result["trie_label"],
            "trigger":label
        }

        out_f_json.write(json.dumps(dict_now,ensure_ascii=False)+"\n")

    return all_collection


def collect_all(all_results,filename):
    out_f = open(filename,"w",encoding="utf-8")
    all_collection = []
    for guid, result in all_results.items():
        labels = result["pred"][:20]
        label = True

        if 1 not in labels:

            label=False

        all_collection.append(result)
        out_f.write("==========================\n\n")
        out_f.write("Query:{}\n".format(result["query"]))
        out_f.write("Golden:{}\n".format(result["golden"]))
        out_f.write("Trigger:{}\n".format(label))
        out_f.write("Trie_label:{}\n".format(result["trie_label"]))
        out_f.write("Pred:\n")
        for pred in result["generate_result"]:
            out_f.write(pred + "\n")


    return all_collection

def get_distinct_and_golden(inname,topks):
    in_f = open(inname,encoding="utf-8").readlines()

    all_results = {}
    for i,line in enumerate(in_f):
        sample = json.loads(line)
        query = sample["input"]
        if i%100000==0:
            print(i)
        if label_dict[query+"[seq]"+sample["golden"]]=="bad":
            continue

        if filter_rule(query):
            continue

        try:
            trie_label = in_trie[query+"[seq]"+sample["golden"]]
        except:
            #print(query+"[seq]"+sample["golden"])
            continue

        if trie_label==False:
            continue

        guid = sample["guid"]
        if guid not in all_results:
            all_results[guid] = {"golden":sample["golden"],
                                 "pred":[],
                                 "generate_result":[],
                                 "query":sample["input"],
                                 "trie_label":trie_label
                                 }
        golden = sample["golden"]
        generated = sample["generated"]
        generated = " ".join(generated).replace(" ##","")

        label=0
        if golden.strip()==generated.strip():
            label=1

        all_results[guid]["pred"].append(label)
        all_results[guid]["generate_result"].append(generated)


    print("total_good_filter_samples:{}".format(len(all_results)))
    collect_bad(all_results,inname+"bad_case20")

    print("{}\t{}\t{}".format("topk", "map", "triggerate"))
    for topk in topks:
        map,triggerate = calc_map(all_results,topk=topk)
    # all_results[query][0][sample["golden"]]=1
    # generated = sample["generated"]
    # all_results[query][1].append()
        print("{}\t{}\t{}".format(topk,map,triggerate))
        #print(map)
        #print(triggerate)


def find_not_in_trie(inname):
    in_f = open(inname, encoding="utf-8").readlines()

    bad=0
    for i,line in enumerate(in_f):

        result = json.loads(line)
        golden = result["golden"]

        if label_dict[result["query"]+"[seq]"+golden[0]]=="bad":
            continue

        if golden[1]==False:
            bad+=1

    print(bad)




def refine_result(inname):
    in_f = open(inname, encoding="utf-8").readlines()
    out_f = open(inname+".samples","w",encoding="utf-8")

    bad = 0
    for i, line in enumerate(in_f):

        result = json.loads(line)
        if label_dict[result["query"]+"[seq]"+result["golden"]]=="bad":
            continue
        out_f.write("==========================\n\n")
        out_f.write("Query:{}\n".format(result["query"]))
        out_f.write("Golden:{}\n".format(result["golden"]))
        out_f.write("In trie:{}\n".format(result["intrie"]))
        out_f.write("Pred:\n")
        for pred in result["pred"]:
            out_f.write(pred + "\n")



def merge_filtes():
    pass


def differ_results(inname1,inname2):
    in_f = open(inname1, encoding="utf-8").readlines()
    result1 = {}
    for line in in_f:
        line = json.loads(line)
        query = line["query"]
        golden = line["golden"]
        trie_label = line["trie_label"]
        pred = line["pred"]
        trigger = line["trigger"]

        result1[query+"[seq]"+golden] = [trigger,trie_label,pred]

    in_f2 = open(inname2, encoding="utf-8").readlines()
    out_f = open(inname2+".bad","w", encoding="utf-8")
    for line in in_f2:
        line = json.loads(line)
        query = line["query"]
        golden = line["golden"]
        trie_label = line["trie_label"]
        pred = line["pred"]
        trigger = line["trigger"]

        trigger1, trie_label1, pred1 = result1[query+"[seq]"+golden]
        if trigger1==True and trigger==False:
            out_f.write("==========================\n\n")
            out_f.write("Query:{}\n".format(line["query"]))
            out_f.write("Golden:{}\n".format(line["golden"]))
            out_f.write("Trie_label:{}\n".format(line["trie_label"]))
            out_f.write("Pred:\n")
            for i,pred in enumerate(line["pred"]):
                out_f.write(pred +"\t"+pred1[i]+"\n")
        else:
            continue


if __name__=="__main__":
    #inname = r"D:\data\production\keywords\test\all_parts\all_result\new_data_beam.tsv"
    inname1 = r"D:\data\production\keywords\test\all_parts\beam_result.tsv"
    #inname2 = r"D:\data\production\keywords\test\all_parts\beam_result_new_ensemble.tsv"
    get_distinct_and_golden(r"D:\data\production\keywords\test\fix_bug_new\beam_result-jianjiao.tsv", [1, 3, 5, 10, 20, 30, 40, 50])
    # get_distinct_and_golden(r"D:\data\production\keywords\test\fix_bug_new\beam_result_old-5.tsv",
    #                         [1, 3, 5, 10, 20, 30, 40, 50])
    # get_distinct_and_golden(
    #     r"D:\data\production\keywords\test\all_parts\part1_test_result\beam_result_ensemble0.tsv",
    #     [1, 3, 5, 10, 20, 30, 40, 50])
    # get_distinct_and_golden(
    #     r"D:\data\production\keywords\test\all_parts\part1_test_result\beam_result_jianjiao.tsv",
    #     [1, 3, 5, 10, 20, 30, 40, 50])
    # get_distinct_and_golden(
    #     r"D:\data\production\keywords\test\all_parts\part1_test_result\beam_result_ship1.tsv",
    #     [1, 3, 5, 10, 20, 30, 40, 50])
    # get_distinct_and_golden(
    #     r"D:\data\production\keywords\test\all_parts\part1_test_result\beam_result_ship2.tsv",
    #     [1, 3, 5, 10, 20, 30, 40, 50])
    #get_distinct_and_golden(inname2, [1, 3, 5, 10, 20, 30, 40, 50])

    # differ_results(inname1+"bad_case20.json",inname2+"bad_case20.json")
    # differ_results(inname2+"bad_case20.json", inname1+"bad_case20.json")

    #find_not_in_trie(inname+"bad_case_trie_check.json")

    #find_in_trie  trie generator
    #refine_result(inname+"bad_case.json.withlabel")