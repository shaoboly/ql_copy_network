def merge_all(kd_name,our_name):
    kd_f = open(kd_name,encoding="utf-8").readlines()[1:]
    in_f = open(our_name,encoding="utf-8").readlines()[1:]

    out_f = open(our_name+".our_bad","w",encoding="utf-8")
    out_f_good = open(our_name + ".our_good", "w", encoding="utf-8")
    out_f.write("{}\t{}\t{}\t{}\t{}\n".format("query", "KDtree_recall", "KDtree_label", "Our_Recall", "Our_label"))
    good_result = {}
    bad_result = {}
    all_result ={}
    for i,(line1,line2) in enumerate(zip(kd_f,in_f)):
        OriQuery, RecalledQuery, JudgeCount, GoodCount, Label = line1.split('\t')
        OriQueryA, RecalledQueryA, JudgeCountA, GoodCountA, LabelA = line2.split('\t')

        Label = Label.strip()
        LabelA = LabelA.strip()

        assert OriQuery==OriQueryA

        if OriQuery not in good_result:
            good_result[OriQuery] = [[],[]]
            bad_result[OriQuery] = [[],[]]
            all_result[OriQuery] = [[],[]]

        all_result[OriQuery][0].append([RecalledQuery,Label])
        all_result[OriQuery][1].append([RecalledQueryA,LabelA])

        if Label=="1":
            good_result[OriQuery][0].append(RecalledQuery)
        else:
            bad_result[OriQuery][0].append(RecalledQuery)
        if LabelA=="1":
            good_result[OriQuery][1].append(RecalledQueryA)
        else:
            bad_result[OriQuery][1].append(RecalledQueryA)

    for k in good_result.keys():
        OriGood,ourGood = good_result[k]
        OriBad, ourBad = bad_result[k]
        Orilines, ourlines = all_result[k]
        if len(OriGood)>len(ourGood):
            for line1,line2 in zip(Orilines,ourlines):
                out_f.write("{}\t{}\t{}\t{}\t{}\n".format(k,line1[0],line1[1],line2[0],line2[1]))
        if len(OriGood)<len(ourGood):
            for line1,line2 in zip(Orilines,ourlines):
                out_f_good.write("{}\t{}\t{}\t{}\t{}\n".format(k,line1[0],line1[1],line2[0],line2[1]))


    print(len(kd_f))
    print(len(in_f))

merge_all(r"D:\data\production\query_index\kdtree.tsv",r"D:\data\production\query_index\generation.tsv")