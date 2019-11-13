import json
def refine_train_and_test(inname):
    in_f = open(inname,encoding="utf-8")
    out_f_src = open(inname+".src","w",encoding="utf-8")
    out_f_tgt = open(inname + ".tgt", "w", encoding="utf-8")
    for line in in_f:
        line = json.loads(line)
        out_f_src.write(line["article"].strip()+"\n")
        out_f_tgt.write(line["title"].strip()+"\n")



# refine_train_and_test("/home/v-boshao/seass/topic_news/train.txt")
# refine_train_and_test("/home/v-boshao/seass/topic_news/dev.txt")
# refine_train_and_test("/home/v-boshao/seass/topic_news/test.txt")