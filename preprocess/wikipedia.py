import os

def merge_data(dirname,count=100):
    out_f = open(os.path.join(dirname,"merge_{}p".format(count)),"w",encoding="utf-8")

    for i in range(count):
        in_f = open(os.path.join(dirname,"wikipedia_split_%04d" % i),encoding="utf-8").readlines()
        out_f.writelines(in_f)


# merge_data(r"/data/yegong/boshao/bert_corpus/wikipedia_sentences.tokenized")

def generate_question_data(inname):
    in_f = open(inname, encoding="utf-8")
    out_f = open(inname + ".question", "w", encoding="utf-8")

    questions = {}
    for i,line in enumerate(in_f):
        line = line.strip().split('|||')[:2]
        if len(line)<2:
            continue
        q1,q2 = line
        q1 = q1.strip().lower()
        questions[q1] = 1
        q2 = q2.strip().lower()
        questions[q2] = 1
        if i%100000==0:
            print(i)
        if len(questions)>1000000:
            break

    for q in questions.keys():
        out_f.write("mask\t{}\n".format(q))

generate_question_data(r"D:\data\Backup_STCA\data\seq2seq\MSPAD5W\qq.wikianswers")



def trans_input_to_sentence(inname):
    out_f = open(inname+".para2sent", "w", encoding="utf-8")
    in_f = open(inname, encoding="utf-8").readlines()

    i=0
    cur_line = []
    while i<len(in_f):
        line = in_f[i]




def concate_input(refdir,autodir,noisedir1,noisedir2,noisedir3):
    out_f = open(os.path.join(refdir, "all_format.txt"),"w",encoding="utf-8")
    for i in range(80):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        pred = [line.strip() for line in open(os.path.join(autodir, filename),encoding="utf-8").readlines()]
        n1 = [line.strip() for line in open(os.path.join(noisedir1, filename),encoding="utf-8").readlines()]
        n2 = [line.strip() for line in open(os.path.join(noisedir2, filename),encoding="utf-8").readlines()]
        n3 = [line.strip() for line in open(os.path.join(noisedir3, filename), encoding="utf-8").readlines()]

        refs = [line.strip() for line in open(os.path.join(refdir, refname),encoding="utf-8").readlines()]

        out_f.write("REF:{}\n".format(" ".join(refs)))
        out_f.write("PRD:{}\n".format(" ".join(pred)))
        out_f.write("NO1:{}\n".format(" ".join(n1)))
        out_f.write("NO2:{}\n".format(" ".join(n2)))
        out_f.write("NO3:{}\n".format(" ".join(n3)))

        out_f.write("\n\n\n")

# concate_input(r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-1/test-results/ref",
#               r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-1/test-results/pred/autoencode",
#                 r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-1/test-results/pred/noise",
#               r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-1/test-results/pred/noise1",
#               r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-1/test-results/pred/noise1_drop0.5")


def concate_l2r(refdir,noisedir1):
    out_f = open(os.path.join(noisedir1, "all_format.txt"),"w",encoding="utf-8")
    for i in range(200):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        n1 = [line.strip() for line in open(os.path.join(noisedir1, filename),encoding="utf-8").readlines()]


        refs = [line.strip() for line in open(os.path.join(refdir, refname),encoding="utf-8").readlines()]

        refs = " ".join(refs)
        refs = " ".join(refs.split()[:512])

        out_f.write("REF:{}\n".format(refs))
        out_f.write("PRD:{}\n".format(" ".join(n1)))


        out_f.write("\n\n\n")

import json
def concate_l2r_intoparallel(refdir,noisedir1,count=27000,reverse= False):
    out_f = open(os.path.join(noisedir1, "noise.parralel"),"w",encoding="utf-8")
    for i in range(count):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        n1 = [line.strip() for line in open(os.path.join(noisedir1, filename),encoding="utf-8").readlines()]


        refs = [line.strip() for line in open(os.path.join(refdir, refname),encoding="utf-8").readlines()]

        refs = " ".join(refs)
        refs = " ".join(refs.split()[:512])
        n1 = " ".join(n1)
        if reverse:
            refs = refs.split()
            refs.reverse()
            refs = " ".join(refs)

            n1 = n1.split()
            n1.reverse()
            n1 = " ".join(n1)

        #SENTENCE_START = '<s>'
        #SENTENCE_END = '</s>'

        #refs = "{} {} {}".format(SENTENCE_START, refs, SENTENCE_END)

        out_line = "{}\t{}\n".format(n1,refs)
        out_f.write(out_line)
        #out_f.write("REF:{}\n".format(refs))
        #out_f.write("PRD:{}\n".format(" ".join(n1)))

        #out_f.write("\n\n\n")


# concate_l2r_intoparallel(r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-l2r/test-results/ref",
#                          r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wiki-03-19-l2r/test-results/pred")
#
# concate_l2r_intoparallel(r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wikire-03-19-re/test-results/ref",
#                          r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wikire-03-19-re/test-results/pred",reverse=True)
#

# list_dir = [r"/data/yegong/boshao/summarize_data/noise_data/noise.parralel1",
#             r"/data/yegong/boshao/summarize_data/noise_data/noise.parralel2",
#             r"/data/yegong/boshao/summarize_data/noise_data/noise.parralel_re"]

def comcate(list_dir):
    dirname = os.path.dirname(list_dir[0])
    out_f = open(os.path.join(dirname,"all_noise"),"w",encoding="utf-8")

    for filename in list_dir:
        in_f = open(filename,encoding="utf-8").readlines()
        out_f.writelines(in_f)

#comcate(list_dir)

def concate_r2l(refdir,noisedir1):
    out_f = open(os.path.join(noisedir1, "all_format_reverse.txt"),"w",encoding="utf-8")
    for i in range(200):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        n1 = [line.strip() for line in open(os.path.join(noisedir1, filename),encoding="utf-8").readlines()]


        refs = [line.strip() for line in open(os.path.join(refdir, refname),encoding="utf-8").readlines()]

        refs = " ".join(refs)
        refs = " ".join(refs.split()[:512])
        n1 = " ".join(n1)

        refs = refs.split()
        refs.reverse()
        refs = " ".join(refs)


        n1 = n1.split()
        n1.reverse()
        n1 = " ".join(n1)

        out_f.write("REF:{}\n".format(refs))
        out_f.write("PRD:{}\n".format(n1))


        out_f.write("\n\n\n")


# concate_r2l(r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wikire-03-19-re/test-results/ref",
#             r"/data/yegong/boshao/summarize_data/bert_s2l_copy-wikire-03-19-re/test-results/pred")

