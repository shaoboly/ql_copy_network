import json
import math

#THRESHOLD=0.005
THRESHOLD=0.0

input_name = r'D:\data\production\keywords\compare\small_model_result.tsv'
f = open(input_name, 'r', encoding='utf-8')
f_out = open(input_name+'_good.txt', 'w', encoding='utf-8')
f_bad = open(input_name+'_bad.txt', 'w', encoding='utf-8')
end_flag = True

c_index=0
pre_input=None
while end_flag:
    tokens = []
    scores = []
    if c_index%10000==0:
        print(c_index)
    c_index+=1
    for _ in range(50):
        line = f.readline()
        #print(line)
        try:
            data = json.loads(line.strip())
        except:
            end_flag = False
            break
        t = data['generated']
        t.append('<eos>')
        s = data['scores'][1:len(t)+1]
        s = [math.exp(float(ss)) for ss in s]
        tokens.append(t)
        scores.append(s)
    for t,s in zip(tokens, scores):
        assert len(t) == len(s)
        least_score = 1
        f_score = 1
        flag = True
        for tt, zz in zip(t,s):
            if '##' in tt:
                f_score = f_score + zz
            else:
                if f_score < THRESHOLD:
                    flag = False
                    break
                f_score = zz
        if f_score < THRESHOLD:
            flag = False
        ttt = ' '.join(t[:-1])
        ttt = ttt.replace(' ##', '')
        if flag:
            f_out.write('{}\t{}\n'.format(data['input'], ttt))
        else:
            f_bad.write('{}\t{}\n'.format(data['input'], ttt))