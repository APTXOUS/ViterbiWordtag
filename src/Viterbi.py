# -*- coding: UTF-8 -*-
import codecs
import sys
import re
import os
from tqdm import tqdm
from sklearn.externals import joblib
import numpy as np
import scipy as sp
from sklearn import tree

tag_set_num = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'Mg', 
'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'Qg', 'q', 'Rg', 'r', 's', 'Tg', 't', 'Ug', 
'u', 'Vg', 'v', 'vd', 'vn', 'w', 'x', 'Yg', 'y', 'z','%']
A= [[0]*len(tag_set_num) for i in range(len(tag_set_num))] 
Pi= [0]*len(tag_set_num)
B = {}
Count = {}

window=3
tt=tree.DecisionTreeClassifier()

def get_predict(string):
    global tt
    return tt.predict(np.array(string))

for tag in tag_set_num:
    B[tag] = {}
    Count[tag] = 0

def get_word(words,now,index):
    return words[now+index]

def get_word_name(word):
    if (word[0] == ']') :
        return ']'

    i=word.rfind('/')
    return word[0:i]

def get_tag(word):
    if (word[0] == ']') :
        i=word.rfind(']')
        return word[i+1:]
        
    i=word.rfind('/')
    return word[i+1:]

def get_tag_num(tag):
    return tag_set_num.index(tag)

def inputWord(file_name,fout_name):
    f = codecs.open(file_name, 'r', 'utf-8')
    fout =codecs.open(fout_name+".a", 'w', 'utf-8')
    fout_b =codecs.open(fout_name+".b", 'w', 'utf-8')
    fout_pi =codecs.open(fout_name+".pi", 'w', 'utf-8')
    contents = f.read()
    contents_list = contents.split('\r')
    contents_list = contents.split('\n')
    contents_list.remove('')
    contents_list.remove('')
 
    for sentences in tqdm(contents_list):
        words=sentences.split(' ')
        i=0
        tt=[]
        word_list=[]
        for word in words:
            if (len(word)!=0):
                tt.append(get_tag_num(get_tag(word)))
                word_list.append(get_word_name(word))

        len_tt=len(tt)
        for i in range(0,len_tt-1):
            A[tt[i]][tt[i+1]]+=1
            Count[tag_set_num[tt[i]]] += 1

        if(len_tt>0):
            Pi[tt[0]]+=1
        for i in range(0,len_tt):
            if word_list[i] not in B[tag_set_num[tt[i]]]:
                B[tag_set_num[tt[i]]][word_list[i]] = 1.0
            else:
                B[tag_set_num[tt[i]]][word_list[i]] += 1

    len_tag=len(tag_set_num)  
    for i in range(0,len_tag):
        for j in range(0,len_tag):
            if(A[i][j] != 0.0 or Count[tag_set_num[i]] != 0):
                A[i][j]=A[i][j]*1.0/Count[tag_set_num[i]]

    sum_pi=sum(Pi)
    for i in range(0,len_tag):
        Pi[i]=Pi[i]*1.0/sum_pi

    for i in B:
        for word in B[i]:
            B[i][word] = B[i][word] / Count[i]

    fout.write(str(A))
    fout_b.write(str(B))
    fout_pi.write(str(Pi))

    fout_b.close()
    fout.close()   
    fout_pi.close()
 

def model_load(file_name): 
    fin = open(file_name, 'r')
    return eval(fin.read())
    
AA= [[0]*len(tag_set_num) for i in range(len(tag_set_num))] 
PP= [0]*len(tag_set_num)
BB = [{}]
#简单的解决 概率为 0 的问题
#->当p为0时，继承上一次的概率值
#  若为开头 概率为1，即平均分布
#->未定义词：概率为 所有词的最小概率的一半

def Viterbi(str):
    global AA
    global BB
    global PP
    V = [{}]
    path = {}
    str_list = str.strip().split(" ")

    min_value=1
    for i in BB:
        for word in BB[i]:
            min_value=min(BB[i][word],min_value)
    min_value=min_value/2.0
    for k in tag_set_num:
        V[0][k] = PP[get_tag_num(k)] * BB[k].get(str_list[0],min_value)
        path[k] = [k]

    sum_v=0
    for k in tag_set_num:
        sum_v+=V[0][k] 
    if(sum_v==0):
        for k in tag_set_num:
            V[0][k]=1 


    for m in range(1, len(str_list)):
        V.append({})
        add_path = {}
        lp=1
        for i in tag_set_num:
            (p, s) = max([(V[m - 1][j] * AA[get_tag_num(j)][get_tag_num(i)] * BB[i].get(str_list[m],min_value), j) for j in tag_set_num])
            if(p==0):
                #p = max([V[m - 1][j]for j in tag_set_num])
                p=lp*0.6
        
            V[m][i] = p
            lp=p
            add_path[i] = path[s] + [i]

        path = add_path

    (p, s) = max([(V[len(str_list) - 1][k], k) for k in tag_set_num])
    return (p, path[s])

def Viterbi_tree(str):
    global AA
    global BB
    global PP

    V = [{}]
    path = {}
    str_list = str.strip().split(" ")

    min_value=1
    for i in BB:
        for word in BB[i]:
            min_value=min(BB[i][word],min_value)
    min_value=min_value/2.0
    for k in tag_set_num:
        V[0][k] = PP[get_tag_num(k)] * BB[k].get(str_list[0],min_value)
        path[k] = [k]

    sum_v=0
    for k in tag_set_num:
        sum_v+=V[0][k] 
    if(sum_v==0):
        for k in tag_set_num:
            V[0][k]=1 


    for m in range(1, len(str_list)):
        V.append({})
        add_path = {}
        lp=1
        for i in tag_set_num:
            if(BB[i].get(str_list[m],0)==0):
                if(sum([BB[gg].get(str_list[m],0) for gg in tag_set_num])==0):
                    (pp, ss) = max([(V[m - 1][k], k) for k in tag_set_num])
                    ll=[]
                    if(m-3<0):
                        ll=[[47,tag_set_num.index(path[ss][m-2]),tag_set_num.index(path[ss][m-1])]]
                    elif(m-2<0):
                        ll=[[47,47,tag_set_num.index(path[ss][m-1])]]
                    elif(m-1<0):
                        ll=[[47,47,47]]
                    else:
                        ll=[[tag_set_num.index(path[ss][m-3]),tag_set_num.index(path[ss][m-2]),tag_set_num.index(path[ss][m-1])]]
                    pre=get_predict(ll)
                    (p, s) = max([(V[m - 1][j] * AA[get_tag_num(j)][get_tag_num(i)] * BB[i].get(str_list[m],2*min_value if i==tag_set_num[pre[0]] else min_value), j) for j in tag_set_num])
                else:
                    (p, s) = max([(V[m - 1][j] * AA[get_tag_num(j)][get_tag_num(i)] * BB[i].get(str_list[m],min_value), j) for j in tag_set_num])
            else:
                (p, s) = max([(V[m - 1][j] * AA[get_tag_num(j)][get_tag_num(i)] * BB[i].get(str_list[m],min_value), j) for j in tag_set_num])

            if(p==0):
                p=lp


            V[m][i] = p
            lp=p
            add_path[i] = path[s] + [i]

        path = add_path

    (p, s) = max([(V[len(str_list) - 1][k], k) for k in tag_set_num])
    return (p, path[s])

def tagWord(model_file,test_file,result_file):
    global AA
    global BB
    global PP
    AA = list(model_load(model_file+".a"))
    BB = dict(model_load(model_file+".b"))
    PP = list(model_load(model_file+".pi"))
    
    fin = codecs.open(test_file, 'r', 'utf-8')
    lines = fin.readlines()
    o = codecs.open(result_file, 'w', 'utf-8')

    for line in tqdm(lines):
        if(len(line)==0):
            o.write('\r\n')
        elif(line[0]=='\n'):
            o.write('\r\n')
        elif(line[0]=='\r'):
            o.write('\r\n') 
        elif(line[0]==' '):
            o.write('\r\n')      
        else:
            if(line.rfind(u'。')==len(line)-4):
                line = line[:line.rfind(u'。')] + line[line.rfind(u'。')+1:]
                p, pos_list = Viterbi_tree(line)
                line_list = line.strip().split(" ")
                for word in range(0, len(line_list)):
                    o.write(line_list[word] + '/' + pos_list[word] + ' ')
                o.write(u' 。'+'/w') 
                o.write('\r\n')
            else:
                p, pos_list = Viterbi_tree(line)
                line_list = line.strip().split(" ")
                for word in range(0, len(line_list)):
                    o.write(line_list[word] + '/' + pos_list[word] + ' ')
                o.write('\r\n')

    fin.close()
    o.close()

    




def main():
    args = sys.argv[1:]
    if(args[0]=="-model"):
        train_file = args[1]
        out_file= args[2]
        inputWord(train_file,out_file)
        return
    if(args[0]=="-test"):
        global tt
        model_file= args[1]
        test_file = args[2]
        result_file= args[3]
        tree_file=args[4]
        tt=joblib.load(tree_file)
        tagWord(model_file,test_file,result_file)
        return 
    print("command error")
        


if __name__ == "__main__":
    main()