
# -*- coding: UTF-8 -*-
import codecs
import sys
import re
from tqdm import tqdm
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sklearn import preprocessing

feature_names = [    
    'n1',    
    'n2',    
    'n3',    
    'n4',   
    'n5',       
    'result'
]



window=3

tag_set_num = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'Mg', 
'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'Qg', 'q', 'Rg', 'r', 's', 'Tg', 't', 'Ug', 
'u', 'Vg', 'v', 'vd', 'vn', 'w', 'x', 'Yg', 'y', 'z','%','_']



def get_tag(word):
    if (word[0] == ']') :
        i=word.rfind(']')
        return tag_set_num.index(word[i+1:])
        
    i=word.rfind('/')
    return tag_set_num.index(word[i+1:])

def get_word(list,now,index):
    if(now+index<0 or now+index>len(list)):
        return get_tag('_')
    else:
        return get_tag(list[now+index])



def inputWord(file_name,fout_name):
    f = codecs.open(file_name, 'r', 'utf-8')
    fout =codecs.open(fout_name, 'w', 'utf-8')

    contents = f.read()
    contents_list = contents.split('\r')
    contents_list = contents.split('\n')
    contents_list.remove('')
    contents_list.remove('')
 
    
    for sentences in tqdm(contents_list):
        words=sentences.split(' ')
        for i in range(0,len(words)):
            if(len(words[i])>0):
                str1=""
                for j in range(window,0,-1):
                    str1=str1+str(get_word(words,i,-j))+' '
                str1+=str(get_word(words,i,0))
                fout.write(str1+'\n')
            
    f.close()
    fout.close()

def Predict(data_file,model_file):
    data = []
    lable= []
    f = codecs.open(data_file, 'r', 'utf-8')
    contents = f.read()
    contents_list = contents.split('\r')
    contents_list = contents.split('\n')
    contents_list.remove('')

    print("load_model")
    for sentences in tqdm(contents_list):
        words=sentences.split(' ')
        tmp1=[]
        for i in range(0,len(words)-1):
            tmp1.append(int(words[i]))
        data.append((tmp1))
        lable.append(int(words[len(words)-1]))


    x = np.array(data)
    y = np.array(lable)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)  
    print clf.score(x_test,y_test)
    
    joblib.dump(clf,model_file)
    f.close()

tt=tree.DecisionTreeClassifier()

def get_predict(string):
    global tt
    return tt.predict(np.array(string))


def main():
    global tt
    args = sys.argv[1:]
    if(args[0]=='-model'):
        train_file = args[1]
        model_file=args[2]
        inputWord(train_file,model_file)

    if(args[0]=='-train'):
        data_file=args[1]
        model_file=args[2]
        Predict(data_file,model_file)
        
    if(args[0]=='-test'):
        model_file=args[1]
        tt=joblib.load(model_file)
        string = [[38 ,41,20 ]]
        print get_predict(string)
    
if __name__ == "__main__":
    main()     
            

        

 
