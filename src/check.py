# -*- coding: UTF-8 -*-
import codecs
import sys
import re
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def main():
    args = sys.argv[1:]
    result_file = args[0]
    test_file= args[1]

    
    re =codecs.open(result_file, 'r', 'utf-8')
    te =codecs.open(test_file, 'r', 'utf-8')
    result = re.read()
    ture = te.read()
    result_list = result.split('\r')
    result_list = result.split('\n')
    result_list = result.split(' ')

    rx=[]
    
    for rl in result_list:
        rx.append((rl[:rl.rfind('/')],rl[rl.rfind('/')+1:]))
    

    ture_list = ture.split('\r')
    ture_list = ture.split('\n')
    ture_list =ture.split(' ')

    tx=[]

    for tl in ture_list:
        tx.append((tl[:tl.rfind('/')],tl[tl.rfind('/')+1:]))

    rr=[]
    xx=[]
    count=0
    for i in range(0,len(tx)):
        for j in range(i,len(rx)):
            t1,t2=tx[i]
            r1,r2=rx[j]
            if(t1==r1):
                rr.append(r2)
                xx.append(t2)
                break
    
    print("acc:   "+str(accuracy_score(rr, xx)))
    #print("recall:"+str(recall_score(rr, xx,average='micro') ))
    #print("F:     "+str(f1_score(rr, xx,average='micro') ))

        


if __name__ == "__main__":
    main()