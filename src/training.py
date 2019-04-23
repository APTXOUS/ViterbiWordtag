#! /usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import sys
from tqdm import tqdm
#from maxent import MaxentModel

def tag4_training_set(file_name,file_name_store):
    fin = codecs.open(file_name, 'r', 'utf-8')
    file_origin = fin.read()
    file_origin = file_origin.replace(u'\r', u' ')
    file_origin = file_origin.replace(u'\n', u' ')
    
    words = file_origin.split(' ')
    print len(words)
    fout = codecs.open(file_name_store, 'w', 'utf-8')

    i = 0
    for word in tqdm(words):
        i += 1
        wordlen=len(word)
        if(i % 100 == 0): 
            fout.write(u'\r')
        if(wordlen == 0):
            continue
        if(wordlen == 1):
            tag_word = word + '/S'
        elif (wordlen== 2):
            tag_word = word[0] + '/B' + word[1] + '/E'
        else:
            tag_word = word[0] + '/B' 
            mid_words = word[1:-1]
            for mid_word in mid_words:
                tag_word += (mid_word + '/M')
            tag_word += (word[-1] + '/E')
        fout.write(tag_word)
    fout.close()


def get_near_char(file_origin, i, times):
    words_len = len(file_origin) / times;
    if (i < 0 or i > words_len - 1):
        return '_'
    else:
        return file_origin[i*times]
 
def get_near_tag(file_origin, i ,times):
    words_len = len(file_origin) / times;
    if (i < 0 or i > words_len - 1):
        return '_'
    else:
        return file_origin[i*times*2]
 
def isPu(char):
    punctuation = [u'，', u'。', u'？', u'！', u'；', u'－－', u'、', u'——', u'（', u'）', u'《', u'》', u'：', u'“', u'”', u'’', u'‘']
    if char in punctuation:
        return '1'
    else:
        return '0'
 
def get_class(char):
    zh_num = [u'零', u'○', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'百', u'千', u'万']
    ar_num = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'.', u'０',u'１',u'２',u'３',u'４',u'５',u'６',u'７',u'８',u'９']
    date = [u'日', u'年', u'月']
    letter = ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    level=[u'级', u'届']
    if char in zh_num or char in ar_num:
        return '1'
    elif char in date:
        return '2'
    elif char in letter:
        return '3'
    elif char in level:
        return '5'
    else:
        return '4'
        


# 获取训练集特征
def get_feature(tag_file_path, event_file_path):
    f = codecs.open(tag_file_path, 'r', 'utf-8')
    file_origin = f.read()
    file_origin = file_origin.replace(u'\r', u'')
    file_origin = file_origin.replace(u'\n', u'')
    words_len = len(file_origin)/3
    
    event_list = []

    fout = codecs.open(event_file_path, 'w', 'utf-8')
 
    index = range(0, words_len)
    for i in tqdm(index):
        pre_char = get_near_char(file_origin, i-1, 3)
        pre_pre_char = get_near_char(file_origin, i-2, 3)
        cur_char = get_near_char(file_origin, i, 3)
        next_char = get_near_char(file_origin, i+1, 3)
        next_next_char = get_near_char(file_origin, i+2, 3)
        fout.write(
            file_origin[i*3+2] + ' '
            + 'C-2='+pre_pre_char + ' ' 
            + 'C-1='+pre_char + ' '
            + 'C0=' + cur_char + ' '
            + 'C1=' + next_char + ' ' 
            + 'C2=' + next_next_char + ' '
            + 'C-2=' + pre_pre_char + 'C-1=' + pre_char + ' '
            + 'C-1=' + pre_char + 'C0=' + cur_char + ' '
            + 'C0=' + cur_char + 'C1=' + next_char + ' '
            + 'C1=' + next_char + 'C2=' + next_next_char + ' '
            + 'C-1=' + pre_char + 'C1=' + next_char + ' '
            + 'C-2=' + pre_pre_char + 'C-1=' + pre_char + 'C0=' + cur_char + ' '
            + 'C-1=' + pre_char + 'C0=' + cur_char + 'C1=' + next_char + ' '
            + 'C0=' + cur_char + 'C1=' + next_char + 'C2=' + next_next_char + ' '
            + 'C-2=' + pre_pre_char + 'C-1=' + pre_char + 'C0=' + cur_char +'C1=' +next_char+' '
            + 'C-1=' + pre_char + 'C0=' + cur_char + 'C1=' + next_char +'C2=' +next_next_char+' '
            + 'Pu=' + isPu(cur_char) + ' '
            + 'Tc-2=' + get_class(pre_pre_char) + 'Tc-1=' + get_class(pre_char)
            + 'Tc0=' + get_class(cur_char) + 'Tc1=' + get_class(next_char)
            + 'Tc2=' + get_class(next_next_char) + ' '
            + '\r')
    fout.close()
 

def training(file_feature_name, file_model_name, times):
    m = MaxentModel()
    fin = codecs.open(file_feature_name, 'r', 'utf-8')
    all_list = []
    print 'begin add_event'


    m.begin_add_event()
    for line in tqdm(fin):
        line = line.rstrip()
        line_list = line.split(' ')
        str_list = []
        for item in line_list:
            str_list.append(item.encode('utf-8'))
        all_list.append(str_list)
        m.add_event(str_list[1:], str_list[0], 1)
    m.end_add_event()


    print 'end add_event'

    print 'begin training'
    m.train(times, "gis")
    print 'end training'

    m.save(file_model_name)
    return all_list


def main():
    args = sys.argv[1:]
    training_file = args[0]
    result_file = args[1]
    time=args[2]

    print training_file 
    print result_file
    print time
 
    tag_training_set_file=training_file +'.tag'
    tag4_training_set(training_file, tag_training_set_file)
    print 'tag train set succeed'
 
    file_feature_name = training_file + ".feature"
    get_feature(tag_training_set_file, file_feature_name)
    print 'get training set feature succeed'

    file_model_name= result_file
    training(file_feature_name, file_model_name, int(time))
    print 'generate model succeed'


if __name__=="__main__":
    main()