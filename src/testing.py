#! /usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import sys
from tqdm import tqdm
from maxent import MaxentModel


def get_near_char(contents, i, times):
    words_len = len(contents) / times;
    if (i < 0 or i > words_len - 1):
        return '_'
    else:
        return contents[i*times]
 
def get_near_tag(contents, i ,times):
    words_len = len(contents) / times;
    if (i < 0 or i > words_len - 1):
        return '_'
    else:
        return contents[i*times*2]
 
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


def max_prob(label_prob_list):
    max_prob = 0
    max_prob_label = ''
    for label_prob in label_prob_list:
        if label_prob[1] > max_prob:
            max_prob = label_prob[1]
            max_prob_label = label_prob[0]
 
    return max_prob_label


def get_feature(test_file_path, feature_file_path):
    f = codecs.open(test_file_path, 'r', 'utf-8')
    contents = f.read()
    contents_list = contents.split('\r')
    contents_list = contents.split('\n')
    contents_list.remove('')
    contents_list.remove('')
 
    fout = codecs.open(feature_file_path, 'w', 'utf-8')
    for line in tqdm(contents_list):
        words_len = len(line)
        feature_list = []
 
        index = range(0, words_len)
        for i in index:
            pre_char = get_near_char(line, i-1, 1)
            pre_pre_char = get_near_char(line, i-2, 1)
            cur_char = get_near_char(line, i, 1)
            next_char = get_near_char(line, i+1, 1)
            next_next_char = get_near_char(line, i+2, 1)
            fout.write(
                  'C-2=' + pre_pre_char + ' ' 
                + 'C-1=' + pre_char + ' '
                + 'C0=' + cur_char + ' '
                + 'C1=' + next_char + ' ' + 'C2=' + next_next_char + ' '
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

        fout.write('split\r\n')
    fout.close()
 


# 标注测试集
def tag_test(test_feature_file, trained_model_file, tag_test_set_file):
    fin = codecs.open(test_feature_file, 'r', 'utf-8')
    fout = codecs.open(tag_test_set_file, 'w', 'utf-8')
    m = MaxentModel()
    m.load(trained_model_file)
    contents = fin.read()
    feature_list = contents.split('\r')
    feature_list.remove('\n')
    for feature in tqdm(feature_list):
        if (feature == 'split'):
            fout.write('\n\n\n')
            continue
        str_feature = []
        u_feature = feature.split(' ')
        for item in u_feature:
            str_feature.append(item.encode('utf-8'))
        label_prob_list = m.eval_all(str_feature)
        label = max_prob(label_prob_list)
 
        try:
            new_tag = str_feature[2].split('=')[1] + '/' + label
        except IndexError:
            print str_feature
        fout.write(new_tag.decode('utf-8'))
        pre_tag = label
 


def tag4_to_words(tag_training_set_file, result_file):
    fin = codecs.open(tag_training_set_file, 'r', 'utf-8')
    fout = codecs.open(result_file, 'w', 'utf-8')
 
    contents = fin.read()
    words_len = len(contents) / 3
    result = []
    i = 0
    while (i < words_len):
        cur_word_label = contents[i*3+2]
        cur_word = contents[i*3]
        if (cur_word_label == 'S'):
            result.append(cur_word + ' ')
        elif (cur_word_label == 'B'):
            result.append(cur_word)
        elif (cur_word_label == 'M'):
            result.append(cur_word)
        elif (cur_word_label == 'E'):
            result.append(cur_word + ' ')
        else:
            result.append(cur_word)
        i += 1
 
    fout.write(''.join(result))

def main():
    args = sys.argv[1:]
    model_file = args[0]
    test_file = args[1]
    result_file = args[2]


    # 测试集生成特征
    test_feature_file = test_file + ".feature"
    get_feature(test_file, test_feature_file)
    print 'get test set features succeed'

    # 标注测试集
    tag_test_set_file = test_file + ".tag"
    tag_test(test_feature_file, model_file, tag_test_set_file)
    print 'tag test set succeed'

    # 获取最终结果
    tag4_to_words(tag_test_set_file, result_file )
    print 'get final result succeed ' + result_file

if __name__ == "__main__":
    main()
