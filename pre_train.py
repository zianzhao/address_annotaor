#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import sklearn_crfsuite
from sklearn.externals import joblib
from modules.crf_tagger import sent2features

# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    pre_train
    Purpose:
        Train the CRF tagger with KB and active learning data
    Parameters:
        [in] object cursor - cursor object for mysql
        [in] list file_list - names of files for retrain
    Returns:
        None
    Author:
        Zian zhao
        5.13.2018
"""


def pre_train(cursor, file_list):

    stop_words = [['回族自治区', '维吾尔自治区', '壮族自治区', '省', '市', '自治区'],
                  ['市', '区',  '自治州'], ['县', '区', '市']]

    # initiate CRF model
    crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=True
            )

    x_train = list()
    y_train = list()

    # add data from files into training set
    for ff in file_list:
        infile = open('./data/' + ff)
        corpus = [line.split('\t')[:-1] for line in infile.readlines()]

        for line in corpus:
            tmp_x = sent2features(line[0])
            
            line[1] = line[1].replace('\'', '')
            line[1] = line[1].replace('[', '')
            line[1] = line[1].replace(']', '')
            line[1] = line[1].replace(' ', '')
            tmp_y = line[1].split(',')

            for i in range(100):
                x_train.append(tmp_x)
                y_train.append(tmp_y)

    print 'Retrain files loaded.'

    # get data from KB
    cursor.execute("select distinct province, city, area, street from geokg;")

    while True:
        data = cursor.fetchone()
        if data is None:
            break

        data2 = copy.deepcopy(list(data))
        data = list(data)

        for i in range(len(stop_words)):
            for item in stop_words[i]:
                data[i] = data[i].replace(item, '')

        d = [data, data2, ['', '', '']]

        # generate synthetic data
        # loop to combine 3-level data with & without stopwords
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    sentence = u''.join([d[i][0], d[j][1], d[k][2], d[0][3]])
                    tmp_x = sent2features(sentence)
                    x_train.append(tmp_x)

                    tmp_p = ['P_m'] * len(unicode(d[i][0]))
                    if len(unicode(d[i][0])):
                        tmp_p[0] = 'P_b'
                        tmp_p[-1] = 'P_e'

                    tmp_c = ['C_m'] * len(unicode(d[j][1]))
                    if len(unicode(d[j][1])):
                        tmp_c[0] = 'C_b'
                        tmp_c[-1] = 'C_e'

                    tmp_a = ['A_m'] * len(unicode(d[k][2]))
                    if len(unicode(d[k][2])):
                        tmp_a[0] = 'A_b'
                        tmp_a[-1] = 'A_e'

                    tmp_y = tmp_p + tmp_c + tmp_a + ['O'] * len(unicode(d[0][3]))
                    y_train.append(tmp_y)

    # train and save the CRF model
    crf.fit(x_train, y_train)
    joblib.dump(crf, './modules/models/crf_tagger.model')

    print "Pre_train finished."
