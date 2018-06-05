#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel

# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    crf_retrain
    Purpose:
        Generate the retrain data for CRF model
    Parameters:
        None
    Returns:
        None
    Author:
        Zian zhao
        5.15.2018
"""


def crf_retrain():

    labels = ['P', 'C', 'A']

    infile = open('./data/al_feedback.tsv')
    infile2 = open('./outputs/i1/al_argu_i1.tsv')
    """
        data pattern:
            id \t raw_address \t correctness,* num. of target info.
            i.e.: id raw_address 0,1,-1
        correctness:
            1 correctly labelled
            -1 wrong label
            0 instance does not exist in raw data
    """

    outfile = open('./data/re_train.tsv', 'w')

    while True:
        feedbeck = infile.readline()
        ann = infile2.readline().split('\t')[:-1]

        if not len(feedbeck):
            break

        if '\r' in feedbeck:
            feedbeck = feedbeck.replace('\r', '')
        if '\n' in feedbeck:
            feedbeck = feedbeck.replace('\n', '')

        address_id = feedbeck.split('\t')[0]
        correctness = feedbeck.split('\t')[2:5]

        high_int = 0
        if address_id == ann[0]:
            sentence = str()
            tmp_y = list()

            low = ann[2].split(',')[1]
            high = ann[2].split(',')[2]

            if '-1' in correctness:
                pass
            else:
                # if all the labels are correct or not exists
                for i in range(3):
                    if low != 'NaN':
                        low = int(low)
                        high = int(high)
                        high_int = high

                        # add correct segments with corresponding label
                        if correctness[i] == '1':
                            sentence += unicode(ann[1])[low:high + 1]
                            tmp = ['%s_m' % labels[i]] * (high - low + 1)
                            tmp[0] = '%s_b' % labels[i]
                            tmp[-1] = '%s_e' % labels[i]
                            tmp_y += tmp
                        # add segments do not contain target info.
                        elif correctness[i] == '0':
                            sentence += unicode(ann[1])[low:high + 1]
                            tmp = ['O'] * (high - low + 1)
                            tmp_y += tmp

                    if i == 2:
                        break

                    low = ann[i + 3].split(',')[1]
                    if low != 'NaN':
                        low = int(low)
                        sentence += unicode(ann[1])[high_int + 1:low]
                        tmp_y += ['O'] * (low - high_int -1)

                    high = ann[i + 3].split(',')[2]

                sentence += unicode(ann[1])[high_int+1:]
                tmp_y += ['O'] * (len(unicode(ann[1])) - 1 - high_int)

                outfile.write(sentence + '\t' + str(tmp_y) + '\t\r\n')
        else:
            print 'ID mismatch!'
            return

    outfile.close()
    return


"""
    itegrator_retrain
    Purpose:
        Train the stacking integrator using AL feedback
    Parameters:
        None
    Returns:
        None
    Author:
        Zian zhao
        5.20.2018
"""


def integrator_retrain():
    infile = open('./data/al_itg_feedback.tsv')
    infile2 = open('./outputs/i3/al_itg_selected.tsv')
    """
        data pattern:
            id \t raw_address \t correctness for each sub-model * num. of target info.
            i.e.: id raw_address 0,1 0,1 0,1
        correctness:
            1 target info. correctly extracted
            0 extraction fails
    """

    d2v = TfidfModel.load('./modules/models/tfidf.model')
    dct = Dictionary.load('./modules/models/dic.model')
    lsi = LsiModel.load('./modules/models/lsi.model')

    infos = ['province', 'city', 'area']

    x_train = list()
    y_train = dict()

    for info in infos:
        for h in range(2):
            y_train['%s%s' % (info, h)] = list()

    while True:
        feedbeck = infile.readline()
        ann = infile2.readline().split('\t')[:-1]

        if not len(feedbeck):
            break

        if '\r' in feedbeck:
            feedbeck = feedbeck.replace('\r', '')
        if '\n' in feedbeck:
            feedbeck = feedbeck.replace('\n', '')

        feedbeck = feedbeck.split('\t')
        address_id = feedbeck[0]

        if address_id != ann[0]:
            print 'ID mismatch!'
            return

        text = u' '.join(unicode(ann[1]))
        word_list = text.split()

        corpus = dct.doc2bow(word_list)
        tmp_x = [item[1] for item in lsi[d2v[corpus]]]
        """
            x includes: 
            vector representation of address
            defined features & confidence
            for each target info. and sub-model
        """

        for i in range(len(infos)):
            # for each info.
            results = ann[i + 5].split(';')
            for j in range(len(results)):
                # for each sub-model
                line = results[j].split(',')
                tmp_x.append((int(line[1])-int(line[0])) / 10.0)    # normalized length
                try:
                    tmp_x.append(int(line[0]) / float(len(unicode(ann[1]))))    # position of the segment
                except ZeroDivisionError:
                    tmp_x.append(0)
                tmp_x.append(float(line[2]))

            fb = feedbeck[i + 2].split(',')
            if int(fb[0]) == 1:
                y_train['%s0' % infos[i]].append(1)
            else:
                y_train['%s0' % infos[i]].append(0)

            if int(fb[1]) == 1:
                y_train['%s1' % infos[i]].append(1)
            else:
                y_train['%s1' % infos[i]].append(0)

        x_train.append(tmp_x)

    # train classifier for each sub-model and target info.
    for item in y_train:
        clf = LogisticRegression(n_jobs=-1)
        clf.fit(x_train, y_train[item])
        joblib.dump(clf, './modules/models/integrator_%s.model' % item)

    return

