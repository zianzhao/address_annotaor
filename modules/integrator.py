#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
from random import randint
from sklearn.externals import joblib

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel

# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    voting
    Purpose:
        integrating the results of hybrid model
    Parameters:
        list results - extraction result of sub-models 
        list infos - types of info. need to be extracted
    Returns:
        dict integrated - extraction result after integration
    Author:
        Zian zhao
        5.13.2018
"""


def voting(results, infos):
    integrated = copy.deepcopy(results[0])

    # find all possible extracted info.
    for info in infos:
        label_dic = {}
        for result in results:
            if result[info] != '':
                if result[info] not in label_dic.keys():
                    label_dic[result[info]] = {
                        'count': 1,
                        'confidence': [result["%s_confidence" % info]],
                        'position':  [result["%s_p" % info]]
                    }
                else:
                    label_dic[result[info]]['count'] += 1
                    label_dic[result[info]]['confidence'].append(result["%s_confidence" % info])
                    label_dic[result[info]]['position'].append(result["%s_p" % info])

        counts = [label_dic[item]['count'] for item in label_dic]

        # sub-models disagree with each other
        if len(counts) > 1:
            if len(counts) != len(list(set(counts))):
                m = max(counts)
                if counts.count(m) == 1:
                    for item in label_dic:
                        if label_dic[item]['count'] == m:
                            integrated[info] = item
                            integrated['%s_confidence' % info] = label_dic[item]['confidence'][0]
                            integrated['%s_p' % info] = label_dic[item]['position'][0]
                        break
                else:
                    cm = 0
                    for item in label_dic:
                        if label_dic[item]['count'] == m:
                            conf = 1.0
                            for c in label_dic[item]['confidence']:
                                conf *= (1 - c)

                            r = randint(1, 2)
                            if cm < conf or (cm == conf and r == 1):
                                integrated[info] = item
                                integrated['%s_confidence' % info] = 1 - conf
                                lows = [p[0] for p in label_dic[item]['position']]
                                highs = [p[1] for p in label_dic[item]['position']]
                                integrated['%s_p' % info] = [max(lows), min(highs)]
                                cm = conf
        # models agree with each other
        # update confidence value
        elif len(counts) == 1:
            for item in label_dic:
                integrated[info] = item

                conf = 1.0
                for c in label_dic[item]['confidence']:
                    conf *= (1 - c)
                integrated['%s_confidence' % info] = 1 - conf

                lows = [p[0] for p in label_dic[item]['position']]
                highs = [p[1] for p in label_dic[item]['position']]
                integrated['%s_p' % info] = [max(lows), min(highs)]
                break

    return integrated


"""
    stacking
    Purpose:
        stacking integrator using LR classifiers 
    Parameters:
        string text - raw address
        list results - extraction result of sub-models
        list infos - types of info. need to be extracted
    Returns:
        dict integrated - extracted result after integration
    Author:
        Zian zhao
        5.21.2018
"""


def stacking(text, results, infos):
    integrated = copy.deepcopy(results[0])

    d2v = TfidfModel.load('./modules/models/tfidf.model')
    dct = Dictionary.load('./modules/models/dic.model')
    lsi = LsiModel.load('./modules/models/lsi.model')

    # generate feature vector
    text = u' '.join(unicode(text))
    word_list = text.split()
    corpus = dct.doc2bow(word_list)
    sent_feature = [item[1] for item in lsi[d2v[corpus]]]

    x = list()
    x += sent_feature
    for info in infos:
        for result in results:
            x.append(len(unicode(result['%s' % info]))/10.0)
            try:
                pos = result['%s_p' % info][0]
            except IndexError:
                pos = 0
            try:
                x.append(pos/float(len(unicode(text))))
            except ZeroDivisionError:
                x.append(0)
            x.append(result['%s_confidence' % info])

    # predict every type of info.
    for info in infos:
        probs = list()
        for i in range(len(results)):
            model = joblib.load('./modules/models/integrator_%s%s.model' % (info, i))
            # print model.predict_proba(x)
            probs.append(model.predict_proba([x])[0][1])

        y = probs.index(max(probs))
        integrated[info] = results[y][info]
        integrated['%s_p' % info] = results[y]['%s_p' % info]

        conf = 1.0
        for result in results:
            conf *= (1 - result['%s_confidence' % info])
        integrated['%s_confidence' % info] = 1 - conf

    return integrated


"""
    pos_check
    Purpose:
        checking whether segments overlap after integration 
    Parameters:
        dict ann - extraction result after integration 
        list infos - types of info. need to be extracted
    Returns:
        dict ann - extracted result after checking
    Author:
        Zian zhao
        5.15.2018
"""


def pos_check(ann, infos):
    for i in range(len(infos)-1):
        if len(ann['%s_p' % infos[i]]) * len(ann['%s_p' % infos[i+1]]):
            # check whether neighboring entities overlap
            if ann['%s_p' % infos[i]][1] >= ann['%s_p' % infos[i+1]][0]:
                conf1 = ann['%s_confidence' % infos[i]]
                conf2 = ann['%s_confidence' % infos[i+1]]
                r = randint(1, 2)
                if conf1 < conf2 or (conf1 == conf2 and r == 1):
                    ann['%s_p' % infos[i]][1] = ann['%s_p' % infos[i + 1]][0] - 1
                else:
                    ann['%s_p' % infos[i + 1]][0] = ann['%s_p' % infos[i]][1] + 1

        elif i < len(infos)-2 and len(ann['%s_p' % infos[i+1]]) == 0 and len(ann['%s_p' % infos[i]]) * len(ann['%s_p' % infos[i+2]]):
            # check whether entities overlap if neighboring entity does not exist
            if ann['%s_p' % infos[i]][1] >= ann['%s_p' % infos[i + 2]][0]:
                conf1 = ann['%s_confidence' % infos[i]]
                conf2 = ann['%s_confidence' % infos[i + 2]]
                r = randint(1, 2)
                if conf1 < conf2 or (conf1 == conf2 and r == 1):
                    ann['%s_p' % infos[i]][1] = ann['%s_p' % infos[i + 2]][0] - 1
                else:
                    ann['%s_p' % infos[i + 2]][0] = ann['%s_p' % infos[i]][1] + 1
            i += 1

    # check whether entity remains after merging
    for info in infos:
        if len(ann['%s_p' % info]) and ann['%s_p' % info][0] >= ann['%s_p' % info][1]:
            ann['%s_p' % info] = []
            ann['%s_confidence' % info] = 0.0
            ann[info] = ''

    return ann


