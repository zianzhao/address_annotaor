#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn.externals import joblib


# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    tagger
    Purpose:
        NER via CRF
    Parameters:
        [in] string text - raw test
        
    Returns:
        None
    Author:
        Zian zhao
        5.11.2018
"""


def tagger(text):
    # tags = ['O', 'C_b', 'C_m', 'C_e',
    #         'P_b', 'P_m', 'P_e',
    #         'A_b', 'A_m', 'A_e']

    # dict to store extracted information and corresponding positions
    address = {'province': '', 'city': '', 'area': '',
               'province_p': [], 'city_p': [], 'area_p': []}

    crf = joblib.load('./modules/models/crf_tagger.model')

    x = [sent2features(text)]
    y_pred = crf.predict(x)
    y_pred = y_pred[0]

    # extracting city
    if 'C_b' in y_pred and 'C_e' in y_pred:
        low = y_pred.index('C_b')
        high = y_pred.index('C_e')
        flag = 1
        for i in range(low+1, high):
            if y_pred[i] != 'C_m' and y_pred[i] != 'O':
                flag = 0
                break

        if flag:
            address['city'] = ''.join(unicode(text)[low:high+1])
            address['city_p'] = [low, high]

    # extracting province
    if 'P_b' in y_pred and 'P_e' in y_pred:
        low = y_pred.index('P_b')
        high = y_pred.index('P_e')

        flag = 1
        for i in range(low+1, high):
            if y_pred[i] != 'P_m' and y_pred[i] != 'O':
                flag = 0
                break

        if flag:
            address['province'] = ''.join(unicode(text)[low:high+1])
            address['province_p'] = [low, high]

    # extracting area
    if 'A_b' in y_pred and 'A_e' in y_pred:
        low = y_pred.index('A_b')
        high = y_pred.index('A_e')
        flag = 1
        for i in range(low+1, high):
            if y_pred[i] != 'A_m' and y_pred[i] != 'O':
                flag = 0
                break

        if flag:
            address['area'] = ''.join(unicode(text)[low:high+1])
            address['area_p'] = [low, high]

    # todo uncomment for street-level information extraction
    # if 'S_b' in y_pred and 'S_e' in y_pred:
    #     low = y_pred.index('S_b')
    #     high = y_pred.index('S_e')
    #     flag = 1
    #     for i in range(low+1, high):
    #         if y_pred != 'S_m' and y_pred != 'O':
    #             flag = 0
    #             break
    #
    #     if flag:
    #         address['street'] = ''.join(text.split()[low:high+1])

    return address


"""
    sent2feature
    Purpose:
        convert sentence into list of features
    Parameters:
        [in] string sent - raw sentence

    Returns:
        list of features for each word
    Author:
        Zian zhao
        5.11.2018
"""


def sent2features(sent):
    sent = unicode(sent)
    return [word2features(sent, i) for i in range(len(sent))]


"""
    word2features
    Purpose:
        convert word into features
    Parameters:
        [in] string sent - raw sentence
        [in] int i - the position of word in raw sentence

    Returns:
        dictionary of features
    Author:
        Zian zhao
        5.11.2018
"""


def word2features(sent, i):
    word = sent[i]

    features = {
        'word': word,
        'word.isdigit': '0' <= word <= '9',
        'word.ischaracter': 'a' <= word <= 'z' or 'A' <= word <= 'Z',
    }
    # if not beginning of the sentence
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word': word1,
        })
    else:
        features['BOS'] = True
    # if not end of the sentence
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word': word1,
        })
    else:
        features['EOS'] = True

    return features
