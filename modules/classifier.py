#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import Levenshtein


# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    entity_clf
    Purpose:
        Assigning extracted segments to predefined entities in KG
    Parameters:
        [in] dict ann - extraced segments
        [in] list entities - predefined entities
        [in] list candidates - processed entities
    Returns:
        dict labeled - extracted information 
    Author:
        Zian zhao
        5.12.2018
"""


def entity_clf(ann, candidates, entities):
    # list of stopwords
    ignore = ['回族自治区', '维吾尔自治区', '壮族自治区', '自治区', '新区', '省', '市',
              '区', '县', '街道', '居委会', '村', '乡', '自治州', '镇']

    labeled = copy.deepcopy(ann)
    infos = ['province', 'city', 'area']

    for info in infos:
        if len(labeled[info]):
            # delete the stop words
            for item in ignore:
                labeled[info] = labeled[info].replace(unicode(item), '')

            # find the most similar entity
            # using levenshtein distance
            pos = -1
            d = len(unicode(labeled[info]))
            d2 = len(unicode(labeled[info]))
            for i in range(len(candidates[info])):
                if len(candidates[info][i]):
                    tmp_d = Levenshtein.distance(unicode(candidates[info][i]), unicode(labeled[info]))
                    if tmp_d <= d:
                        d2 = d
                        d = tmp_d
                        pos = i

            if pos != -1:
                # compute confidence level with margin of similarity
                l = len(unicode(labeled[info]))
                labeled[info] = entities[info][pos]
                sim1 = 1 - float(d) / l
                sim2 = 1 - float(d2) / l
                margin = sim1 - sim2
                labeled['%s_confidence' % info] = margin
            else:
                labeled['%s_confidence' % info] = 0.0
        else:
            labeled['%s_confidence' % info] = 0.5

    return labeled
