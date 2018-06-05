#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pymysql
import copy
import random

from modules import coarse_parser
from modules.pre_processor import doc_stats
from modules import crf_tagger
from pre_train import pre_train
from modules import integrator
from modules.classifier import entity_clf


# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


# -------------------------------------------------
"""
    export
    Purpose:
        export the annotation result according to export types
    Parameters:
        [in] dict ann - integrated annotation result
        [in] list anns - list of annotation results from multiple models
        [in] list exp_infos - target information
        [in] string raw - raw address
        [in] int add_id - id for the address
        [in] object outfile - object of the output file
        [in] string typ - selected export types 
    Returns:
        None
    Author:
        Zian zhao
        5.20.2018
"""


def export(ann, anns, exp_infos, raw, add_id, outfile, typ):
    outstr = add_id + '\t' + raw + '\t'

    # export data used for evaluation
    if typ == 'eva':
        for inf in exp_infos:
            outstr += str(ann[inf]) + '['
            for i in range(len(anns)-1):
                outstr += str(anns[i][inf]) + ','
            outstr += str(anns[-1][inf])
            outstr += ']' + '\t'
    # export data used for active learning
    elif typ == 'al':
        for inf in exp_infos:
            outstr += str(ann[inf])

            try:
                outstr += ',' + str(ann['%s_p' % inf][0])
                outstr += ',' + str(ann['%s_p' % inf][1]) + '\t'
            except IndexError:
                outstr += ',' + 'NaN'
                outstr += ',' + 'NaN' + '\t'

        for inf in exp_infos:
            outstr += '['
            for i in range(len(anns)-1):
                outstr += str(anns[i][inf]) + ','
            outstr += str(anns[-1][inf])
            outstr += ']' + '\t'
    # export data used for training stacking integrator
    elif typ == 'itg':
        for inf in exp_infos:
            outstr += str(ann[inf]) + '['
            for i in range(len(anns)-1):
                outstr += str(anns[i][inf]) + ','
            outstr += str(anns[-1][inf])
            outstr += ']' + '\t'

        for inf in exp_infos:
            for i in range(len(anns)):
                try:
                    outstr += str(anns[i]['%s_p' % inf][0])
                    outstr += ',' + str(anns[i]['%s_p' % inf][1])
                except IndexError:
                    outstr += '0'
                    outstr += ',' + '0'
                outstr += ',' + str(anns[i]['%s_confidence' % inf])
                if i < len(anns)-1:
                    outstr += ';'

            outstr += '\t'

    outstr += '\r\n'
    outfile.write(outstr)
    return
# -------------------------------------------------


'''
module name: address_annotator
Description:
    Main section for address information extraction
Author:
Zian Zhao
5.10.2018
'''


# initialize the database
db = pymysql.connect(user='root', passwd='your_password', db='address', charset='utf8')
cursor = db.cursor()
cursor.execute("SET NAMES utf8")
cursor.execute("FLUSH QUERY CACHE")

# ---------------------------------------------
# retrieve information from domain-specific KB
ignore = ['回族自治区', '维吾尔自治区', '壮族自治区', '自治区', '新区', '省', '市',
          '区', '县', '街道', '居委会', '村', '乡', '自治州', '镇']

infos = ['province', 'city', 'area']  # , 'street'

export_infos = ['province', 'province_confidence',
                'city', 'city_confidence',
                'area', 'area_confidence', ]

candidates = {}
entities = {}
for info in infos:
    cursor.execute("select distinct %s from geokg;" % info)
    entities[info] = [item[0] for item in cursor.fetchall()]
    candidates[info] = copy.deepcopy(entities[info])

    for item in ignore:
        for i in range(len(entities[info])):
            candidates[info][i] = candidates[info][i].replace(unicode(item), '')

# ---------------------------------------------

# load raw data
infile = open('./data/data.utf8', 'r')
raw = infile.readlines()

infile.close()
raw_data = []

for line in raw:
    line = line[:-2].replace('"', '')
    line = line.split(',')
    raw_data.append(line)

del raw

# run general statistics for raw data
doc_stats(raw_data)

# train CRF model using domain KG
file_list = []     # todo add retrain files into the list
pre_train(cursor, file_list)


# NER via hybrid model
al_file = open('./outputs/al.tsv', 'w')
itg_file = open('./outputs/al_itg.tsv', 'w')
eva_file = open('./outputs/evaluation.tsv', 'w')

count = 0
al_threshold = 0.2

random.seed(1000)
random.shuffle(raw_data)

for line in raw_data:
    if len(unicode(line[1])) <= 7:
        continue

    if ' ' in line[1]:
        line[1] = line[1].replace(' ', '')
    anns = list()

    # annotate the data with different models
    anns.append(entity_clf(coarse_parser.parser(line[1]), candidates, entities))
    anns.append(entity_clf(crf_tagger.tagger(line[1]), candidates, entities))

    # merge the result of multi-models using a voting/stacking method
    # ann = integrator.voting(anns, infos)  # voting method
    ann = integrator.stacking(line[1], anns, infos)     # stacking method
    ann = integrator.pos_check(ann, infos)

    # export evaluation data
    if random.randint(1, 250) == 125:
        export(ann, anns, export_infos, line[1], line[0], eva_file, typ='eva')

    # active learning
    else:
        for item in infos:
            if ann["%s_confidence" % item] < al_threshold:
                export(ann, anns, infos, line[1], line[0], al_file, typ='al')
                export(ann, anns, infos, line[1], line[0], itg_file, typ='itg')

    count += 1

print "process done"

eva_file.close()
al_file.close()
cursor.close()
db.close()
