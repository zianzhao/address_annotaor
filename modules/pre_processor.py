#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from gensim.models import Word2Vec

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LsiModel

# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


def preprocessor(raw, cursor):
    count = 0
    processed = []
    x = []
    for item in raw:
        line = item[1]

        tmp = u' '.join(unicode(line))

        try:
            tmp.encode()
            processed.append([item[0], item[1], tmp])
            x.append(tmp)
        except:
            count += 1
            pass

    # corpus in domain data set
    cursor.execute("select * from geokg;")
    while True:
        data = cursor.fetchone()
        if data is None:
            break
        tmp = ''.join(data)
        tmp = u' '.join(unicode(tmp))
        x.append(tmp)

    model = Word2Vec(x, size=10, window=5, min_count=5, workers=4)
    model.save('./modules/models/vectorizer.model')

    # bigram_transformer = gensim.models.Phrases(sentences)
    # model = Word2Vec(bigram_transformer[sentences], size=100, ...)

    return processed


"""
    doc_stats
    Purpose:
        train docment2vector model with raw data
    Parameters:
        [in] list raw - list of raw addresses
    Returns:
        None
    Author:
        Zian zhao
        5.11.2018
"""


def doc_stats(raw):
    sentences = []

    # split raw addresses into words
    for item in raw:
        line = item[1]
        tmp = u' '.join(unicode(line))
        sentences.append(tmp.split())

    dct = Dictionary(sentences)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in sentences]
    model = TfidfModel(corpus)  # tf-idf model
    corpus_tfidf = model[corpus]    # convert corpus into tf-idf matrix
    lsi = LsiModel(corpus_tfidf, num_topics=10)     # lcompress the matrix using lsi model

    # store the models
    model.save('./modules/models/tfidf.model')
    dct.save('./modules/models/dic.model')
    lsi.save('./modules/models/lsi.model')

    print 'Doc2Vec model training finished'
    return




