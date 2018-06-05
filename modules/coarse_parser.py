#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

# Set the default code
stdout = sys.stdout
reload(sys)
sys.stdout = stdout
sys.setdefaultencoding('utf-8')


"""
    parser
    Purpose:
        NER via RE
    Parameters:
        [in] string text - raw test
    Returns:
        dict address - extracted information
    Author:
        Zian zhao
        12.31.2017
"""

"""
    parser
    Update:
        delete raw test before the extracted info.
        use the first matching item as the extracted info.
    Author:
        Zian zhao
        1.02.2018
"""


def parser(text):
    raw = unicode(text)
    text = unicode(text)

    address = dict()

    # province extraction
    pat = u"[^省市区0-9]{2,3}?[省]|[^省市区]{2,5}?自治区|上海市|天津市|重庆市|北京市"
    mch = list(set(re.findall(pat, text)))
    address['province'] = ''
    address['province_p'] = []
    if len(mch):
        tmp = ""
        pos = 1000
        for item in mch:
            if len(item) > 0:
                if pos > text.index(item):
                    pos = text.index(item)
                    tmp = item
        # tmp = tmp.replace(' ', '')
        address['province'] = tmp

        if len(tmp):
            low = raw.index(tmp)
            address['province_p'] = [low, low+len(unicode(tmp)) - 1]

        text = text[text.index(tmp):]
        text = text.replace(tmp, '')

    # city extraction
    pat = u"([^省市区0-9]+?市|[^省市区]+?自治州){0,1}"
    mch = list(set(re.findall(pat, text)))
    address['city'] = ''
    address['city_p'] = []
    if len(mch):
        tmp = ""
        pos = 1000
        for item in mch:
            if len(item) > 0:
                if pos > text.index(item):
                    pos = text.index(item)
                    tmp = item
        # tmp = tmp.replace(' ', '')
        address['city'] = tmp

        if len(tmp):
            low = raw.index(tmp)
            address['city_p'] = [low, low + len(unicode(tmp)) - 1]

        text = text[text.index(tmp):]
        text = text.replace(tmp, '')

    # area extraction
    pat = u"([^0-9省市区县镇小]+?[区市县]){0,1}"
    mch = list(set(re.findall(pat, text)))
    address['area'] = ''
    address['area_p'] = []
    if len(mch):
        tmp = ""
        pos = 1000
        for item in mch:
            if len(item) > 0:
                if pos > text.index(item):
                    pos = text.index(item)
                    tmp = item
        # tmp = tmp.replace(' ', '')
        address['area'] = tmp
        if len(tmp):
            low = raw.index(tmp)
            address['area_p'] = [low, low + len(unicode(tmp)) - 1]

    return address


