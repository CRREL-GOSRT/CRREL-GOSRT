# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:58:56 2021

@author: RDCRLJTP
"""

import os

# search for Snow directory tool
def directory_find(root, word):
    for path, dirs, files in os.walk(root):
        if word in dirs:
            return os.path.join(path, word)