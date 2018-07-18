import pickle
from pyxdameraulevenshtein import damerau_levenshtein_distance
import re
import pandas as pd
from Symspell import build

if __name__=='__main__':
    ls={}
    i=0
    with open('dependencies/full.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            w,f =line.split()
            f=int(f)
            ls[w]=f

    print(ls['debi'])
    print(ls['kredi'])