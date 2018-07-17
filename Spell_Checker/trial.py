import pickle
from gensim.models import KeyedVectors
import time
from pyxdameraulevenshtein import damerau_levenshtein_distance
import numpy as np

WORD2VEC_FILE_PATH='/home/vircon/Desktop/word2vec/3_word2vec_100'

word_vectors = KeyedVectors.load(WORD2VEC_FILE_PATH)


file = open("full.txt", 'r', encoding="utf-8")
dict_with_frequencies = {}

for line in file.readlines():
    word = line.split()[0]
    frequency = int(line.split()[1])
    if frequency > 15:
        dict_with_frequencies[word] = frequency
file.close()

def isCorrect(word):
    return  word in dict_with_frequencies

def has_two_swaps(s1, s2, dld):
    d1 = {}
    d2 = {}
    alp_set = set()
    changes = 0
    for i in (range(max(len(s1), len(s2)))):
        if len(s1) > i:
            if s1[i] not in d1:
                d1[s1[i]] = 0
            d1[s1[i]] += 1
            alp_set.add(s1[i])

        if len(s2) > i:
            if s2[i] not in d2:
                d2[s2[i]] = 0
            d2[s2[i]] += 1
            alp_set.add(s2[i])

    for char in alp_set:
        if char not in d1:
            if char in d2:
                changes += 1
        elif char not in d2:
            changes += 1
        else:
            if d1[char] != d2[char]:
                changes += 1

    if changes == 0:
        if dld == 2:
            return True
    return changes / 2 == 2


def save_obj(obj, name ):
    with open('/home/vircon/Desktop/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def dum(x):
    i=0
    my_dict={}
    tick=time.time()
    for word in word_vectors.vocab:
        ls=[]
        i+=1
        for similar_word,similarity in word_vectors.most_similar(word,topn=20):
            dist = damerau_levenshtein_distance(word, similar_word)
            if dist <= 2:
                swaps = has_two_swaps(word, similar_word, dist)
            else:
                swaps = True

            if similarity>0.8 or (not swaps and dist <= 2 and similarity > 0.6 and isCorrect(similar_word)):
                ls.append((similar_word,similarity))
        my_dict[word]=ls

        if i==1000:
            print('{}/{}'.format(i,len(word_vectors.vocab)))

    print(my_dict)
    save_obj(my_dict,'w2v')

if __name__ == '__main__':
    with open('/home/vircon/Desktop/correctedw2v.pkl','rb') as f:
        my_dict=pickle.load(f)

    print(my_dict['kolaygelsin'])

