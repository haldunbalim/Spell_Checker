import numpy as np
import pickle
from pyxdameraulevenshtein import damerau_levenshtein_distance

DEPENDENCY_FOLDER_PATH='dependencies/'
FREQUENCY_FILE='full.txt'
ATTENTION_WORDS_FILE='attention_words.txt'
MAXIMUM_VALUE=9999999
TYPO_DICTIONARY_SYMSPELL_FILE = '/home/vircon/Desktop/trial.pkl'

"""
LOAD_PICKLE = False

if LOAD_PICKLE:
    with open(TYPO_DICTIONARY_SYMSPELL_FILE, 'rb') as f:
        typo_dict = pickle.load(f)
"""
attention_words=[]
with open(DEPENDENCY_FOLDER_PATH+ATTENTION_WORDS_FILE, 'r', encoding="utf-8") as file:
    for line in file.readlines():
        attention_words.append(line[:-1])
attention_words=set(attention_words)


threshold_levensthein=2

def generate_deletes(string, max_distance):
    deletes = []
    queue = [string]
    for _ in range(max_distance):
        temp_queue = []
        for word in queue:
            if len(word) > 1:
                for c in range(len(word)):  # character index
                    word_minus_c = word[:c] + word[c + 1:]
                    if word_minus_c not in deletes:
                        deletes.append(word_minus_c)
                    if word_minus_c not in temp_queue:
                        temp_queue.append(word_minus_c)
        queue = temp_queue

    return deletes

def build(min=15,save=False):
    typo_dictionary={}
    with open(DEPENDENCY_FOLDER_PATH+FREQUENCY_FILE, 'r') as file:
        lines = file.readlines()
        for line in lines:
            w, f = line.split()
            f = int(f)
            if w in attention_words:
                f = MAXIMUM_VALUE
            if f > min:
                if w in typo_dictionary:
                    typo_dictionary[w] = (typo_dictionary[w][0], f)
                else:
                    typo_dictionary[w] = ([], f)

                deletes = generate_deletes(w, threshold_levensthein)
                for d in deletes:
                    if d in typo_dictionary:
                        typo_dictionary[d][0].append(w)
                    else:
                        typo_dictionary[d] = ([w], 0)

    if save:
        with open(TYPO_DICTIONARY_SYMSPELL_FILE,'wb') as f:
            pickle.dump(typo_dictionary,f,protocol=pickle.HIGHEST_PROTOCOL)

    return typo_dictionary

def correct(string,typo_dictionary):
    corrections_dict = {}
    min_correct_len = float('inf')
    queue = sorted(list(set([string] + generate_deletes(string, threshold_levensthein))), key=len, reverse=True)

    while len(queue) > 0:
        q_item = queue.pop(0)

        if ((len(corrections_dict) > 0) and ((len(string) - len(q_item)) > min_correct_len)):
            break
        if (q_item in typo_dictionary) and (q_item not in corrections_dict):
            if (typo_dictionary[q_item][1] > 0):
                corrections_dict[q_item] = (typo_dictionary[q_item][1], len(string) - len(q_item))
                if len(string) == len(q_item):
                    break
                elif (len(string) - len(q_item)) < min_correct_len:
                    min_correct_len = len(string) - len(q_item)

            for sc_item in typo_dictionary[q_item][0]:
                if (sc_item not in corrections_dict):
                    if len(q_item) == len(string):
                        item_dist = len(sc_item) - len(q_item)

                    item_dist = damerau_levenshtein_distance(sc_item, string)

                    if item_dist > min_correct_len:
                        pass
                    elif item_dist <= threshold_levensthein:
                        corrections_dict[sc_item] = (typo_dictionary[sc_item][1], item_dist)
                        if item_dist < min_correct_len:
                            min_correct_len = item_dist

                    corrections_dict = {k: v for k, v in corrections_dict.items() if v[1] <= min_correct_len}

    return corrections_dict


def best(string,typo_dictionary):
    try:
        as_list = correct(string,typo_dictionary).items()
        outlist = sorted(as_list, key=lambda item: (item[1][1], -item[1][0]))
        return outlist[0][0]
    except:
        return string
