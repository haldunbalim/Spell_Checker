from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from collections import Counter
import itertools
from operator import itemgetter
import time
from pyxdameraulevenshtein import damerau_levenshtein_distance

alphabet="q w e r t y u ı o p ğ ü a s d f g h j k l ş i z x c v b n m ö ç"
consonants='bcçdfgğhjklmnprsştvyz'
ascii_map={'c': 'ç','o': 'ö', 'u': 'ü','g': 'ğ','i': 'ı','s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}
latin_map={'s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}


FILE_WORDS_WITH_FREQUENCIES="full.txt"
FILE_WORD2VEC="new_word2vec"
FILE_BUZZWORDS='buzzwords.txt'

# here we load dictionary with frequencies

file=open(FILE_WORDS_WITH_FREQUENCIES,'r',encoding="utf-8")

dict_with_frequencies = {}
for line in file.readlines():
    word=line.split()[0]
    frequency=int(line.split()[1])
    dict_with_frequencies[word]=frequency
file.close()

file=open(FILE_BUZZWORDS,'r')
buzzwords = []
for line in file.readlines():
    word=line.split()[0]
    buzzwords.append(word)


def make_bad_match_table(pattern):

    length = len(pattern)
    table = {}
    for i, c in enumerate(pattern):
        if i == length-1 and not c in table:
            table[c] = length
        else:
            table[c] = length - i - 1

    return table


def boyer_moore(pattern, text):

    match_table = []
    pattern_length = len(pattern)
    text_length = len(text)
    if pattern_length > text_length:
        return match_table

    table = make_bad_match_table(pattern)
    index = pattern_length - 1
    pattern_index = pattern_length - 1

    while index < text_length:
        if pattern[pattern_index] == text[index]:
            if pattern_index == 0:
                match_table.append(index)
                pattern_index = pattern_length - 1
                index += (pattern_length * 2 - 1)
            else:
                pattern_index -= 1
                index -= 1
        else:
            index += table.get(text[index], pattern_length)
            pattern_index = pattern_length - 1

    return len(match_table) !=0

def is_buzzword(word,use_boyer_moore=False):
    latin=latinizer(word,True)
    if use_boyer_moore:
        for buzzword in buzzwords:
            if boyer_moore(buzzword,word):
                return word
            elif boyer_moore(buzzword,word):
                return latin
        return False
    else:
        for buzzword in buzzwords:
            if buzzword in word:
                return word
            elif buzzword in latin:
                return latin
        return False

word_vectors = KeyedVectors.load(FILE_WORD2VEC)

def question_suffix(word):
    if (len(word) > 5 and word[-5:-3] in ['mı', 'mi', 'mu', 'mü']):
        if (isCorrect(word[:-5])):
            return word[:-5] + ' ' + word[-5:]
    if word[-2:] in ['mı', 'mi', 'mu', 'mü']:
        if (isCorrect(word[:2])):
            return word[:-2] + ' ' + word[-2:]
    if word[-7:] in ['mısınız', 'misiniz']:
        return word[:-7] + ' ' + word[-7:]
    return False


def isCorrect(word, check_buzzwords=True):
    if check_buzzwords:
        return word in dict_with_frequencies or word.isdigit() or is_buzzword(word)
    return word in dict_with_frequencies or word.isdigit()


def my_lower(word):
    return word.replace("I", "ı").lower().replace('i̇', "i")


def hasSameChars(s1, s2):
    last = 0
    for char in s1:
        if char not in consonants:
            return False
        last = s2.find(char)
        s2 = s2[last:]
        if (last == -1):
            return False
    return True


def deascify_n_char(word, n, ls_deasc_pos_nrs):
    ls_asc = []
    ls_deasc_pos = list(itertools.combinations(ls_deasc_pos_nrs, n))
    for i in range(len(ls_deasc_pos)):
        temp = list(word)
        for j in range(len(ls_deasc_pos[i])):
            temp[ls_deasc_pos[i][j]] = ascii_map[temp[ls_deasc_pos[i][j]]]
        candidate = ''.join(temp)
        if isCorrect(candidate, False):
            return candidate
        else:
            ls_asc.append(candidate)
    return ls_asc


def deascify(word):
    ls = []
    ls_deasc_pos_nrs = []
    for i in range(len(word)):
        if word[i] in ascii_map:
            ls_deasc_pos_nrs.append(i)
    for i in range(len(ls_deasc_pos_nrs) + 1):
        output = deascify_n_char(word, i, ls_deasc_pos_nrs)
        if type(output) == list:
            ls.extend(output)
        else:
            return output
    return word


def latinizer(word, check):
    if (check):
        return ''.join(list(map(lambda x: latin_map[x] if x in latin_map else x, list(word))))
    else:
        return word


def seperator(word):
    for i in range(len(word)):
        left = word[:i]
        right = word[i:]
        if (isCorrect(left) and isCorrect(right)):
            return left + " " + right
    return word

def has_two_swaps(s1, s2, dld):
    d1 = {}
    d2 = {}
    alp_set = {}
    changes = 0
    for i in (range(max(len(s1), len(s2)))):
        if len(s1) > i:
            if s1[i] not in d1:
                d1[s1[i]] = 0
            d1[s1[i]] += 1
            alp_set.append(s1[i])

        if len(s2) > i:
            if s2[i] not in d2:
                d2[s2[i]] = 0
            d2[s2[i]] += 1
            alp_set.append(s2[i])

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


def correction(word):
    "Most probable spelling correction for word."
    temp = max(candidates(word), key=itemgetter(1))[0]
    return temp if len(temp) != 1 else seperator(word)


def candidates(word):
    "Generate possible spelling corrections for word."
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set((w, dict_with_frequencies[w]) for w in words if w in dict_with_frequencies)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcçdefgğhıijklmnoöprsştuüvyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def spell_check_word(word, num_total=1000, threshold_levensthein=2,
                     similiarity_threshold=0.85, max_rec=4000, firstTime=True, latin=False,
                     vector_space=word_vectors, similarity_min=0.6):
    if (firstTime):
        word = my_lower(word).strip(".?*!,;:")
        if (word == ''):
            return ''
        if (len(word) == 1):
            return word + ' '
        word = deascify(word)

    if (isCorrect(word)):
        return latinizer(word, latin)

    qs = question_suffix(word)
    if (qs != False):
        return qs

    list_close_words = []

    try:
        similiar_words = vector_space.most_similar(word, topn=num_total)

        for i in range(len(similiar_words)):
            similiar_word = similiar_words[i][0]
            similarity = similiar_words[i][1]
            dist = damerau_levenshtein_distance(word, similiar_word)

            if dist <= 2:
                swaps = has_two_swaps(word, similiar_word, dist)
            else:
                swaps = True

            if (similarity > similiarity_threshold and
                    isCorrect(similiar_word) and hasSameChars(word, similiar_word) and len(word) <= 4
                    and len(word) > 2):
                return similiar_word
            elif (not isCorrect(similiar_word) and np.abs(len(similiar_word) - len(word)) > threshold_levensthein):
                continue
            elif ((len(word) >= 3 and len(similiar_word) >= 3) and
                  word[0] != similiar_word[0] and word[1] != similiar_word[1] and word[2] != similiar_word[2]):
                continue
            elif (not swaps and dist <= threshold_levensthein and isCorrect(similiar_word)
                  and similarity > similarity_min):
                list_close_words.append((similiar_word, dict_with_frequencies[similiar_word]))

        if (len(list_close_words) == 0):
            # no similiar word found that fits the conditions in num_total words
            if (num_total <= max_rec):
                return spell_check_word(word, num_total * 2, threshold_levensthein=2, firstTime=False)
            else:
                # a typo that is not well fitted in the vector space
                return latinizer(correction(word), latin)

        return max(list_close_words, key=itemgetter(1))[0]

        # if you want to see the list comment the top and uncomment the bot
        # list_close_words.sort(key=lambda x: x[1], reverse=True)
        # return list_close_words[0][0]

    except KeyError:
        # a typo that has not been seen before
        return latinizer(correction(word), latin)



def sentence_spell_checker(sentence):
    sentence_corrected=""
    for word in sentence.split():
        spell_checked=spell_check_word(word,vector_space=word_vectors)
        if(spell_checked != ''):
            sentence_corrected+= spell_checked+" "
    return sentence_corrected[:-1]



if __name__  == '__main__':
    print(sentence_spell_checker('kelme'))

