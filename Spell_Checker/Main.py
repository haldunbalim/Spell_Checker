import numpy as np
import pandas as pd
import time, itertools, pickle
from operator import itemgetter
from Symspell import best,build

alphabet="q w e r t y u ı o p ğ ü a s d f g h j k l ş i z x c v b n m ö ç"
consonants='bcçdfgğhjklmnprsştvyz'
ascii_map={'c': 'ç','o': 'ö', 'u': 'ü','g': 'ğ','i': 'ı','s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}
latin_map={'s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}

DEPENDENCY_FOLDER_PATH='dependencies/'
FREQUENCY_FILE='full.txt'
MANUAL_FILE='manual.txt'
BUZZWORDS_FILE='buzzwords.txt'
SIMILARITY_MAP_FILE='similarity_typos_map.pkl'
ATTENTION_WORDS_FILE='attention_words.txt'
USE_PICKLE_FOR_SYMSPELL=True
SAVE_PICKLE_SYMSPELL=False
TYPO_DICTIONARY_SYMSPELL_FILE='/home/vircon/Desktop/trial.pkl'
MAXIMUM_VALUE=9999999


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


with open(DEPENDENCY_FOLDER_PATH+SIMILARITY_MAP_FILE,'rb') as f:
    similarity_map=pickle.load(f)

if USE_PICKLE_FOR_SYMSPELL:
    with open(TYPO_DICTIONARY_SYMSPELL_FILE, 'rb') as f:
        tick = time.time()
        typo_dict = pickle.load(f)
        tock = time.time()
        print('It took {} seconds to load the typo dictionary for symspell'.format(tock - tick))
else:
    tick=time.time()
    typo_dict=build(save=SAVE_PICKLE_SYMSPELL)
    tock = time.time()
    print('It took {} seconds to prepare the typo dictionary for symspell'.format(tock-tick))


# here we load dictionary with frequencies
dict_with_frequencies = {}
with open(DEPENDENCY_FOLDER_PATH+FREQUENCY_FILE, 'r', encoding="utf-8") as file:
    for line in file.readlines():
        word,frequency = line.split()
        frequency=int(frequency)
        if frequency > 15:
            dict_with_frequencies[word] = frequency


with open(DEPENDENCY_FOLDER_PATH+ATTENTION_WORDS_FILE, 'r', encoding="utf-8") as file:
    for line in file.readlines():
        dict_with_frequencies[line[:-1]]=MAXIMUM_VALUE
        MAXIMUM_VALUE-=1


file = open(DEPENDENCY_FOLDER_PATH+BUZZWORDS_FILE)
buzzwords = []
for line in file.readlines():
    word = line.split()[0]
    buzzwords.append(word)

file = open(DEPENDENCY_FOLDER_PATH+MANUAL_FILE)
manual = {}
for line in file.readlines():
    manual[line.split()[0]] = line[line.find(' ')+1:]


def latinizer(word, check):
    if (check):
        return ''.join(list(map(lambda x: latin_map[x] if x in latin_map else x, list(word))))
    else:
        return word


def is_buzzword(word, use_boyer_moore=False):
    latin = latinizer(word, True)
    if use_boyer_moore:
        for buzzword in buzzwords:
            if boyer_moore(buzzword, word):
                return word
            elif boyer_moore(buzzword, word):
                return latin
        return False
    else:
        for buzzword in buzzwords:
            if buzzword in word:
                return word
            elif buzzword in latin:
                return latin
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


def deacify_wrt_sound(word):
    thick_counter = 0
    slim_counter = 0
    slim_map = {'ı': 'i', 'o': 'ö', 'u': 'ü'}
    thick_map = {'i': 'ı', 'ö': 'o', 'ü': 'u'}
    for char in word:
        if char == 'a':
            thick_counter += 1
        elif char == 'e':
            slim_counter += 1

    if slim_counter >= thick_counter:
        return ''.join(list(map(lambda x: slim_map[x] if x in slim_map else x, list(word))))
    elif thick_counter > slim_counter:
        return ''.join(list(map(lambda x: thick_map[x] if x in thick_map else x, list(word))))


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


def question_suffix(word, force=False):
    quest_suffixes = ['muyum', 'müyüm', 'miyim', 'mıyım', 'musun', 'müsün', 'misin', 'mısın', 'mudur', 'müdür', 'midir',
                      'mıdır', 'muyuz', 'müyüz', 'miyiz', 'mıyız', 'mular', 'müler', 'miler', 'mılar']

    if (len(word) > 5 and word[-5:] in quest_suffixes):
        if (force or isCorrect(word[:-5])):
            return word[:-5] + ' ' + word[-5:]
    if word[-2:] in ['mı', 'mi', 'mu', 'mü']:
        if (force or isCorrect(word[:-2])):
            return word[:-2] + ' ' + word[-2:]
    if word[-7:] in ['mısınız', 'misiniz', 'musunuz', 'müsünüz']:
        return word[:-7] + ' ' + word[-7:]
    return None



def seperator(word):
    ls_both_cand=[]
    ls_single_cand=[]
    last_len=0
    for i in range(2,len(word)-2):
        left = word[:i]
        right = word[i:]
        deascified_left=deascify(left)
        deascified_right=deascify(right)
        if (isCorrect(left) or deascified_left!=left) and (isCorrect(right) or deascified_right!=right):
            ls_both_cand.append(deascified_left + " " + deascified_right)
        elif (isCorrect(left) or deascified_left!=left) or (isCorrect(right) or deascified_right!=right):
            if len(deascified_left)-last_len <=3 or last_len== 0:
                last_len=len(deascified_left)
                ls_single_cand.append((deascified_left , deascified_right))
            else:
                break

    if len(ls_both_cand) !=0:
        return ls_both_cand[-1]
    if len(ls_single_cand) !=0:
        l,r=ls_single_cand[-1]
        return spell_check_word(l)+' '+spell_check_word(r)
    return word


def last_check(word, use_exception_handler=True):
    if use_exception_handler:
        corr = best(word,typo_dict)
        if corr != word:
            return corr

    sound_fixed = deacify_wrt_sound(word)
    qs = question_suffix(sound_fixed, True)
    if qs:
        return spell_check_word(qs.split()[0]) + ' ' + qs.split()[1]

    sep = seperator(word)
    if (sep != word):
        return sep

    return word


def remove_redundant(word,redundant):
    s=""
    for char in word:
        if char not in redundant:
            s+=char
    return s


def spell_check_word(word,similiarity_threshold=0.85,latin=False,similarity_min=0.6,
                     use_manual=True,use_deasciifier=True,firstTime=True):
    if firstTime:
        word = my_lower(word)
        redundant = ("0123456789")
        word = remove_redundant(word, redundant)
        if (word == ''):
            return ''
        if (len(word) == 1):
            return word
        if use_deasciifier:
            word = deascify(word)

    if use_manual and word in manual:
        return manual[word][:-1]

    # check if correct
    if (isCorrect(word, check_buzzwords=False)):
        return latinizer(word, latin)


    # check if it is a buzzword
    buzz = is_buzzword(word)
    if (buzz != False):
        return buzz

    # checks to seperate question word
    qs = question_suffix(word, force=False)
    if qs:
        root,suffix=qs.split()
        return latinizer(spell_check_word(root,firstTime=False),latin)+' '+latinizer(suffix,latin)

    closest = (word, 0)

    try:
        similar_word_list = similarity_map[word]

        if len(similar_word_list)==0:
            return latinizer(last_check(word), latin)

        if similar_word_list[0][1] >= similiarity_threshold and hasSameChars(similar_word_list[0][0],word) and len(word)<=4:
            return latinizer(similar_word_list[0][0],latin)

        for similar_word,similarity in similar_word_list:
            if dict_with_frequencies[similar_word] > closest[1] and similarity >= similarity_min:
                closest=(similar_word,dict_with_frequencies[similar_word])

        if (closest[1] == 0):
            # no similiar word found that fits the conditions in num_total words
           return latinizer(last_check(word), latin)
        else:
            return latinizer(closest[0],latin)

    except KeyError:
        # a typo that has not been seen before
        return latinizer(last_check(word), latin)

def my_split(sentence,redundant='.,?:;!'):
    for char in redundant:
        sentence=sentence.replace(char,' ')
    return sentence.split()


def fix(sentence):
    words=sentence.split()
    fixed_sentence = [words[0]]
    cursor=0
    for i in range(1,len(words)):
        if words[i] not in ['ne','de','da','ki'] and isCorrect(fixed_sentence[cursor]+words[i],check_buzzwords=False):
            candidate = fixed_sentence[cursor]+words[i]
            qs = question_suffix(candidate)
            if qs:
                cursor+=1
                fixed_sentence.append(words[i])
            else:
                fixed_sentence[cursor] = candidate
        else:
            cursor += 1
            fixed_sentence.append(words[i])

    return ' '.join(fixed_sentence)

def sentence_spell_checker(sentence,fixer=True):
    sentence_corrected=""
    for word in my_split(sentence):
        spell_checked=spell_check_word(word)
        if(spell_checked != ''):
            sentence_corrected+= spell_checked+" "

    if fixer and len(sentence_corrected.split()) >=2:
        return fix(sentence_corrected[:-1])
    else:
        return sentence_corrected[:-1]


def convert(fileIn,num_samples=500,do_all=False):
    df = pd.read_excel(fileIn)
    if do_all:
        df = df['MESSAGE']
    else:
        df = df['MESSAGE'][:num_samples]
    new_frame = pd.DataFrame(columns=['Original', 'Corrected'])
    new_frame['Original'] = df
    new_frame['Corrected'] = df

    samples_done=len(new_frame)

    tick = time.time()
    new_frame['Corrected'] = new_frame['Corrected'].apply(lambda x: sentence_spell_checker(x,fixer=True))
    tock = time.time()
    print('It took {} seconds to convert {} sentences'.format(tock - tick, samples_done))

    new_frame.to_excel('/home/vircon/Desktop/Spell_Checker_Output.xlsx')
    return new_frame


if __name__ == '__main__':
    convert('/home/vircon/Desktop/ing bank.xls',do_all=True)


