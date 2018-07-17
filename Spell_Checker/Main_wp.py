from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import time
import itertools
import pickle
from operator import itemgetter
from pyxdameraulevenshtein import damerau_levenshtein_distance



alphabet="q w e r t y u ı o p ğ ü a s d f g h j k l ş i z x c v b n m ö ç"
#key_code={'q': 0,'w': 1,'e': 2,'r': 3,'t': 4,'y': 5,'u': 6,'ı': 7,
#'o': 8,'p': 9,'ğ': 10,'ü': 11,'a': 12,'s': 13,'d': 14,'f': 15,'g': 16,'h': 17,'j': 18,'k': 19,'l': 20,'ş': 21,'i': 22,'z': 23,
#'x': 24,'c': 25,'v': 26,'b': 27,'n': 28,'m': 29,'ö': 30,'ç': 31}
#turkish_chars="ğ ü ö ı ş ç Ğ Ü Ö I Ş Ç"
consonants='bcçdfgğhjklmnprsştvyz'
ascii_map={'c': 'ç','o': 'ö', 'u': 'ü','g': 'ğ','i': 'ı','s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}
latin_map={'s': 'ş','ç': 'c','ö': 'o', 'ü': 'u','ğ': 'g','ı': 'i','ş': 's'}
SIMILARITY_MAP_FILE_PATH='/home/vircon/Desktop/correctedw2v.pkl'

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

with open(SIMILARITY_MAP_FILE_PATH,'rb') as f:
    similarity_map=pickle.load(f)

# here we load dictionary with frequencies
file = open("full.txt", 'r', encoding="utf-8")
dict_with_frequencies = {}

for line in file.readlines():
    word = line.split()[0]
    frequency = int(line.split()[1])
    if frequency > 15:
        dict_with_frequencies[word] = frequency
file.close()


file = open('buzzwords.txt')
buzzwords = []
for line in file.readlines():
    word = line.split()[0]
    buzzwords.append(word)

file = open('manual.txt')
manual = {}
for line in file.readlines():
    manual[line.split()[0]] = line.split()[1]


def latinizer(word, check, reverse=False):
    if (check):
        return ''.join(list(map(lambda x: latin_map[x] if x in latin_map else x, list(word))))
    else:
        return word


def last_check(word):
    corr = correction(word)
    if corr != word:
        return corr

    word = deacify_wrt_sound(word)

    qs = question_suffix(word, True)
    if qs:
        return spell_check_word(qs.split()[0]) + ' ' + qs.split()[1]

    sep = seperator(word, True)
    if (sep != word):
        #print(sep,word)
        return spell_check_word(sep.split()[0]) + ' ' + spell_check_word(sep.split()[1])

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


def question_suffix(word, force=False):
    if (len(word) > 5 and word[-5:-3] in ['mı', 'mi', 'mu', 'mü']):
        if (force or isCorrect(word[:-5])):
            return word[:-5] + ' ' + word[-5:]
    if word[-2:] in ['mı', 'mi', 'mu', 'mü']:
        if (force or isCorrect(word[:-2])):
            return word[:-2] + ' ' + word[-2:]
    if word[-7:] in ['mısınız', 'misiniz', 'musunuz', 'müsünüz']:
        return word[:-7] + ' ' + word[-7:]
    return None


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


def seperator(word, force=False):
    for i in range(1,len(word)-1):
        left = word[:i]
        right = word[i:]
        if isCorrect(left) and isCorrect(right):
            return left + " " + right
        elif force:
            if isCorrect(left) or isCorrect(right):
                return left + " " + right
    return word


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

def remove_redundant(word,redundant):
    s=""
    for char in word:
        if char not in redundant:
            s+=char
    return s


def spell_check_word(word,similiarity_threshold=0.85,
                     latin=False,similarity_min=0.6,
                     use_manual=True,use_deasciifier=True):

    word = my_lower(word)

    redundant = (".?*!,;:123456789")
    word = remove_redundant(word, redundant)
    if (word == ''):
        return ''
    if (len(word) == 1):
        return word
    if use_deasciifier:
        word = deascify(word)

    if use_manual and word in manual:
        return manual[word]

    # check if it is a buzzword
    buzz = is_buzzword(word)
    if (buzz != False):
        return buzz

    # check if correct
    if (isCorrect(word, check_buzzwords=False)):
        return latinizer(word, latin)

    # checks to seperate question word
    qs = question_suffix(word, force=False)
    if qs:
        return latinizer(qs,latin)

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


def sentence_spell_checker(sentence):
    sentence_corrected=""
    for word in sentence.split():
        spell_checked=spell_check_word(word)
        if(spell_checked != ''):
            sentence_corrected+= spell_checked+" "
    return sentence_corrected[:-1]


def validation(fileIn,num_samples,m1,m2,name_1,name_2):
    df=pd.read_excel(fileIn)
    df=df['MESSAGE'][:num_samples]
    new_frame=pd.DataFrame(columns=['Original',name_1,name_2])
    new_frame['Original']=df
    new_frame[name_1]=df
    new_frame[name_2]=df


    tick=time.time()
    new_frame[name_1]=new_frame[name_1].apply(lambda x: m1(x))
    tock=time.time()
    print('It took {} seconds for m1 to convert {} sentences'.format(tock-tick,num_samples))


    tick=time.time()
    new_frame[name_2]=new_frame[name_2].apply(lambda x: m2(x))
    tock = time.time()
    print('It took {} seconds for m2 to convert {} sentences'.format(tock - tick, num_samples))


    new_frame=new_frame[new_frame[name_1]!=new_frame[name_2]]
    new_frame.to_excel('/home/vircon/Desktop/Comparison.xlsx')
    print('There are {} differences'.format(len(new_frame)))
    return new_frame

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
    new_frame['Corrected'] = new_frame['Corrected'].apply(lambda x: sentence_spell_checker(x))
    tock = time.time()
    print('It took {} seconds to convert {} sentences'.format(tock - tick, samples_done))

    new_frame.to_excel('/home/vircon/Desktop/Check.xlsx')
    return new_frame


if __name__ == '__main__':
    convert('/home/vircon/Desktop/ing bank.xls',do_all=True)
    #validation('/home/vircon/Desktop/ing bank.xls',500,sentence_spell_checker,sentence_spell_checker_1,name_1='100_1000',name_2='100_50')
    #print(spell_check_word('zman'))