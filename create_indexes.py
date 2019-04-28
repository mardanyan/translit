# -*- coding: utf-8 -*


import argparse
import utils
#import random
# import numpy as np
# import codecs
import json
# from datetime import datetime
import glob
import re

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
    Make vocabulary specifying using languages

    Usage:
        python create_indexes.py --languages=hy,ru,en --data_size=50_000_000
        python create_indexes.py --languages=hy --data_size=50_000_000
        python create_indexes.py --languages=hy-en --data_size=50_000_000
        python create_indexes.py --languages=hy-en,hy-ru --data_size=50_000_000

    ''')
    parser.add_argument('--languages', default=None, required=True)
    parser.add_argument('--data_size', default=50_000_000, type=int)
    args = parser.parse_args()

    languages = utils.parse_languages(args.languages)

    chars = set()
    trans_chars = set()

    for key, value in languages.items():
        if len(value) == 0:
            dirs = glob.glob('data_preprocessed/' + key + '/mapping_to_*')
            for dir in dirs:
                p = re.compile("mapping_to_(.*)")
                result = p.search(dir)
                lang = result.group(1)
                chars = chars.union(get_processed_text('data_preprocessed/' + key + "/mapping_to_" + lang + "/*.txt"))
                trans_chars = trans_chars.union(get_processed_text('data_preprocessed/' + key + "/mapping_to_" + lang + "/*.txt_translate"))
        else:
            for lang in value:
                chars = chars.union(get_processed_text('data_preprocessed/' + key + "/mapping_to_" + lang + "/*.txt"))
                trans_chars = trans_chars.union(get_processed_text('data_preprocessed/' + key + "/mapping_to_" + lang + "/*.txt_translate"))

    print("Chars length: ", len(chars))
    print("Chars: ", chars)
    print("Translated chars length: ", len(trans_chars))
    print("Translated chars: ", trans_chars)

    chars = list(chars)
    char_to_index = {chars[i]: i for i in range(len(chars))}
    index_to_char = {i: chars[i] for i in range(len(chars))}

    # TODO order languages for file name
    langs = args.languages

    print()

    open('data_preprocessed/' + langs + '_char_to_index.json', 'w').write(json.dumps(char_to_index))
    open('data_preprocessed/' + langs + '_index_to_char.json', 'w').write(json.dumps(index_to_char))

    trans_chars = list(trans_chars)
    trans_to_index = {trans_chars[i]: i for i in range(len(trans_chars))}
    index_to_trans = {i: trans_chars[i] for i in range(len(trans_chars))}
    # trans_vocab_size = len(trans_chars)

    open('data_preprocessed/' + langs + '_trans_to_index.json', 'w').write(json.dumps(trans_to_index))
    open('data_preprocessed/' + langs + '_index_to_trans.json', 'w').write(json.dumps(index_to_trans))


def get_processed_text(path):
    chars = set()
    files = glob.glob(path)
    for file in files:
        text = open(file, encoding='utf-8').read()
        chars = chars.union(set(text))
    return chars




# def make_vocabulary_files(data, language, transes):
#
#     ### Makes jsons for future mapping of letters to indices and vice versa
#
#     begin_point = 0
#     done = False
#     read_size = 100_000
#     chars = set()
#     trans_chars = set()
#     data = ' \t' + u'\u2001'  + data # to get these symbols in vocab
#     valids = utils.get_valid_chars(transes)
#     while not done:
#         end_poing = min(begin_point + read_size, len(data))
#         raw_native = data[begin_point:end_poing]
#         print(end_poing / len(data), "% split to words.", end_poing, len(data), end='\r')
#         if end_poing != len(data):
#             begin_point = end_poing
#             raw_native = ' ' + raw_native + ' '
#         else:
#             raw_native = ' ' + raw_native + ' '
#             done = True
#         native = []
#         translit = []
#         for ind in range(1, len(raw_native)-1):
#             trans_char = toTranslit(raw_native[ind-1], raw_native[ind], raw_native[ind+1], transes)
#             translit.append(trans_char[0])
#             native.append(raw_native[ind])
#             if len(trans_char) > 1:
#                 native.append(u'\u2000')
#                 translit.append(trans_char[1])
#
#         translit = utils.validate(valids, translit)[0]
#         for i in range(len(native)):
#             if translit[i] == utils.UNKNOWN_CHAR:
#                 native[i] = utils.UNKNOWN_CHAR
#         chars = chars.union(set(native))
#         trans_chars = trans_chars.union(set(translit))
#         print(str(100.0*begin_point/len(data)) + "% done       ", end='\r')
#
#     chars = list(chars)
#     char_to_index = {chars[i]: i for i in range(len(chars))}
#     index_to_char = {i: chars[i] for i in range(len(chars))}
#
#     language.sort()
#
#     l = '_'.join(language)
#
#     print()
#
#     open('data_preprocessed/' + l + '_char_to_index.json', 'w').write(json.dumps(char_to_index))
#     open('data_preprocessed/' + l + '_index_to_char.json', 'w').write(json.dumps(index_to_char))
#
#     trans_chars = list(trans_chars)
#     trans_to_index = { trans_chars[i] : i for i in range(len(trans_chars)) }
#     index_to_trans = { i : trans_chars[i] for i in range(len(trans_chars)) }
#     # trans_vocab_size = len(trans_chars)
#
#     open('data_preprocessed/' + l + '_trans_to_index.json', 'w').write(json.dumps(trans_to_index))
#     open('data_preprocessed/' + l + '_index_to_trans.json', 'w').write(json.dumps(index_to_trans))
#


if __name__ == '__main__':
    main()
