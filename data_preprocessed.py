"""
Preprocess of that generates following data for training
"""

import argparse
import json
import glob
import utils
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
    Train you model specifying parameters
    
    Usage:
        python data_preprocessed.py --language=hy
    
    ''')
    parser.add_argument('--language', default="hy", help='Specify language to train.')
    args = parser.parse_args()

    mappings = glob.glob('mappings/' + args.language + '/mapping_to_*.json')

    data_file_list = glob.glob('data/' + args.language + '/*.txt')

    for m in mappings:
        process_mapping_to(m, args.language, data_file_list)


def process_mapping_to(mapping_to, language, data_file_list):
    print("process_mapping_to:", mapping_to)

    to_language = mapping_to.split("_")[-1].split(".")[0]

    mapping = json.loads(open(mapping_to, 'r', encoding='utf-8').read())
    long_letters = json.loads(open('mappings/' + language + '/long_letters.json', 'r', encoding='utf-8').read())
    long_letter_mapping = {long_letters[i]: chr(ord(u'\u2002') + i) for i in range(len(long_letters))}  # move to utils

    print("Long letters:", long_letters)
    print("Long letter mapping:" , long_letter_mapping)
    tmp_mapping = mapping.copy()
    for c in tmp_mapping:
        if c in long_letters:
            mapping[long_letter_mapping[c]] = mapping[c]
            del mapping[c]
    del tmp_mapping

    print("Mapping:", mapping)

    # replace long letters
    for file_path in data_file_list:
        print("***Generating data:", file_path)
        data = open(file_path, encoding='utf-8').read()
        for letter in long_letter_mapping:
            data = data.replace(letter, long_letter_mapping[letter])
        samples = create_samples(data, mapping)
        base_name = os.path.basename(file_path)
        orig_file = "data_preprocessed/{}/mapping_to_{}/{}".format(language, to_language, base_name)
        translage_file = "data_preprocessed/{}/mapping_to_{}/{}_translate".format(language, to_language, base_name)
        print(orig_file)
        print(translage_file)

        with open(orig_file, 'w') as file:
            for s in samples:
                file.write(''.join(s[1]))

        with open(translage_file, 'w') as file:
            for s in samples:
                file.write(''.join(s[0]))


def create_samples(chunk, mapping_to):
    # return ''

    seq_len = 500

    chunk_size = len(chunk)
    seqs = []
    word = ''
    i = 0
    seq = ''
    while i < chunk_size:
        if i % 100000 == 0:
            print(i / chunk_size, "% splitted to sequences.", end='\r')
        if utils.is_delimiter(chunk[i]):
            # if word != '':
            #         words.append(word)
            # word = ''
            delimiter = chunk[i]
            while i + 1 < chunk_size and utils.is_delimiter(chunk[i + 1]):
                i += 1
                delimiter += chunk[i]
            if len(seq) + len(word) + len(delimiter) < seq_len:
                seq += word + delimiter
            else:
                seqs.append(seq)
                seq = word + delimiter
            word = ''
            # delimiters.append(delimiter)
        else:
            word += chunk[i]
        i += 1

    if word != '':
        if len(seq) + len(word) < seq_len:
            seq += word
            word = ''
    if seq != '':
        seqs.append(seq)
        seq = ''

    print()
    print("Creating samples.")

    valids = utils.get_valid_chars(mapping_to)

    seq_size = len(seqs)
    # create list of groups with 3 elements, original/translated/non valid chars
    samples = []
    # for seq in sequences:
    i = 0
    while i < seq_size:
        if i % 100 == 0:
            print(i / seq_size, "% sampled.", end='\r')
        # skip processing sequence if native letters less than 30 percent
        # native_letter_count = sum([1 for c in seq if utils.isNativeLetter(c, mapping_to)])
        # if is_train and native_letter_count * 3 < len(seq):
        #     continue

        # translate based on mapping, generated text should be the same size
        seq = u' ' + seqs[i] + u' '
        translit = []
        native = []
        for ind in range(1, len(seq)-1):
            trans_char = utils.toTranslit(seq[ind-1], seq[ind], seq[ind+1], mapping_to)
            translit.append(trans_char[0])
            trans_ind = 1
            native.append(seq[ind])
            while len(trans_char) > trans_ind:
                native.append(utils.EMPTY_CHAR)
                translit.append(trans_char[trans_ind])
                trans_ind += 1

        # example Թ": {"T": 0.9, "Th": 0.05, "TH": 0.05}
        # Թ\u2000 -> Th

        # validate both texts, changing unknown chars with #
        translit, non_valids = utils.validate(valids, translit)
        for ind in range(len(native)):
            if translit[ind] == utils.UNKNOWN_CHAR:
                native[ind] = utils.UNKNOWN_CHAR
        samples.append((translit, native, non_valids))
        i += 1
    del seqs

    return samples


if __name__ == '__main__':
    main()