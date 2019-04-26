# -*- coding: utf-8 -*


# import numpy as np
# import codecs
# import json
import argparse
import utils
# from datetime import datetime

PRINT_FREQ = 1


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='hy')
    args = parser.parse_args()
   
    print("Loading Files")
    (train_text, val_text, trans) = utils.load_language_data(language = args.language)

    print("Making Vocabulary Files")
    utils.make_vocabulary_files(train_text, args.language, trans)


if __name__ == '__main__':
    main()
