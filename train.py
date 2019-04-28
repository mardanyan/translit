# -*- coding: utf-8 -*

import argparse
import random
import utils
import glob
import re

from datetime import datetime


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
    Train you model specifying parameters
    
    Use-cases:
        python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy-en,hy-ru,ru-en
        python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy,ru-en,en
    
    ''')
    parser.add_argument('--hdim', default=512, type=int, help='Dimension of hidden layers')
    parser.add_argument('--depth', default=2, type=int, help='Depth of network.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for learning.')
    parser.add_argument('--seq_len', default=100, type=int, help='Sequences size for splitting text for training.')
    parser.add_argument('--languages', default=None, required=True, help='Specify language to train.')
    # parser.add_argument('--grad_clip', default=100, type=int, help='')
    # parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--epoch', default=10, type=int, help='Epochs of train.')
    # parser.add_argument('--model', default=None, help='')
    parser.add_argument('--prefix', default='m', help='Used for model name prefix.')
    # parser.add_argument('--start_from', default=0, type=float, help='')
    parser.add_argument('--model_path', type=str, help='Specify model path to save, or will we saved under languages/<lang>models/model_name_prefix***')
    parser.add_argument('--validate', type=bool, default=True, help='Evaluate percentage of validation data. Default:True')
    parser.add_argument('--data_size', type=int, default=5_000_000, help='Split date size in chars: Set 0 to train all data.')

    args = parser.parse_args()

    languages = utils.parse_languages(args.languages)

    print("Languages to train: " + str(languages))

    list_languages = []
    for key, value in languages.items():
        list_languages.append(key)
        if len(value) == 0:
            dirs = glob.glob('data_preprocessed/' + key + "/mapping_to_*")
            print(dirs)
            for dir in dirs:
                p = re.compile("mapping_to_(.*)")
                result = p.search(dir)
                list_languages.append(result.group(1))
        else:
            list_languages.extend(value)

    list_languages = list(dict.fromkeys(list_languages))
    list_languages.sort()

    print(list_languages)

   
    print("Loading Files")

    char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size = \
                                                            utils.load_vocabulary(list_languages)

    print("vocab size: ", vocab_size)
    print("trans vocab size: ", trans_vocab_size)

    train_text, train_translated_text = utils.load_preprocessed_data(languages, args.data_size, 'train')

    print("Train text size:", len(train_text))
    print("Train translated text size:", len(train_translated_text))

    print(char_to_index)

    print('а' in char_to_index)
    print(ord('а'))

    x_train = utils.generator_biniries(train_text, args.seq_len, char_to_index)


    return



    # shuffle train data
    train_text = train_text.split('։')
    random.shuffle(train_text)
    train_text = '։'.join(train_text)

    if args.data_size != 0:
        val_size = round(args.data_size / 0.7 * 0.3)
        print("Data splitted, train:", args.data_size, ", val:", val_size)
        train_text = train_text[:args.data_size] # 226_849_593
        val_text = val_text[:val_size]       #34722649

    import utilsk

    print("Building Network ...")

    model = utilsk.define_model(args.hdim, args.depth, trans_vocab_size, vocab_size, is_train=True)
    print(model.summary())
    
    print("Preparing data ...")
    before_fit_time = datetime.now()
    (x_train, y_train) = utils.data_generator(train_text, args.seq_len, trans, trans_to_index, char_to_index, is_train=True)

    print("Training ...")
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=args.epoch, batch_size=args.batch_size)
    loss = history.history["loss"][-1]
    print(history.history)


    # save model
    model_file_path = utils.get_model_file_path(args, before_fit_time, loss)
    model.save_weights(model_file_path)
    print('Model saved:', model_file_path)

    print("Validate exact....")
    if args.validate:
        (x_test, y_test) = utils.data_generator(val_text, args.seq_len, trans, trans_to_index, char_to_index, is_train=True)
        score = model.evaluate(x_test, y_test, verbose=1)
        print("Evaluated on validation data", score)
    else:
        print("Validation disabled.")

    utils.save_acc_loss_results(args, history)
    utils.write_results_file(args, history, train_text, val_text)



    #     sample_cost = train(x, np.reshape(y,(-1,vocab_size)))
    #     sample_cost = float(sample_cost)
    #     count += 1
    #     num_of_samples += x.shape[0]
    #     num_of_chars += x.shape[0] * x.shape[1]
    #
    #     if (time_now - last_time).total_seconds() > 60 * 1: # 10 minutes
    #         print('Computing validation loss...')
    #         val_cost = 0.0
    #         val_count = 0.0
    #         for ((x_val, y_val, indices, delimiters), non_valids_list) in utils.data_generator(val_text, args.seq_len, args.batch_size, trans, trans_to_index, char_to_index, is_train = False):
    #             val_cost += x_val.shape[0] *cost(x_val,np.reshape(y_val,(-1,vocab_size)))
    #             val_count += x_val.shape[0]
    #         print('Validation loss is {}'.format(val_cost/val_count))
    #
    #         file_name = 'languages/{}/models/{}.hdim{}.depth{}.seq_len{}.bs{}.time{:4f}.epoch{}.loss{:.4f}'.format(args.language, args.model_name_prefix, args.hdim, args.depth, args.seq_len, args.batch_size, (time_now - date_at_beginning).total_seconds()/60, epoch, val_cost/val_count)
    #         print("saving to -> " + file_name)
    #         # np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
    #         # last_time = datetime.now()
    #
    #     print("On step #{} loss is {:.4f}, samples passed {}, chars_passed {}, {:.4f}% of an epoch {} time passed {:4f}"\
    #           .format(count, sample_cost, num_of_samples, num_of_chars, 100.0*num_of_chars/len(train_text), epoch, (time_now - date_at_beginning).total_seconds()/60.0))
    #
    #     avg_cost += sample_cost
    # date_after = datetime.now()
    # print("After epoch {} average loss is {:.4f} Time {} sec.".format( epoch , avg_cost/count, (date_after - date_at_beginning).total_seconds()))


# def main():
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
#     Train you model specifying parameters
#
#     Use-cases:
#         python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy-en,hy-ru,ru-en
#         python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy,ru-en,en
#
#     ''')
#     parser.add_argument('--hdim', default=512, type=int, help='Dimension of hidden layers')
#     parser.add_argument('--depth', default=2, type=int, help='Depth of network.')
#     parser.add_argument('--batch_size', default=32, type=int, help='Batch size for learning.')
#     parser.add_argument('--seq_len', default=100, type=int, help='Sequences size for splitting text for training.')
#     parser.add_argument('--languages', default=None, required=True, help='Specify language to train.')
#     # parser.add_argument('--grad_clip', default=100, type=int, help='')
#     # parser.add_argument('--lr', default=0.01, type=float, help='')
#     parser.add_argument('--epoch', default=10, type=int, help='Epochs of train.')
#     # parser.add_argument('--model', default=None, help='')
#     parser.add_argument('--prefix', default='m', help='Used for model name prefix.')
#     # parser.add_argument('--start_from', default=0, type=float, help='')
#     parser.add_argument('--model_path', type=str,
#                         help='Specify model path to save, or will we saved under languages/<lang>models/model_name_prefix***')
#     parser.add_argument('--validate', type=bool, default=True,
#                         help='Evaluate percentage of validation data. Default:True')
#     parser.add_argument('--data_size', type=int, default=5_000_000,
#                         help='Split date size in chars: Set 0 to train all data.')
#
#     args = parser.parse_args()
#
#     languages = utils.parse_languages(args.languages)
#
#     print("Languages to train: " + str(languages))
#
#     print("Loading Files")
#
#     return
#
#     char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size = \
#         utils.load_vocabulary(language=args.language)
#     (train_text, val_text, trans) = utils.load_language_data(language=args.language)
#
#     # shuffle train data
#     train_text = train_text.split('։')
#     random.shuffle(train_text)
#     train_text = '։'.join(train_text)
#
#     if args.data_size != 0:
#         val_size = round(args.data_size / 0.7 * 0.3)
#         print("Data splitted, train:", args.data_size, ", val:", val_size)
#         train_text = train_text[:args.data_size]  # 226_849_593
#         val_text = val_text[:val_size]  # 34722649
#
#     import utilsk
#
#     print("Building Network ...")
#
#     model = utilsk.define_model(args.hdim, args.depth, trans_vocab_size, vocab_size, is_train=True)
#     print(model.summary())
#
#     print("Preparing data ...")
#     before_fit_time = datetime.now()
#     (x_train, y_train) = utils.data_generator(train_text, args.seq_len, trans, trans_to_index, char_to_index,
#                                               is_train=True)
#
#     print("Training ...")
#     history = model.fit(x_train, y_train, validation_split=0.1, epochs=args.epoch, batch_size=args.batch_size)
#     loss = history.history["loss"][-1]
#     print(history.history)
#
#     # save model
#     model_file_path = utils.get_model_file_path(args, before_fit_time, loss)
#     model.save_weights(model_file_path)
#     print('Model saved:', model_file_path)
#
#     print("Validate exact....")
#     if args.validate:
#         (x_test, y_test) = utils.data_generator(val_text, args.seq_len, trans, trans_to_index, char_to_index,
#                                                 is_train=True)
#         score = model.evaluate(x_test, y_test, verbose=1)
#         print("Evaluated on validation data", score)
#     else:
#         print("Validation disabled.")
#
#     utils.save_acc_loss_results(args, history)
#     utils.write_results_file(args, history, train_text, val_text)
#
#     #     sample_cost = train(x, np.reshape(y,(-1,vocab_size)))
#     #     sample_cost = float(sample_cost)
#     #     count += 1
#     #     num_of_samples += x.shape[0]
#     #     num_of_chars += x.shape[0] * x.shape[1]
#     #
#     #     if (time_now - last_time).total_seconds() > 60 * 1: # 10 minutes
#     #         print('Computing validation loss...')
#     #         val_cost = 0.0
#     #         val_count = 0.0
#     #         for ((x_val, y_val, indices, delimiters), non_valids_list) in utils.data_generator(val_text, args.seq_len, args.batch_size, trans, trans_to_index, char_to_index, is_train = False):
#     #             val_cost += x_val.shape[0] *cost(x_val,np.reshape(y_val,(-1,vocab_size)))
#     #             val_count += x_val.shape[0]
#     #         print('Validation loss is {}'.format(val_cost/val_count))
#     #
#     #         file_name = 'languages/{}/models/{}.hdim{}.depth{}.seq_len{}.bs{}.time{:4f}.epoch{}.loss{:.4f}'.format(args.language, args.model_name_prefix, args.hdim, args.depth, args.seq_len, args.batch_size, (time_now - date_at_beginning).total_seconds()/60, epoch, val_cost/val_count)
#     #         print("saving to -> " + file_name)
#     #         # np.save(file_name, lasagne.layers.get_all_param_values(output_layer))
#     #         # last_time = datetime.now()
#     #
#     #     print("On step #{} loss is {:.4f}, samples passed {}, chars_passed {}, {:.4f}% of an epoch {} time passed {:4f}"\
#     #           .format(count, sample_cost, num_of_samples, num_of_chars, 100.0*num_of_chars/len(train_text), epoch, (time_now - date_at_beginning).total_seconds()/60.0))
#     #
#     #     avg_cost += sample_cost
#     # date_after = datetime.now()
#     # print("After epoch {} average loss is {:.4f} Time {} sec.".format( epoch , avg_cost/count, (date_after - date_at_beginning).total_seconds()))
#

if __name__ == '__main__':
    main()
