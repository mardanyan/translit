# -*- coding: utf-8 -*

# from __future__ import print_function
import numpy as np
# import theano
# import theano.tensor as T
# import lasagne
# import codecs
# import json
import os
import argparse
import random
import utils
from datetime import datetime
import matplotlib.pyplot as plt
import sys



def main():

    parser = argparse.ArgumentParser(description='''Train you model specifying parameters''')
    parser.add_argument('--hdim', default=512, type=int, help='Dimension of hidden layers')
    parser.add_argument('--depth', default=2, type=int, help='Depth of network.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for learning.')
    parser.add_argument('--seq_len', default=100, type=int, help='Sequences size for splitting text for training.')
    parser.add_argument('--language', default="hy-AM", help='Specify language to train.')
    # parser.add_argument('--grad_clip', default=100, type=int, help='')
    # parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--epochs', default=10, type=int, help='Epochs of train.')
    # parser.add_argument('--model', default=None, help='')
    parser.add_argument('--model_name_prefix', default='model', help='Used for model name prefix.')
    # parser.add_argument('--start_from', default=0, type=float, help='')
    parser.add_argument('--model_path', type=str, help='Specify model path to save, or will we saved under languages/<lang>models/model_name_prefix***')
    parser.add_argument('--validate', type=bool, default=True, help='Evaluate percentage of validation data. Default:True')
    parser.add_argument('--data_size', type=int, default=5_000_000, help='Split date size in chars: Set 0 to train all data.')

    args = parser.parse_args()
   
    print("Loading Files")
    
    char_to_index, index_to_char, vocab_size, trans_to_index, index_to_trans, trans_vocab_size = \
        utils.load_vocabulary(language=args.language)
    (train_text, val_text, trans) = utils.load_language_data(language=args.language)

    # shuffle train data
    train_text = train_text.split('։')
    random.shuffle(train_text)
    train_text = '։'.join(train_text)

    if args.data_size != 0:
        val_size = round(args.data_size / 0.7 * 0.3)
        print("Data splitted, train:", args.data_size, ", val:", val_size)
        train_text = train_text[:args.data_size] # 226_849_593
        val_text = val_text[:val_size]       #34722649

    print("Building Network ...")

    model = utils.define_model(args.hdim, args.depth, trans_vocab_size, vocab_size, is_train=True)
    print(model.summary())
    
    # if args.model:
    #     f = np.load('languages/' + args.language + '/models/' + args.model)
    #     param_values = [np.float32(f[i]) for i in range(len(f))]
    #    # lasagne.layers.set_all_param_values(output_layer, param_values)

    print("Preparing data ...")
    # step_cnt = 0
    date_at_beginning = datetime.now()
    # last_time = date_at_beginning

    # avg_cost = 0.0
    # count = 0
    # num_of_samples = 0
    # num_of_chars = 0
    (x_train, y_train) = utils.data_generator(train_text, args.seq_len, trans, trans_to_index, char_to_index, is_train=True)
    print("Training ...")
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size)

    print(history.history)

    loss = history.history["loss"][-1]

    time_now = datetime.now()

    if args.model_path:
        file_name = args.model_path
    else:
        dir = 'languages/{}/models/'.format(args.language)
        if not os.path.exists(dir):
            os.mkdir(dir)
        image_name = 'languages/{}/models/{}.{}-{}-{}--{}-{}.hdim{}.depth{}.seq_len{}.bs{}.time{:.3f}.epoch{}.loss{:.3f}'.format(
            args.language, args.model_name_prefix, str(time_now.day), str(time_now.month), str(time_now.year),
            str(time_now.hour), str(time_now.minute), args.hdim, args.depth, args.seq_len, args.batch_size, (time_now -
            date_at_beginning).total_seconds()/60, args.epochs, loss)
        file_name = image_name + '.h5'



    model.save(file_name)
    print('Model saved:', file_name)

    print("Validate....")
    if args.validate:
        (x_test, y_test) = utils.data_generator(val_text, args.seq_len, trans, trans_to_index, char_to_index, is_train=True)
        score = model.evaluate(x_test, y_test, verbose=1)
        print("Evaluated on validation data", score)
    else:
        print("Validation disabled.")

    print("Save accuracy image " + image_name + '_acc.png')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['acc', 'val_acc'], loc='upper left')
    #plt.show()
    plt.savefig(image_name + '_acc.png')

    print("Save accuracy image " + image_name + '_loss.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig(image_name + '_loss.png')


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

        
if __name__ == '__main__':
    main()
