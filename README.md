# Multilingual transliteration

> Bring all mixed texts to original language…

A growing rate of users generating content in "Latinized" formats pring huge problem of transliteration. The problem exists for mangy languages like, India, Greek, Armenia...
In this work we provide solutinal for this problem based on Neural Networks.
Using Neural Network we present solution with high accuracy to translate any Latinized text to native canonical format.

Transliterated text can break key aspects of all the highest scaled and highest value platforms today:

- Search

- Advertising

- Recommendations – eg products, feed items

- Content moderation – eg spam, abuse, porn

- Security – eg terrorist communications

- Dialogue systems – eg virtual assistants, chat bots

- Text analysis – sentiment analysis, entity recognition…

### Supported transliterations
In the current implemented available transliteration 
- Latin -> Armenian
- Cyrillic -> Armenian


## Architecture

Using Recurrent Neural Network expecially Bid-LSTM architecture we provide character level language model.

## Integration more languages

Platform designed specially for integrate more languages.

### How to integrate more languages ?

TODO.

## Achived results

## Accuracy validation based on generated dataset

| Input alphabets |   Languages   | Example rows |  Accuracy |
| --------------- | ------------- | -------- |-------- |
|     Latin       |   Armenian    |  -  | 97%   |
| Latin, Cyrillic |   Armenian    |  -  | 97%   |
|     Latin       |   English, Armenian    |  -  | 97%   |
|     Latin       |   Armenian    |  -  | 97%   |

# Accuracy validation based on human expectations

| Input alphabets | Languages | Accuracy |
| --------------- | --------- | -------- |
|     Latin       | Armenian  |    *     |
| Latin, Cyrillic | Armenian  |    *   |


## Requirements

Please check requirements.txt:

> pip install -r requirements.txt

Python 3.6 or later is required.

# Program overview


## Training process

Training process consists of the followings steps

### Preprocessing

Preprocess data based on romanization rule.<br>

Each languages should be preprocessed like.
 
    python data_preprocessed.py --language=hy


### Indexing
Indexing based on mappings.

    python create_indexes.py --languages=hy --data_size=50_000_000

### Training
 
    python train.py --epoch=10 --depth=1 --seq_len=30 --train_size=5_000 --languages=hy-en,hy-ru,ru-en

## Run the program specifying model

    python predict.py --language=hy --model=models/2019-5-1_12\:59_hy/m.hy.hdim512.depth1.seq_len30.bs32.time0.200.epoch2.loss2.908.h5

# Source of the data

Armenian Wikipedia as the easiest available large corpus of Armenian text. The dumps are available [here](https://dumps.wikimedia.org/hywiki)

# Models

In the repository provided best models for different combinations of transliteration.


# Thanks
Special thanks to [Adam Bittlingmayer](https://www.linkedin.com/in/bittlingmayer) and [YerevanN](http://yerevann.github.io/) to valuable advices.