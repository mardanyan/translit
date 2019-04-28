# Automatic transliteration

Many languages have their own non-Latin alphabets but the web is full of content in those languages written in Latin letters, which makes it inaccessible to various NLP tools (e.g. automatic translation). Transliteration is the process of converting the romanized text back to the original writing system. In theory every language has a strict set of romanization rules, but in practice people do not follow the rules and most of the romanized content is hard to transliterate using rule based algorithms. We believe this problem is solvable using the state of the art NLP tools, and we demonstrate a high quality solution for Armenian based on recurrent neural networks.
This is a tool to transliterate inconsistently romanized text. It is tested on Armenian (hy-AM)

## Problem description
 Every language has its own story, and these stories are usually not known outside their own communities. Wikipedia has similar stories for [Greek](https://en.wikipedia.org/wiki/Greeklish), [Persian](https://en.wikipedia.org/wiki/Fingilish). TODO 

## Source of the data

Armenian Wikipedia as the easiest available large corpus of Armenian text. The dumps are available [here](https://dumps.wikimedia.org/hywiki)

## Romanization rules

To generate the input sequences for the network we need to romanize the texts. We use probabilistic rules, as different people prefer different romanizations. Armenian alphabet has 39 characters, while Latin has only 26. Some of the Armenian letters are romanized in a unique way



## Prepare the data for a language

The first, we prepare the corpus

1. Download the [Wiki dump](https://dumps.wikimedia.org/hywiki/20180901/) (e.g. https://dumps.wikimedia.org/hywiki/20180901/hywiki-20180901-pages-articles-multistream.xml.bz2) 
2. Extract the dump using [WikiExtractor](https://github.com/attardi/wikiextractor)
3. Remove the remaining tags from exported files(in first and last lines) (strings starting with '<')
4. Split data thre parts (70% - `train.txt`, 15% - `val.txt`, 15% - `test.txt`) and store under `languages/LANG_CODE/data` folder

points 3 and 4 can be done running preparewikitext.py

Next we add some language specific configuration files:

1. Populate the `languages/LANG_CODE/transliteration.json` file with romanization rules
2. Populate the `languages/LANG_CODE/long_letters.json` file with an array of the multi-symbol letters of the current language
3. Run `make_vocab.py` to generate the "vocabulary"


## Generate model

Before training on the corpus we need to compute the vocabularies by the following command:

	python make_vocab.py --language hy-AM

The actual training is initiated by a command like this:

    python train.py --num_epochs 10


## Predict

    python predict.py --model=languages/hy-AM/models/model.20-4-2019--20-47.hdim1024.depth2.seq_len30.bs100.time5.931.epoch10.loss0.082.h5
 
 
# Pipeline 

#extract xml files into small
    python WikiExtractor.py -b=50M enwiki-20190420-pages-articles-multistream.xml
    
    sed -i -e "/<\/doc/d" *
    sed -i -e "/<doc/d" *
    sed  -i '/^[[:space:]]*$/d' *
    
# preprocess data for hy    
    python data_preprocessed.py --language=hy
# preprocess data for ru 
    python data_preprocessed.py --language=ru
    
# create indexes mapping
    python create_indexes.py --languages=hy --data_size=50_000_000
    
# train model by specifying preprocessed data
    python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy-en,hy-ru,ru-en    
    python train.py --depth=10 --seq_len=30 --data_size=5_000 --languages=hy,ru-en