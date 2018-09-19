# Automatic transliteration


## Prepare the data for a language

The first, we prepare the corpus

1. Download the [Wiki dump](https://dumps.wikimedia.org/hywiki/20180901/) e.g. https://dumps.wikimedia.org/hywiki/20180901/hywiki-20180901-pages-articles-multistream.xml.bz2 
2. Extract the dump using [WikiExtractor](https://github.com/attardi/wikiextractor)
3. Remove the remaining tags from exported files(in first and last lines) (strings starting with '<')
4. Split data thre parts (70% - `train.txt`, 11% - `val.txt`, 15% - `test.txt`) and store under `languages/LANG_CODE/data` folder
... points 3 and 4 can be done running ths script preparewikitext.py

Next we add some language specific configuration files:

1. Populate the `languages/LANG_CODE/transliteration.json` file with romanization rules
2. Populate the `languages/LANG_CODE/long_letters.json` file with an array of the multi-symbol letters of the current language
3. Run `make_vocab.py` to generate the "vocabulary"


## Generate model

Before training on the corpus we need to compute the vocabularies by the following command:

	python make_vocab.py --language hy-AM

The actual training is initiated by a command like this:

    python -u train.py --hdim 1024 --depth 2 --batch_size 200 --seq_len 30 --language hy-AM &> log.txt



    
# Results

For results need add at least one more language, generate model for 2 language and calculate accuracy.