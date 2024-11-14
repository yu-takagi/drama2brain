#!/bin/bash

mkdir ./data/gensim_models
cd ./data/gensim_models

# For glove
mkdir ./glove
cd ./glove
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d . && rm glove.840B.300d.zip
# See more info: https://nlp.stanford.edu/projects/glove/

cd ..

# For Word2Vec
# At first, download GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/
# (You need a google account)
mkdir ./word2vec
cd ./word2vec
gzip -d GoogleNews-vectors-negative300.bin.gz
# See more info: https://code.google.com/archive/p/word2vec/
