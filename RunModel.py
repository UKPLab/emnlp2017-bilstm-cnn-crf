#!/usr/bin/python
#Usage: python RunModel.py modelPath inputPath"
from __future__ import print_function
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

with open(inputPath, 'r') as f:
    text = f.read()


# :: Load the model ::
lstmModel = BiLSTM()
lstmModel.loadModel(modelPath)


# :: Prepare the input ::
sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
addCharInformation(sentences)
addCasingInformation(sentences)

dataMatrix = createMatrices(sentences, lstmModel.mappings)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    tokenTags = tags[sentenceIdx]
    for tokenIdx in range(len(tokens)):
        print("%s\t%s" % (tokens[tokenIdx], tokenTags[tokenIdx]))
    print("")