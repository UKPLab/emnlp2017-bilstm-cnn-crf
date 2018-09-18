from __future__ import (division, absolute_import, print_function, unicode_literals)
import os
import numpy as np
import gzip
import os.path
import nltk
import logging
from nltk import FreqDist

from .WordEmbeddings import wordNormalize
from .CoNLL import readCoNLL

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl
    from io import open

def perpareDataset(embeddingsPath, datasets, frequencyThresholdUnknownTokens=50, reducePretrainedEmbeddings=False, valTransformations=None, padOneTokenSentence=True):
    """
    Reads in the pre-trained embeddings (in text format) from embeddingsPath and prepares those to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least frequencyThresholdUnknownTokens times
    
    # Arguments:
        embeddingsPath: Full path to the pre-trained embeddings file. File must be in text format.
        datasetFiles: Full path to the [train,dev,test]-file
        frequencyThresholdUnknownTokens: Unknown words are added, if they occure more than frequencyThresholdUnknownTokens times in the train set
        reducePretrainedEmbeddings: Set to true, then only the embeddings needed for training will be loaded
        valTransformations: Column specific value transformations
        padOneTokenSentence: True to pad one sentence tokens (needed for CRF classifier)
    """
    embeddingsName = os.path.splitext(embeddingsPath)[0]
    pklName = "_".join(sorted(datasets.keys()) + [embeddingsName])
    outputPath = 'pkl/' + pklName + '.pkl'

    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath

    casing2Idx = getCasingVocab()
    embeddings, word2Idx = readEmbeddings(embeddingsPath, datasets, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings)
    
    mappings = {'tokens': word2Idx, 'casing': casing2Idx}
    pklObjects = {'embeddings': embeddings, 'mappings': mappings, 'datasets': datasets, 'data': {}}

    for datasetName, dataset in datasets.items():
        datasetColumns = dataset['columns']
        commentSymbol = dataset['commentSymbol']

        trainData = 'data/%s/train.txt' % datasetName 
        devData = 'data/%s/dev.txt' % datasetName 
        testData = 'data/%s/test.txt' % datasetName 
        paths = [trainData, devData, testData]

        logging.info(":: Transform "+datasetName+" dataset ::")
        pklObjects['data'][datasetName] = createPklFiles(paths, mappings, datasetColumns, commentSymbol, valTransformations, padOneTokenSentence)

    
    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()

    return pklObjects['embeddings'], pklObjects['mappings'], pklObjects['data']



def readEmbeddings(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens, reducePretrainedEmbeddings):
    """
    Reads the embeddingsPath.
    :param embeddingsPath: File path to pretrained embeddings
    :param datasetName:
    :param datasetFiles:
    :param frequencyThresholdUnknownTokens:
    :param reducePretrainedEmbeddings:
    :return:
    """
    # Check that the embeddings file exists
    if not os.path.isfile(embeddingsPath):
        if embeddingsPath in ['komninos_english_embeddings.gz', 'levy_english_dependency_embeddings.gz', 'reimers_german_embeddings.gz']:
            getEmbeddings(embeddingsPath)
        else:
            print("The embeddings file %s was not found" % embeddingsPath)
            exit()

    logging.info("Generate new embeddings files for a dataset")

    neededVocab = {}
    if reducePretrainedEmbeddings:
        logging.info("Compute which tokens are required for the experiment")

        def createDict(filename, tokenPos, vocab):
            for line in open(filename):
                if line.startswith('#'):
                    continue
                splits = line.strip().split()
                if len(splits) > 1:
                    word = splits[tokenPos]
                    wordLower = word.lower()
                    wordNormalized = wordNormalize(wordLower)

                    vocab[word] = True
                    vocab[wordLower] = True
                    vocab[wordNormalized] = True

        for dataset_name, dataset in datasetFiles.items():
            dataColumnsIdx = {y: x for x, y in dataset['columns'].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % dataset_name

            for dataset_file_name in ['train.txt', 'dev.txt', 'test.txt']:
                createDict(datasetPath + dataset_file_name, tokenIdx, neededVocab)

    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")

    embeddingsDimension = None

    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    # Extend embeddings file with new tokens
    def createFD(filename, tokenIndex, fd, word2Idx):
        for line in open(filename):
            if line.startswith('#'):
                continue

            splits = line.strip().split()

            if len(splits) > 1:
                word = splits[tokenIndex]
                wordLower = word.lower()
                wordNormalized = wordNormalize(wordLower)

                if word not in word2Idx and wordLower not in word2Idx and wordNormalized not in word2Idx:
                    fd[wordNormalized] += 1

    if frequencyThresholdUnknownTokens != None and frequencyThresholdUnknownTokens >= 0:
        fd = nltk.FreqDist()
        for datasetName, datasetFile in datasetFiles.items():
            dataColumnsIdx = {y: x for x, y in datasetFile['columns'].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % datasetName
            createFD(datasetPath + 'train.txt', tokenIdx, fd, word2Idx)

        addedWords = 0
        for word, freq in fd.most_common(10000):
            if freq < frequencyThresholdUnknownTokens:
                break

            addedWords += 1
            word2Idx[word] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

            assert (len(word2Idx) == len(embeddings))

        logging.info("Added words: %d" % addedWords)
    embeddings = np.array(embeddings)

    return embeddings, word2Idx


def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)

def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       
def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing

def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}


def createMatrices(sentences, mappings, padOneTokenSentence):
    data = []
    numTokens = 0
    numUnknownTokens = 0    
    missingTokens = FreqDist()
    paddedSentences = 0

    for sentence in sentences:
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    numTokens += 1
                    idx = str2Idx['UNKNOWN_TOKEN']
                    
                    if entry in str2Idx:
                        idx = str2Idx[entry]
                    elif entry.lower() in str2Idx:
                        idx = str2Idx[entry.lower()]
                    elif wordNormalize(entry) in str2Idx:
                        idx = str2Idx[wordNormalize(entry)]
                    else:
                        numUnknownTokens += 1    
                        missingTokens[wordNormalize(entry)] += 1
                        
                    row['raw_tokens'].append(entry)
                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)
                
        if len(row['tokens']) == 1 and padOneTokenSentence:
            paddedSentences += 1
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)
            
        data.append(row)
    
    if numTokens > 0:           
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens/float(numTokens)*100))
        
    return data
    
  
  
def createPklFiles(datasetFiles, mappings, cols, commentSymbol, valTransformation, padOneTokenSentence):
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol, valTransformation)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol, valTransformation)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol, valTransformation)    
   
    extendMappings(mappings, trainSentences+devSentences+testSentences)

                
    
    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset
    
    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    
    addCharInformation(testSentences)   
    addCasingInformation(testSentences)

    logging.info(":: Create Train Matrix ::")
    trainMatrix = createMatrices(trainSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Dev Matrix ::")
    devMatrix = createMatrices(devSentences, mappings, padOneTokenSentence)

    logging.info(":: Create Test Matrix ::")
    testMatrix = createMatrices(testSentences, mappings, padOneTokenSentence)

    
    data = {
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def extendMappings(mappings, sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens') #No need to map tokens

    for sentence in sentences:
        for name in sentenceKeys:
            if name not in mappings:
                mappings[name] = {'O':0} #'O' is also used for padding

            for item in sentence[name]:              
                if item not in mappings[name]:
                    mappings[name][item] = len(mappings[name])



    

def getEmbeddings(name):
    if not os.path.isfile(name):
        download("https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/"+name)



def getLevyDependencyEmbeddings():
    """
    Downloads from https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    the dependency based word embeddings and unzips them    
    """ 
    if not os.path.isfile("levy_deps.words.bz2"):
        print("Start downloading word embeddings from Levy et al. ...")
        os.system("wget -O levy_deps.words.bz2 http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2")
    
    print("Start unzip word embeddings ...")
    os.system("bzip2 -d levy_deps.words.bz2")

def getReimersEmbeddings():
    """
    Downloads from https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/
    embeddings for German
    """
    if not os.path.isfile("2014_tudarmstadt_german_50mincount.vocab.gz"):
        print("Start downloading word embeddings from Reimers et al. ...")
        os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2014_german_embeddings/2014_tudarmstadt_german_50mincount.vocab.gz")
    
   

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
    from urllib.request import urlretrieve
else:
    import urllib2
    import urlparse
    from urllib import urlretrieve


def download(url, destination=os.curdir, silent=False):
    filename = os.path.basename(urlparse.urlparse(url).path) or 'downloaded.file'

    def get_size():
        meta = urllib2.urlopen(url).info()
        meta_func = meta.getheaders if hasattr(
            meta, 'getheaders') else meta.get_all
        meta_length = meta_func('Content-Length')
        try:
            return int(meta_length[0])
        except:
            return 0

    def kb_to_mb(kb):
        return kb / 1024.0 / 1024.0

    def callback(blocks, block_size, total_size):
        current = blocks * block_size
        percent = 100.0 * current / total_size
        line = '[{0}{1}]'.format(
            '=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3.1f}/{3:3.1f} MB'
        sys.stdout.write(
            status.format(
                percent, line, kb_to_mb(current), kb_to_mb(total_size)))

    path = os.path.join(destination, filename)

    logging.info(
        'Downloading: {0} ({1:3.1f} MB)'.format(url, kb_to_mb(get_size())))
    try:
        (path, headers) = urlretrieve(url, path, None if silent else callback)
    except:
        os.remove(path)
        raise Exception("Can't download {0}".format(path))
    else:
        print()
        logging.info('Downloaded to: {0}'.format(path))

    return path