from __future__ import print_function
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

def perpareDataset(embeddingsPath, datasetFiles, frequencyThresholdUnknownTokens=50, reducePretrainedEmbeddings=False, commentSymbol=None):
    """
    Reads in the pre-trained embeddings (in text format) from embeddingsPath and prepares those to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least frequencyThresholdUnknownTokens times
    
    # Arguments:
        datasetName: The name of the dataset. This function creates a pkl/datasetName.pkl file
        embeddingsPath: Full path to the pre-trained embeddings file. File must be in text format.
        datasetFiles: Full path to the [train,dev,test]-file
        tokenIndex: Column index for the token 
        frequencyThresholdUnknownTokens: Unknown words are added, if they occure more than frequencyThresholdUnknownTokens times in the train set
        reducePretrainedEmbeddings: Set to true, then only the embeddings needed for training will be loaded
        commentSymbol: If not None, lines starting with this symbol will be skipped
    """
    embeddingsName = os.path.splitext(embeddingsPath)[0]
    datasetName = "_".join(sorted([datasetFile[0] for datasetFile in datasetFiles])+[embeddingsName])
    outputPath = 'pkl/'+datasetName+'.pkl'
    
    if os.path.isfile(outputPath):
        logging.info("Using existent pickle file: %s" % outputPath)
        return outputPath
    
    #Check that the embeddings file exists
    if not os.path.isfile(embeddingsPath):
        if embeddingsPath == 'levy_deps.words':
            getLevyDependencyEmbeddings()
        elif embeddingsPath == '2014_tudarmstadt_german_50mincount.vocab.gz':
            getReimersEmbeddings()
        else:
            print("The embeddings file %s was not found" % embeddingsPath)
            exit()
    
    logging.info("Generate new embeddings files for a dataset: %s" % outputPath)
    
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
                
                
        for datasetFile in datasetFiles:
            dataColumnsIdx = {y:x for x,y in datasetFile[1].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % datasetName
            
            for dataset in ['train.txt', 'dev.txt', 'test.txt']:  
                createDict(datasetPath+dataset, tokenIdx, neededVocab)

        
    
    # :: Read in word embeddings ::   
    logging.info("Read file: %s" % embeddingsPath) 
    word2Idx = {}
    embeddings = []
    
    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
    
    embeddingsDimension = None
    
    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]
        
        if embeddingsDimension == None:
            embeddingsDimension = len(split)-1
            
        if (len(split)-1) != embeddingsDimension:  #Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension) 
            embeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension) #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
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
        for datasetFile in datasetFiles:
            dataColumnsIdx = {y:x for x,y in datasetFile[1].items()}
            tokenIdx = dataColumnsIdx['tokens']
            datasetPath = 'data/%s/' % datasetFile[0]            
            createFD(datasetPath+'train.txt', tokenIdx, fd, word2Idx)        
        
        addedWords = 0
        for word, freq in fd.most_common(10000):
            if freq < frequencyThresholdUnknownTokens:
                break
            
            addedWords += 1        
            word2Idx[word] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)  #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)
            
            assert(len(word2Idx) == len(embeddings))
        
        
        logging.info("Added words: %d" % addedWords)
    embeddings = np.array(embeddings)
    
    pklObjects = {'embeddings': embeddings, 'word2Idx': word2Idx, 'datasets': {}}
    
    casing2Idx = getCasingVocab()
    for datasetName, datasetColumns in datasetFiles:
        trainData = 'data/%s/train.txt' % datasetName 
        devData = 'data/%s/dev.txt' % datasetName 
        testData = 'data/%s/test.txt' % datasetName 
        paths = [trainData, devData, testData]
    
        pklObjects['datasets'][datasetName] = createPklFiles(paths, word2Idx, casing2Idx, datasetColumns, commentSymbol, padOneTokenSentence=True)
    
    
    f = open(outputPath, 'wb')
    pkl.dump(pklObjects, f, -1)
    f.close()
    
    logging.info("DONE - Embeddings file saved: %s" % outputPath)
    
    return outputPath


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


def createMatrices(sentences, mappings, padOneTokenSentence=True):
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
    
  
  
def createPklFiles(datasetFiles, word2Idx, casing2Idx, cols, commentSymbol=None, valTransformation=None, padOneTokenSentence=False):       
              
    trainSentences = readCoNLL(datasetFiles[0], cols, commentSymbol, valTransformation)
    devSentences = readCoNLL(datasetFiles[1], cols, commentSymbol, valTransformation)
    testSentences = readCoNLL(datasetFiles[2], cols, commentSymbol, valTransformation)    
   
    mappings = createMappings(trainSentences+devSentences+testSentences)
    mappings['tokens'] = word2Idx
    mappings['casing'] = casing2Idx
                
    
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
    
  
    trainMatrix = createMatrices(trainSentences, mappings)
    devMatrix = createMatrices(devSentences, mappings)
    testMatrix = createMatrices(testSentences, mappings)       

    
    data = { 'mappings': mappings,
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': testMatrix
            }        
       
    
    return data

def createMappings(sentences):
    sentenceKeys = list(sentences[0].keys())
    sentenceKeys.remove('tokens')
    
    
    vocabs = {name:{'O':0} for name in sentenceKeys} #Use 'O' also for padding
    #vocabs = {name:{} for name in sentenceKeys}
    for sentence in sentences:
        for name in sentenceKeys:
            for item in sentence[name]:              
                if item not in vocabs[name]:
                    vocabs[name][item] = len(vocabs[name]) 
                    
   
    return vocabs  


    
def loadDatasetPickle(embeddingsPickle):
    """ Loads the cPickle file, that contains the word embeddings and the datasets """
    f = open(embeddingsPickle, 'rb')
    pklObjects = pkl.load(f)
    f.close()
    

        
        
    return pklObjects['embeddings'], pklObjects['word2Idx'], pklObjects['datasets']



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
    
   
