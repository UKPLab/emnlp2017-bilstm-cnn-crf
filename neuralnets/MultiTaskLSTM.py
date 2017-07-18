"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging used for multi-task learning.

Author: Nils Reimers
License: CC BY-SA 3.0
"""

from __future__ import print_function
from util import BIOF1Validation

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging
from .keraslayers.ChainCRF import ChainCRF


class MultiTaskLSTM:    
    earlyStopping = 0
    mainModelName = None
   
        
    addFeatureDimensions = 10
    miniBatchSize = 32
    learning_rate_updates = {'adam': {1:0.01, 4:0.005, 7:0.001}, 'sgd': {1: 0.1, 3:0.05, 5:0.01} } 
    epoch = 0

    
    def __init__(self, embeddings, datasetTuples, params=None):
        """
        datasetTuples: dict {name: (dataset, labelKey)}
        """
        self.embeddings = embeddings
        self.modelNames = list(datasetTuples.keys())
        self.evaluateModelNames = []
        self.datasets = {}
        self.labelKeys = {}
        self.models = {}
        self.idx2Labels = {}
        
        for modelName in self.modelNames:
            dataset = datasetTuples[modelName][0]
            labelKey = datasetTuples[modelName][1]
            evaluateModel = datasetTuples[modelName][2]
            
        
            self.datasets[modelName] = dataset
            self.labelKeys[modelName] = labelKey
            self.models[modelName] = None
            self.idx2Labels[modelName] = {v: k for k, v in dataset['mappings'][labelKey].items()}
            
            if evaluateModel:
                self.evaluateModelNames.append(modelName)
            
            logging.info("--- %s ---" % modelName)
            logging.info("%d train sentences" % len(dataset['trainMatrix']))
            logging.info("%d dev sentences" % len(dataset['devMatrix']))
            logging.info("%d test sentences" % len(dataset['testMatrix']))
            
        
        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]
             
        self.casing2Idx = self.datasets[self.modelNames[0]]['mappings']['casing']
        
        self.resultsOut = None
        
        defaultParams = {'dropout': 0, 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {}, 'optimizer': 'adam',
                         'charEmbeddings': None, 'charEmbeddingsSize':30, 'charFilterSize': 30, 'charFilterLength':3, 'charLSTMSize': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams
        
        
        self.buildModel()
        
    def buildModel(self):
        
        
          
        tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], trainable=False)(tokens_input)
                
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        caseMatrix = np.identity(len(self.casing2Idx), dtype='float32')
        casing = Embedding(input_dim=caseMatrix.shape[0], output_dim=caseMatrix.shape[1], weights=[caseMatrix], trainable=False)(casing_input) 
        
        self.featureNames = ['tokens', 'casing']        
        mergeInputLayers = [tokens, casing]
        inputNodes = [tokens_input, casing_input]
        
        # :: Character Embeddings ::
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            logging.info("Pad characters to uniform length")
            self.padCharacters()
            logging.info("Words padded to %d charachters" % (self.maxCharLen))
            
            charset = None
            for dataset in self.datasets.values():
                if charset == None:
                    charset = dataset['mappings']['characters']
                else: #Ensure that the character to int mapping is equivalent
                    tmpCharset = dataset['mappings']['characters']
                    for key, value in charset.items():
                        if key not in tmpCharset or tmpCharset[key] != value:
                            logging.info("Two dataset with different characters mapping have been passed to the model")
                            assert("False")
                
            charEmbeddingsSize = self.params['charEmbeddingsSize']
            maxCharLen = self.maxCharLen
            charEmbeddings= []
            for _ in charset:
                limit = math.sqrt(3.0/charEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, charEmbeddingsSize) 
                charEmbeddings.append(vector)
                
            charEmbeddings[0] = np.zeros(charEmbeddingsSize) #Zero padding
            charEmbeddings = np.asarray(charEmbeddings)
            
            chars_input = Input(shape=(None,maxCharLen), dtype='int32', name='char_input')
            chars = TimeDistributed(Embedding(input_dim=charEmbeddings.shape[0], output_dim=charEmbeddings.shape[1],  weights=[charEmbeddings], trainable=True, mask_zero=True), name='char_emd')(chars_input)
            
            if self.params['charEmbeddings'].lower() == 'lstm': #Use LSTM for char embeddings from Lample et al., 2016
                charLSTMSize = self.params['charLSTMSize']
                chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, return_sequences=False)), name="char_lstm")(chars)
            else: #Use CNNs for character embeddings from Ma and Hovy, 2016
                charFilterSize = self.params['charFilterSize']
                charFilterLength = self.params['charFilterLength']
                chars = TimeDistributed(Convolution1D(charFilterSize, charFilterLength, border_mode='same'), name="char_cnn")(chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)
            
            mergeInputLayers.append(chars)
            inputNodes.append(chars_input)
            self.featureNames.append('characters')
            
        # :: Task Identifier :: 
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()
            
            taskID_input = Input(shape=(None,), dtype='int32', name='task_id_input')
            taskIDMatrix = np.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = Embedding(input_dim=taskIDMatrix.shape[0], output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix], trainable=False)(taskID_input) 
        
            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.featureNames.append('taskID')
        
        merged_input = merge(mergeInputLayers, mode='concat')
        
        
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:      
            if isinstance(self.params['dropout'], (list, tuple)):  
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout_W=self.params['dropout'][0], dropout_U=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)   
            else:
                """ Naive dropout """
                shared_layer = Bidirectional(LSTM(size, return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer) 
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            
            cnt += 1
            
            
        for modelName in self.modelNames:
            output = shared_layer
            
            modelDecoder = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']
            
            cnt = 1
            for decoder in modelDecoder:                            
                if decoder == 'Softmax':   
                    output = TimeDistributed(Dense(len(self.datasets[modelName]['mappings'][self.labelKeys[modelName]]), activation='softmax'), name=modelName+'_softmax')(output)
                    lossFct = 'sparse_categorical_crossentropy'
                elif decoder == 'CRF':
                    output = TimeDistributed(Dense(len(self.datasets[modelName]['mappings'][self.labelKeys[modelName]]), activation=None), name=modelName+'_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_CRF')
                    output = crf(output)          
                    lossFct = crf.sparse_loss 
                elif decoder == 'Tanh-CRF':
                    output = TimeDistributed(Dense(len(self.datasets[modelName]['mappings'][self.labelKeys[modelName]]), activation='tanh'), name=modelName+'_hidden_tanh_layer')(output)
                    crf = ChainCRF()
                    output = crf(output)          
                    lossFct = crf.sparse_loss
                elif decoder[0] == 'LSTM':            
                            
                    size = decoder[1]
                    if isinstance(self.params['dropout'], (list, tuple)): 
                        output = Bidirectional(LSTM(size, return_sequences=True, dropout_W=self.params['dropout'][0], dropout_U=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """ 
                        output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output) 
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)                    
                else:
                    assert(False) #Wrong decoder
                    
                cnt += 1
                
                
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']
            
            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']
            
            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop': 
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=0.1, **optimizerParams)
            
            
            model = Model(input=inputNodes, output=[output])
            model.compile(loss=lossFct, optimizer=opt)
            
            model.summary(line_length=200)
            logging.info(model.get_config())
            logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))
            
            self.models[modelName] = model
        


    def trainModel(self):
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:       
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:            
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch]) 
                
            
        for batch in self.minibatch_iterate_dataset():
            for modelName in self.modelNames:         
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                self.models[modelName].train_on_batch(nnInput, nnLabels)  
                
                               
            
          
    trainMiniBatchRanges = None 
    trainSentenceLengthRanges = None
    def minibatch_iterate_dataset(self, modelNames = None):
        """ Create based on sentence length mini-batches with approx. the same size. Sentences and 
        mini-batch chunks are shuffled and used to the train the model """
        
        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            self.trainSentenceLengthRanges = {}
            self.trainMiniBatchRanges = {}            
            for modelName in self.modelNames:
                trainData = self.datasets[modelName]['trainMatrix']
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by sentence length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])            
                idxStart = 0
                
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])
                    
                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx
                    
                    oldSentLength = sentLength
                
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))
                
                
                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]
                    
                    
                    bins = int(math.ceil(rangeLen/float(self.miniBatchSize)))
                    binSize = int(math.ceil(rangeLen / float(bins)))
                    
                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                      
                self.trainSentenceLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges
                
        if modelNames == None:
            modelNames = self.modelNames
            
        #Shuffle training data
        for modelName in modelNames:      
            #1. Shuffle sentences that have the same length
            x = self.datasets[modelName]['trainMatrix']
            for dataRange in self.trainSentenceLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
               
            #2. Shuffle the order of the mini batch ranges       
            random.shuffle(self.trainMiniBatchRanges[modelName])
     
        
        #Iterate over the mini batch ranges
        if self.mainModelName != None:
            rangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
            logging.info("Main Model:", self.mainModelName)
            logging.info("Main Model range length", rangeLength)
        else:
            rangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])
        
        
        
        batches = {}
        for idx in range(rangeLength):
            batches.clear()
            
            for modelName in modelNames:   
                trainMatrix = self.datasets[modelName]['trainMatrix']
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])] 
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                
                
                batches[modelName] = [labels]
                
                for featureName in self.featureNames:                
                    inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                    batches[modelName].append(inputData)
            
            yield batches   
            
          
                    
            
                
             
    def online_iterate_dataset(self):
        """ Iterate dataset sentence-by-sentence (online training) """
        
        ranges = {}
        
        for modelName in self.modelNames: 
            rndRange = list(range(len(self.datasets[modelName]['trainMatrix'])))
            random.shuffle(rndRange)
            ranges[modelName] = rndRange
        
        rangeLength = max([len(ranges[modelName]) for modelName in self.modelNames])
        
        for idx in range(rangeLength):
            batches = {}
            
            for modelName in self.modelNames:  
                modelIdx = ranges[modelName][idx % len(ranges[modelName])]        
                tokens = self.datasets[modelName]['trainMatrix'][modelIdx]['tokens']
                casing = self.datasets[modelName]['trainMatrix'][modelIdx]['casing']
                labelKey = self.labelKeys[modelName]
                labels = np.expand_dims([self.datasets[modelName]['trainMatrix'][modelIdx][labelKey]], -1)          
                batches[modelName] = [labels, np.asarray([tokens]), np.asarray([casing])]  
            
            yield batches   
        
    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsOut = open(resultsFilepath, 'w')
        else:
            self.resultsOut = None 
        
    def evaluate(self, epochs):       
      
        total_train_time = 0
        max_dev_score = {modelName:0 for modelName in self.models.keys()}
        max_test_score = {modelName:0 for modelName in self.models.keys()}
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time() 
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
            start_time = time.time() 
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
                dev_score, test_score = self.computeScore(modelName, self.datasets[modelName]['devMatrix'], self.datasets[modelName]['testMatrix'])
         
                
                if dev_score > max_dev_score[modelName]:
                    max_dev_score[modelName] = dev_score
                    max_test_score[modelName] = test_score
                    no_improvement_since = 0
                else:
                    no_improvement_since += 1
                    
                    
                if self.resultsOut != None:
                    self.resultsOut.write("\t".join(map(str, [epoch+1, modelName, dev_score, test_score, max_dev_score[modelName], max_test_score[modelName]])))
                    self.resultsOut.write("\n")
                    self.resultsOut.flush()
                
                logging.info("Max: %.4f dev; %.4f test" % (max_dev_score[modelName], max_test_score[modelName]))
                logging.info("")
                
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.earlyStopping > 0 and no_improvement_since >= self.earlyStopping:
                logging.info("!!! Early stopping, no improvement after ",no_improvement_since," epochs !!!")
                break
            
            
            
            
    
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths

    def predictLabels(self, model, sentences):

        predLabels = [None]*len(sentences)
        
        sentenceLengths = self.getSentenceLengths(sentences)
        
        for indices in sentenceLengths.values():   
            nnInput = []                  
            for featureName in self.featureNames:                
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                nnInput.append(inputData)
            
            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) #Predict classes            
           
            
            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]    
                predIdx += 1   
        
        return predLabels
    
   
    def computeScore(self, modelName, devMatrix, testMatrix):
        if self.labelKeys[modelName].endswith('_BIO') or self.labelKeys[modelName].endswith('_IOBES') or self.labelKeys[modelName].endswith('_IOB'):
            return self.computeF1Scores(modelName, devMatrix, testMatrix)
        else:
            return self.computeAccScores(modelName, devMatrix, testMatrix)   

    def computeF1Scores(self, modelName, devMatrix, testMatrix):
        #train_pre, train_rec, train_f1 = self.computeF1(modelName, self.datasets[modelName]['trainMatrix'])
        #print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (train_pre, train_rec, train_f1)
        
        dev_pre, dev_rec, dev_f1 = self.computeF1(modelName, devMatrix)
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))
        
        test_pre, test_rec, test_f1 = self.computeF1(modelName, testMatrix)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1
    
    def computeAccScores(self, modelName, devMatrix, testMatrix):
        dev_acc = self.computeAcc(modelName, devMatrix)
        test_acc = self.computeAcc(modelName, testMatrix)
        
        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        
        return dev_acc, test_acc   
        
        
    def computeF1(self, modelName, sentences):
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]
        
        correctLabels = [sentences[idx][labelKey] for idx in range(len(sentences))]
        predLabels = self.predictLabels(model, sentences) 

        labelKey = self.labelKeys[modelName]
        encodingScheme = labelKey[labelKey.index('_')+1:]
        
        pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'O', encodingScheme)
        pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'B', encodingScheme)
        
        if f1_b > f1:
            logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
            pre, rec, f1 = pre_b, rec_b, f1_b
        
        return pre, rec, f1
    
    def computeAcc(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        predLabels = self.predictLabels(self.models[modelName], sentences) 
        
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1

  
        return numCorrLabels/float(numLabels)
    
    def padCharacters(self):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = 0
        for dataset in self.datasets.values():
            for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:            
                for sentence in data:
                    for token in sentence['characters']:
                        maxCharLen = max(maxCharLen, len(token))
          
        for dataset in self.datasets.values():   
            for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:       
                #Pad each other word with zeros
                for sentenceIdx in range(len(data)):
                    for tokenIdx in range(len(data[sentenceIdx]['characters'])):
                        token = data[sentenceIdx]['characters'][tokenIdx]
                        data[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')
    
        self.maxCharLen = maxCharLen
        
    def addTaskIdentifier(self):
        """ Adds an identifier to every token, which identifies the task the token stems from """
        taskID = 0
        for modelName in self.modelNames:
            dataset = self.datasets[modelName]
            for dataName in ['trainMatrix', 'devMatrix', 'testMatrix']:            
                for sentenceIdx in range(len(dataset[dataName])):
                    dataset[dataName][sentenceIdx]['taskID'] = [taskID] * len(dataset[dataName][sentenceIdx]['tokens'])
            
            taskID += 1
                    