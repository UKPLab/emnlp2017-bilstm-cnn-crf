from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




######################################################
#
# Data preprocessing
#
######################################################


# :: Train / Dev / Test-Files ::
datasetName = 'GermEval'
dataColumns = {1:'tokens', 2:'NER_BIO'} #Tab separated columns, column 1 contains the token, 2 the NER using BIO-encoding
labelKey = 'NER_BIO'

embeddingsPath = '2014_tudarmstadt_german_50mincount.vocab.gz' #Word embeddings by Reimers et al: https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/

#Parameters of the network
params = {'dropout': [0.25, 0.25], 'classifier': 'CRF', 'LSTM-Size': [100,75], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 32}



frequencyThresholdUnknownTokens = 50 #If a token that is not in the pre-trained embeddings file appears at least 50 times in the train.txt, then a new embedding is generated for this word



datasetFiles = [
        (datasetName, dataColumns),
    ]


# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasetFiles)




######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]



print("Dataset:", datasetName)
print(data['mappings'].keys())
print("Label key: ", labelKey)
print("Train Sentences:", len(data['trainMatrix']))
print("Dev Sentences:", len(data['devMatrix']))
print("Test Sentences:", len(data['testMatrix']))


model = BiLSTM(params)
model.setMappings(embeddings, data['mappings'])
model.setTrainDataset(data, labelKey)
model.verboseBuild = True
model.modelSavePath = "models/%s/%s/[DevScore]_[TestScore]_[Epoch].h5" % (datasetName, labelKey) #Enable this line to save the model to the disk
model.evaluate(50)


