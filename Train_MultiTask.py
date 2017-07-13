from __future__ import print_function
import os
import logging
import sys
from neuralnets.MultiTaskLSTM import MultiTaskLSTM
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

posName = 'unidep_pos'
posColumns = {1:'tokens', 3:'POS'}

chunkingName = 'conll2000_chunking'
chunkingColumns = {0:'tokens', 1:'POS', 2:'chunk_BIO'}


datasetFiles = [
        (posName, posColumns),
        (chunkingName, chunkingColumns)
    ]

embeddingsPath = 'levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasetFiles)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)


datasetTuples = {
    'POS': (datasets[posName], 'POS', True),
    'Chunking': (datasets[chunkingName], 'chunk_BIO', True)    
    }


params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25), 'charEmbeddings': False}


model = MultiTaskLSTM(embeddings, datasetTuples, params=params)
model.evaluate(25)



