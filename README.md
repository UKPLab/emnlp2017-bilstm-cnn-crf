# BiLSTM-CNN-CRF Implementation for Sequence Tagging

In the following repository you can find an BiLSTM-CRF implementation used for Sequence Tagging, e.g. POS-tagging, Chunking, or Named Entity Recognition. The implementation is based on Keras 1.x and can be run with Theano (0.9.0) or Tensorflow (0.12.1) as backend.

The architecture is described in our papers [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://arxiv.org/abs/1707.09861) and [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/abs/1707.06799).


The hyperparameters of this network can be easily configured, so that you can re-run the proposed system by [Huang et al., Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991), [Ma and Hovy, End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) and [Lample et al, Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360).

The implementation was optimized for **performance** using a smart shuffeling of the trainings data to group sentences with same length together. This increases the training speed by multiple factors in comparison to the implementations by Ma or Lample.

The training of the network is simple and can easily be extended to new datasets and languages. For example, see [Train_POS.py](Train_POS.py).

Pretrained models can be stored and loaded for inference. Simply execute `python RunModel.py models/modelname.h5 input.txt`. Pretrained-models for some sequence tagging task using this LSTM-CRF implementations are provided in [Pretrained Models](Pretrained_Models.md).

This implementation can be used for **Multi-Task Learning**, i.e. learning simultanously several task with non-overlapping datasets. The file [Train_MultiTask.py](Train_MultiTask.py) depicts an example, how the LSTM-CRF network can be used to learn POS-tagging and Chunking simultaneously. The number of tasks is not limited. Tasks can be supervised at the same level or at different output level, for example, to re-implement the approach by [Sogaard and Goldberg, Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038).
 
 

# Citation
If you find the implementation useful, please cite the following paper: [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://arxiv.org/abs/1707.09861)

```
@InProceedings{Reimers:2017:EMNLP,
  author    = {Reimers, Nils, and Gurevych, Iryna},
  title     = {{Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging}},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {09},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  pages     = {338--348},
  url       = {http://aclweb.org/anthology/D17-1035}
}
``` 

> **Abstract:** In this paper we show that reporting a single performance score is insufficient to compare non-deterministic approaches. We demonstrate this for common sequence tagging tasks that the seed value for the random number generator can result in statistically significant (p < 10^{-4}) differences for state-of-the-art systems. For two recent systems for NER, we observe an absolute difference of one percentage point F1-score depending on the selected seed value, making these systems perceived either as state-of-the-art or mediocre. Instead of publishing and reporting single performance scores, we propose to compare score distributions based on multiple executions. Based on the evaluation of 50.000 LSTM-networks for five sequence tagging tasks, we present network architectures that perform superior as well as produce results with higher stability on unseen data.


Contact person: Nils Reimers, reimers@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

# Setup
In order to run the code, you need either Python 2.7 or Python 3.6, Keras 1.2.x and as Backend either Theano 0.9.0 or TensorFlow 0.12.1. Note, at the moment the code **cannot** be run with Keras 2.x or Tensorflow 1.x!

If you want to use the character based word representations (**charEmbeddings**), you have to use the **Theano backend**. You can change this for Keras in the home folder in the file: `.keras/keras.json` by setting the option `backend` to `theano`. Another option is to set an environment variable:
```
export KERAS_BACKEND=theano
```

**Note:** The CNN for character based representations doesn't work with Tensorflow. You will get the error message:
```
TypeError: unsupported operand type(s) for *: 'IndexedSlices' and 'int'
```

Solution: Use Theano as backend or don't use character based representations using CNN (parameter: charEmbeddings). 

It would be highly appriated if someone could port the code (the CRF-layer) to Keras 2.x and Tensorflow 1.x.

## Setup with virtual environment

Setup a Python virtual environment (optional):
``` 
virtualenv .env
source .env/bin/activate
```

Install the requirements:
```
.env/bin/pip install -r requirements.txt
```

If everything works well, you can run `python Train_POS.py` to train a deep POS-tagger for the POS-tagset from universal dependencies.

## Setup with docker
See the [docker-folder](docker/) for more information how to run these scripts in a docker container.


# Training
Training new models is simple. Look at `Train_POS.py` and `Train_Chunking.py` for examples.

Place new datasets in the folder `data`. The system expects three files `train.txt`, `dev.txt` and `test.txt` in a CoNLL format. I.e. each token is in a new line, different columns are seperated by a white space (either a space or a tab). Sentences are seperated by an empty line.

For an example look at `data/conll2000_chunking/train.txt`. Files with multiple columns, like `data/unidep_pos/train.txt` are no problem, as we will specify later which columns should be used for training.

To train a LSTM-network, you must specify the following lines of code (`Train_POS.py`):
```
datasetName = 'unidep_pos'
dataColumns = {1:'tokens', 3:'POS'} #Tab separated columns, column 1 contains the token, 3 the universal POS tag
labelKey = 'POS'

embeddingsPath = 'levy_deps.words' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
```

`datasetName` defines the name of the dataset, here it will use the data in the folder `data/unidep_pos`. `dataColumns` specifies the columns that should be read from the CoNLL file, in this case the first column and the third column should be read. The counting starts at 0. The first column contains the tokens, and the third column the POS-tag. Note, that we must always specify a 'tokens' column. The other columns can be named arbitrarily.

`labelKey` will specify which column should serve as label, in this case we want to perform POS-tagging. The name must match with the name specified in the dictionary `dataColumns`.

`embeddingsPath` contains the path to pre-trained word embeddings. The format for this must be text-based, i.e. each line contains the embedding for a word and the first column in that line is the word, followed by the dense vector. Our script will automatically download the embeddings by Levy et al. if they are not present.


If you want to perform chunking instead of POS-tagging, simple change the first lines (`Train_Chunking.py`):
```
datasetName = 'conll2000_chunking'
dataColumns = {0:'tokens', 1:'POS', 2:'chunk_BIO'} #Tab separated columns, column 0 contains the token, 1 the POS, 2 the chunk information using a BIO encoding
labelKey = 'chunk_BIO'
```

**Note:** By appending a *_BIO* to a column name, we indicate that this column is BIO encoded. The system will then compute the F1-score instead of the accuracy. 

# Running a stored model
If enabled during the trainings process, models are stored to the 'models' folder. Those models can be loaded and be used to tag new data. An example is implemented in `RunModel.py`:

```
python RunModel.py models/modelname.h5 input.txt
```

This script will read the model `models/modelname.h5` as well as the text file `input.txt`. The text will be splitted into sentences and tokenized using NLTK. The tagged output will be written in a CoNLL format to standard out.


# Multi-Task-Learning
The class `neuralnets/MultiTaskLSTM.py` implements a Multi-Task-Learning setup using LSTM. The code and parameters are similar to the Single-Task setup.

The file `Train_MultiTask.py` contains an example how to run the code. There, we define which datasets should be used:
```
posName = 'unidep_pos'
posColumns = {1:'tokens', 3:'POS'}

chunkingName = 'conll2000_chunking'
chunkingColumns = {0:'tokens', 1:'POS', 2:'chunk_BIO'}


datasetFiles = [
        (posName, posColumns),
        (chunkingName, chunkingColumns)
    ]

#....

datasetTuples = {
    'POS': (posData, 'POS', True),
    'Chunking': (chunkingData, 'chunk_BIO', True)
    }
```

As before, we define the dataset names with the column names and store these information in the `datasetFiles` array. The dictionary `datasetTuples` contains the preprocessed datasets (`posData` and `chunkingData`), the column we like to use as label (`POS` and `chunk_BIO`). The boolean parameter defines whether this dataset should be evaluated. If it is set to `False`, no performance scores will be printed for this dataset.


# LSTM-Hyperparameters
The parameters in the LSTM-CRF network can be configured by passing a parameter-dictionary to the BiLSTM-constructor: `BiLSTM(params)`.

The following parameters exists:
* **miniBatchSize**: Size (Nr. of sentences) for mini-batch training. Default value: 32
* **dropout**: Set to 0, for no dropout. For naive dropout, set it to a real value between 0 and 1. For variational dropout, set it to a two-dimensional tuple or list, with the first entry corresponding to output dropout and the second entry to the recurrent dropout. Default value: [0.25, 0.25]
* **classifier**: Set to `Softmax` to use a softmax classifier or to `CRF` to use a CRF-classifier as the last layer of the network. Default value: `Softmax`
* **LSTM-Size**: List of integers with the number of recurrent units for the stacked LSTM-network. The list [100,75,50] would create 3 stacked BiLSTM-layers with 100, 75, and 50 recurrent units. Default value: [100]
* **optimizer**: Available optimizers: SGD, AdaGrad, AdaDelta, RMSProp, Adam, Nadam. Default value: `nadam`
* **earlyStopping**: Early stoppig after certain number of epochs, if no improvement on the development set was achieved. Default value: 5
* **addFeatureDimensions**: Dimension for additional features, that are passed to the network. Default value: 10
* **charEmbeddings**: Available options: [None, 'CNN', 'LSTM']. If set to `None`, no character-based representations will be used. With `CNN`, the approach by [Ma & Hovy](https://arxiv.org/abs/1603.01354) using a CNN will be used. With `LSTM`, an LSTM network will be used to derive the character-based representation ([Lample et al.](https://arxiv.org/abs/1603.01360)). Default value: `None`. Note, **charEmbeddings** does only work with Theano as backend.
	* **charEmbeddingsSize**: The dimension for characters, if the character-based representation is enabled. Default value: 30
	* **charFilterSize**: If the CNN approach is used, this parameters defined the filter size, i.e. the output dimension of the convolution. Default: 30
	* **charFilterLength**: If the CNN approach is used, this parameters defines the filter length. Default: 3
	* **charLSTMSize**: If the LSTM approach is used, this parameters defines the size of the recurrent units. Default: 25
* **clipvalue**: If non-zero, the gradient will be clipped to this value. Default: 0
* **clipnorm**: If non-zero, the norm of the gradient will be normalized to this value. Default: 1

For the MutliTaskLSTM.py-network, the following additional parameter exists:
* **customClassifier**: A dictionary, that maps each dataset an individual classifier. For example, the POS tag could use a Softmax-classifier, while the Chunking dataset is trained with a CRF-classifier.


# Acknowledgments
This code uses the CRF-Implementation of [Philipp Gross](https://github.com/phipleg) from the Keras Pull Request [#4621](https://github.com/fchollet/keras/pull/4621). Thank you for contributing this to the community.
