# BiLSTM-CNN-CRF Implementation for Sequence Tagging

This repository contains a BiLSTM-CRF implementation that used for NLP Sequence Tagging (for example POS-tagging, Chunking, or Named Entity Recognition). The implementation is based on Keras 2.2.0 and can be run with Tensorflow 1.8.0 as backend. It was optimized for Python 3.5 / 3.6. It does **not work** with Python 2.7.

The architecture is described in our papers:
 - [Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging](https://arxiv.org/abs/1707.09861) 
 - [Why Comparing Single Performance Scores Does Not Allow to Draw Conclusions About Machine Learning Approaches](https://arxiv.org/abs/1803.09578)
 - [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks](https://arxiv.org/abs/1707.06799).

The implementation is **highly configurable**, so you can tune the different hyperparameters easily. You can use it for **Single Task Learning** as well as different options for **Multi-Task Learning**. You can also use it for **Multilingual Learning** by using multilingual word embeddings.

This code can be used to run the systems proposed in the following papers:
* [Huang et al., Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) - You can choose between a softmax and a CRF classifier
* [Ma and Hovy, End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) - Character based word representations using CNNs is achieved by setting the parameter `charEmbeddings` to `CNN`
* [Lample et al, Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) - Character based word representations using LSTMs is achieved by setting the parameter `charEmbeddings` to `LSTM`
* [Sogard, Goldberg: Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038) - Train multiple task and supervise them on different levels.


The implementation was optimized for **speed**: By grouping sentences with the same lengths together, this implementation is multiple factors faster than the systems by Ma et al. or Lample et al.

The training of the network is simple and the neural network can easily be trained on new datasets. For an example, see [Train_POS.py](Train_POS.py).


Trained models can be **stored** and **loaded** for inference. Simply execute `python RunModel.py models/modelname.h5 input.txt`. Pretrained-models for some sequence tagging task using this LSTM-CRF implementations are provided in [Pretrained Models](docs/Pretrained_Models.md).

This implementation can be used for **Multi-Task Learning**, i.e. learning simultanously several task with non-overlapping datasets. The file [Train_MultiTask.py](Train_MultiTask.py) depicts an example, how the LSTM-CRF network can be used to learn POS-tagging and Chunking simultaneously. The number of tasks are not limited. Tasks can be supervised at the same level or at different output level.
 
 # Implementation with ELMo representations
The repository [elmo-bilstm-cnn-crf](https://github.com/UKPLab/elmo-bilstm-cnn-crf) contains an extension of this architecture to work with the ELMo representations from AllenNLP (from the Paper: Peters et al., 2018, [Deep contextualized word representations](http://arxiv.org/abs/1802.05365)). ELMo representations are computationally expensive to compute, but they usually improve the performance by about 1-5 percentage points F1-measure.

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



Contact person: Nils Reimers, reimers@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/ https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

# Setup
In order to run the code, I recommend Python 3.5 or higher. The code is based on Keras 2.2.0 and as backend I recommend Tensorflow 1.8.0. I cannot ensure that the code works with different versions for Keras / Tensorflow or with different backends for Keras. The code **does not** work with Python 2.7.



## Setup with virtual environment (Python 3)

Setup a Python virtual environment (optional):
``` 
virtualenv --system-site-packages -p python3 env
source env/bin/activate
```

Install the requirements:
```
env/bin/pip3 install -r requirements.txt
```

If everything works well, you can run `python3 Train_POS.py` to train a deep POS-tagger for the POS-tagset from universal dependencies.

## Setup with docker
See the [docker-folder](docker/) for more information how to run these scripts in a docker container.



# Running a stored model
If enabled during the trainings process, models are stored to the 'models' folder. Those models can be loaded and be used to tag new data. An example is implemented in `RunModel.py`:

```
python RunModel.py models/modelname.h5 input.txt
```

This script will read the model `models/modelname.h5` as well as the text file `input.txt`. The text will be splitted into sentences and tokenized using NLTK. The tagged output will be written in a CoNLL format to standard out.


# Training
See `Train_POS.py` for a simple example how to train the model. More details can be found in [docs/Training.md](docs/Training.md).

For training, you specify the datasets you want to train on:
```
datasets = {
    'unidep_pos':                            #Name of the dataset
        {'columns': {1:'tokens', 3:'POS'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'POS',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
```

And you specify the pass to a pre-trained word embedding file:
```
embeddingsPath = 'komninos_english_embeddings.gz'
```

The `util.preprocessing.py` fle contains some methods to read your dataset (from the `data` folder) and to store a pickle file in the `pkl` folder. 

You can then train the network in the following way:
```
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}
model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)
```


# Multi-Task-Learning
Multi-Task Learning can simply be done by specifying multiple datasets (`Train_MultiTask.py`)
```
datasets = {
    'unidep_pos':
        {'columns': {1:'tokens', 3:'POS'},
         'label': 'POS',
         'evaluate': True,
         'commentSymbol': None},
    'conll2000_chunking':
        {'columns': {0:'tokens', 2:'chunk_BIO'},
         'label': 'chunk_BIO',
         'evaluate': True,
         'commentSymbol': None},
}
```

Here, the networks trains jointly for the POS-task (`unidep_pos`) and for the chunking task (`conll2000_chunking`).

You can also train task on different levels. For details, see [docs/Training_MultiTask.md](docs/Training_MultiTask.md).


# LSTM-Hyperparameters
The parameters in the LSTM-CRF network can be configured by passing a parameter-dictionary to the BiLSTM-constructor: `BiLSTM(params)`.

The following parameters exists:

* **dropout**: Set to 0, for no dropout. For naive dropout, set it to a real value between 0 and 1. For variational dropout, set it to a two-dimensional tuple or list, with the first entry corresponding to output dropout and the second entry to the recurrent dropout. Default value: [0.5, 0.5]
* **classifier**: Set to `Softmax` to use a softmax classifier or to `CRF` to use a CRF-classifier as the last layer of the network. Default value: `Softmax`
* **LSTM-Size**: List of integers with the number of recurrent units for the stacked LSTM-network. The list [100,75,50] would create 3 stacked BiLSTM-layers with 100, 75, and 50 recurrent units. Default value: [100]
* **optimizer**: Available optimizers: SGD, AdaGrad, AdaDelta, RMSProp, Adam, Nadam. Default value: `nadam`
* **earlyStopping**: Early stoppig after certain number of epochs, if no improvement on the development set was achieved. Default value: 5
* **miniBatchSize**: Size (Nr. of sentences) for mini-batch training. Default value: 32
* **addFeatureDimensions**: Dimension for additional features, that are passed to the network. Default value: 10
* **charEmbeddings**: Available options: [None, 'CNN', 'LSTM']. If set to `None`, no character-based representations will be used. With `CNN`, the approach by [Ma & Hovy](https://arxiv.org/abs/1603.01354) using a CNN will be used. With `LSTM`, an LSTM network will be used to derive the character-based representation ([Lample et al.](https://arxiv.org/abs/1603.01360)). Default value: `None`.
	* **charEmbeddingsSize**: The dimension for characters, if the character-based representation is enabled. Default value: 30
	* **charFilterSize**: If the CNN approach is used, this parameters defined the filter size, i.e. the output dimension of the convolution. Default: 30
	* **charFilterLength**: If the CNN approach is used, this parameters defines the filter length. Default: 3
	* **charLSTMSize**: If the LSTM approach is used, this parameters defines the size of the recurrent units. Default: 25
* **clipvalue**: If non-zero, the gradient will be clipped to this value. Default: 0
* **clipnorm**: If non-zero, the norm of the gradient will be normalized to this value. Default: 1
* **featureNames**: Which features the network should use. You can specify additinal features that are used, for example, this could be POS-tags. See `Train_Custom_Features.py` for an example. Default: ['tokens', 'casing']
* **addFeatureDimensions**: Size of the embedding matrix for all features except `tokens'. Default: 10

For multi-task learning scenarios, the following additional parameter exists:
* **customClassifier**: A dictionary, that maps each dataset an individual classifier. For example, the POS tag could use a Softmax-classifier, while the Chunking dataset is trained with a CRF-classifier.
* **useTaskIdentifier**: Including a task-ID as an input feature. Default: False 

# Documentation

- [docs/Training.md](docs/Training.md): How to train simple, single-task architectures
- [docs/Training_MultiTask.md](docs/Training_MultiTask.md): How to use the architecture for Multi-Task Learning
- [docs/Save_Load_Models.md](docs/Save_Load_Models.md): How to save & load models
- [docs/Pretrained_Models.md](docs/Pretrained_Models.md): Several pretraiend models that can be downloaded for common sequence tagging tasks.

# Acknowledgments
This code uses the CRF-Implementation of [Philipp Gross](https://github.com/phipleg) from the Keras Pull Request [#4621](https://github.com/fchollet/keras/pull/4621). Thank you for contributing this to the community.
