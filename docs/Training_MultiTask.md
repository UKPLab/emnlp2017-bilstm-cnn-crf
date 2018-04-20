# Multi-Task-Learning BiLSTM-CNN-CRF
The network can be used to perform multi-task learning.

See `Train_MultiTask.py` for an example.

You specify the different datasets using the `datasets` variable:
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
Here we specify to train on the two datasets `unidep_pos` and `conll2000_chunking`. The format of this dictionary is as described in [Training.md](Training.md).

**Note:** If columns in two datasets have the same name, then they are mapped to the same embeddings space. I.e., if columns have the same names, ensure that they have the same content. It would **not** be advisable to name your label column `label` in all datasets. 

The variable `evaluate` can be set to `False` for Multi-Task Learning setups: Then, the networks trains on this dataset, however, it is not evaluated on the development and test set for this dataset. This can be useful for example if you have a main task and some auxiliary tasks.

 
 ## Multi-Task Specific Hyperparameters
 You can use the hyperparameter `customClassifier` to specify custom classifiers for all your different datasets.
 
 See `Train_MultiTask_Different_Levels.py` for an example. In that file, we pass the following parameters to the network:
```
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),
          'customClassifier': {'unidep_pos': ['Softmax'], 'conll2000_chunking': [('LSTM', 50), 'CRF']}}
```

For our POS-task (`unidep_pos`) we specify a simple softmax classifier. However, the chunking task (`conll2000_chunking`) has a task-specific LSTM-layer with 50 recurrent units followed by a CRF-classifier.

Using this technique, you can implement the technique that is described in this paper: [Sogard, Goldberg: Deep multi-task learning with low level tasks supervised at lower layers](http://anthology.aclweb.org/P16-2038)