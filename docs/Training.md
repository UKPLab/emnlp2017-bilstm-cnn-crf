# Training
Training new models is simple. Look at `Train_POS.py` and `Train_Chunking.py` for examples.

Place new datasets in the folder `data`. The system expects three files `train.txt`, `dev.txt` and `test.txt` in a CoNLL format. I.e. each token is in a new line, different columns are seperated by a white space (either a space or a tab). Sentences are seperated by an empty line.

For an example look at `data/conll2000_chunking/train.txt`. Files with multiple columns, like `data/unidep_pos/train.txt` are no problem, as we will specify later which columns should be used for training.


## Specifying datasets
The dataset is specified in the `datasets` variable (`Train_POS.py`):
```
datasets = {
    'unidep_pos':                            #Name of the dataset
        {'columns': {1:'tokens', 3:'POS'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'POS',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
```

`unidep_pos` defines the name of the dataset. The code looks for the `train.txt`/`dev.txt`/`test.txt` in the folder `data/unidep_pos`. If your data folder has a different name, you must change the value of `unidep_pos`.

`columns` specifies the columns that should be read from the CoNLL file, in this case the column 1 and the column 3 should be read. The counting starts at 0. Column 1 contains the tokens, and the column 3 the POS-tag. Note, that we must always specify a 'tokens' column. The other columns can be named arbitrarily. If your data format changes, you must change the `columns` value.

`label` will specify which column should serve as label, in this case we want to predict the `POS` column. The name must match with the name specified in the dictionary `dataColumns`.

`evaluate` defines if we want to evaluate on this task. Must be true for single-task learning.

`commentSymbol` - Lines starting with `commentSymbol` will be skipped when the data files are read. We set it here to `None` as our input data does not have comment lines.

## Word Embeddings
The path to pre-trained word embeddings must be specified in the variable `embeddingsPath`:
```
embeddingsPath = 'komninos_english_embeddings.gz'
```

The format for the pre-trained embeddings file must be text-based, i.e. each line contains the embedding for a word. The first column in that line is the word, followed by the dense vector. Our script will automatically download the embeddings `komninos_english_embeddings.gz` if they are not present in the current folder. The data can be compressed using `.gz`.

## Training the network
The network is trained using the following lines of code (`Train_POS.py`):
```
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}

model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/unidep_pos_results.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
model.fit(epochs=25)
```

`params` defines the hyperparameters of the network. See [Hyperparameters.md](Hyperparameters.md) for more details.

`model.modelSavePath` defines the path where the trained models should be stored. `[ModelName]` is replaced by the name of your dataset, `[DevScore]` with the score on the development set, `[TestScore]` with the score on the test set and `[Epoch]` is replaced by the epoch.

## Storing performance Scores
By calling the `model.storeResults()` we specify the path where the performance scores during training should be stored. The file contains for each training epoch a line that contains the following information:
- epoch
- dataset name
- Performance on development set
- Performance on test set
- Highest development set performance so far
- Test performance for epoch with highest development score



## Training BIO-Encoded Labels
If you want to perform chunking instead of POS-tagging, simple change the `datasets` variable (`Train_Chunking.py`):
```
datasets = {
    'conll2000_chunking':                                   #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 2 contains POS and column 2 contains chunk information using BIO encoding
         'label': 'chunk_BIO',                                #Which column we like to predict
         'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}
```

**Note:** By appending a *_BIO* to a column name, we indicate that this column is BIO encoded. The system will then compute the F1-score instead of the accuracy. 
