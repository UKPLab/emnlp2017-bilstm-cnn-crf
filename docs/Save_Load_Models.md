# Saving and loading of models
Models trained with the BiLSTM-CNN-CRF architecture and be stored on disk and later be loaded to tag new data.

## Saving
The path to store the models is found in the `modelSavePath` variable (for example `Train_POS.py`):
```
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
```

This variable specifies that the model should be stored in the `models`-folder. There exists some variables which are replaced during the runtime with specific values:
- `[ModelName]`: replaced by the name of the dataset, for example, `unidep_pos` for `Train_POS.py`
- `[DevScore]`: Score (accuracy or F1-score) on the development set
- `[TestScore]`: Score (accuracy or F1-score) on the test set
- `[Epoch]`: Trainings epoch


A new model is only stored if the performance on the development set increases.

## Loading
Models can be loaded by calling the static method `BiLSTM.loadModel()`. See `RunModel.py` for an example:
```
lstmModel = BiLSTM.loadModel(modelPath)
#...
tags = lstmModel.tagSentences(dataMatrix)
```

Input data (in the right format) can be tagged using the `tagSentences()` method. Note that the passed `dataMatrix` must have all input values that were used during the training. Per-default, `tokens`, `casing` and maybe `characters` are used as input features. You might need to extend your input data by this features. Only passing `tokens` to the model is usually not sufficient. 
