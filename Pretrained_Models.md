# Pretrained Sequence Tagging Models
In the following some pre-trained models are provided for different common sequence tagging task. These models can be used by executing:
```
python RunModel.py modelname.h5 input.txt
```

For the English models, we used the word embeddings by [Levy et al.](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/). For the German files, we used the word embeddings by [Reimers et al.](https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)


## POS
We trained POS-tagger on the [Universal Dependencies]((http://universaldependencies.org/)) v1.3 dataset:
Trained on universal dependencies v1.3 Englisch: 

| Language | Development (Accuracy) | Test (Accuracy) |
|----------|:-----------:|:----:|
|[English (UD)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_UD_POS.h5) | 95.47% | 95.55% |
|[German (UD)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/DE_UD_POS.h5) | 94.86% | 93.99% | 

Further, we trained models on the Wall Street Journal:

| Language | Development (Accuracy) | Test (Accuracy) |
|----------|:-----------:|:----:|
|[English (WSJ)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_WSJ_POS.h5) | 97.18% | 97.21% |

The depicted performance is accuracy.


## Chunking
Trained on [CoNLL 2000 Chunking dataset](http://www.cnts.ua.ac.be/conll2000/chunking/). Performance is F1-score.

| Language | Development (F1) | Test(F1) |
|----------|:-----------:|:----:|
|[English (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_Chunking.h5) | 95.40% | 94.70% |


## NER
Trained on [CoNLL 2003](http://www.cnts.ua.ac.be/conll2003/ner/) and [GermEval 2014](https://sites.google.com/site/germeval2014ner/)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_NER.h5) | 94.29% | 90.87% | 
|[German (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/DE_NER_CoNLL.h5) | 80.80% | 77.49% | 
|[German (GermEval 2014)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/DE_NER_GermEval.h5) | 80.85% | 80.00% |


## Entities
Trained on ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_Entities.h5) | 82.46% | 85.78% | 


## Events
Trained on TempEval3 (https://www.cs.york.ac.uk/semeval-2013/task1/)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/EN_Events.h5) |- | 82.28% | 


## Parameters
In the following are some parameters & configurations listed for the pretrained models.

```
English NER:
Glove 6B 100 embeddings with params = {'dropout': [0.25, 0.25], 'classifier': 'CRF', 'LSTM-Size': [100,75], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 32}

German NER (CoNLL 2003 and GermEval 2014):
Reimers et al., 2014, GermEval embeddings with params = {'dropout': [0.25, 0.25], 'classifier': 'CRF', 'LSTM-Size': [100,75], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 32}

Entities:
Glove 6B 100 embeddings, params = {'dropout': [0.25, 0.25], 'classifier': 'CRF', 'LSTM-Size': [100,75], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 32}
```


