# Pretrained Sequence Tagging Models
In the following some pre-trained models are provided for different common sequence tagging tasks. These models can be used by executing:
```
python RunModel.py modelname.h5 input.txt
```

For the English models, we used the word embeddings by [Komninos et al.](https://www.cs.york.ac.uk/nlp/extvec/). For the German files, we used the word embeddings by [Reimers et al.](https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)


## POS
We trained POS-tagger on the [Universal Dependencies]((http://universaldependencies.org/)) v1.3 dataset:
Trained on universal dependencies v1.3 Englisch: 

| Language | Development (Accuracy) | Test (Accuracy) |
|----------|:-----------:|:----:|
|[English (UD)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_UD_POS.h5) | 95.58% | 95.58% |
|[German (UD)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/DE_UD_POS.h5) | 94.50% | 93.88% | 

Further, we trained models on the Wall Street Journal:

| Language | Development (Accuracy) | Test (Accuracy) |
|----------|:-----------:|:----:|
|[English (WSJ)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_WSJ_POS.h5) | 97.33% | 97.39% |

The depicted performance is accuracy.


## Chunking
Trained on [CoNLL 2000 Chunking dataset](http://www.cnts.ua.ac.be/conll2000/chunking/). Performance is F1-score.

| Language | Development (F1) | Test(F1) |
|----------|:-----------:|:----:|
|[English (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_Chunking.h5) | 95.30% | 94.71% |


## NER
Trained on [CoNLL 2003](http://www.cnts.ua.ac.be/conll2003/ner/) and [GermEval 2014](https://sites.google.com/site/germeval2014ner/)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_NER.h5) | 93.87% | 90.22% | 
|[German (CoNLL 2003)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/DE_NER_CoNLL.h5) | 80.12% | 77.52% | 
|[German (GermEval 2014)](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/DE_NER_GermEval.h5) | 80.74% | 79.96% |


## Entities
Trained on ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_Entities.h5) | 83.93% | 85.68% | 


## Events
Trained on TempEval3 (https://www.cs.york.ac.uk/semeval-2013/task1/)

| Language | Development (F1) | Test (F1) |
|----------|:-----------:|:----:|
|[English](https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_SequenceTaggingModels/v2.1.5/EN_Events.h5) |- | 83.45% | 





