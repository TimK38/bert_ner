# ner_project 

命名實體實作_by_bert

# name entity recognition-Pytorch-Implementation

build data preprocessing, dataset construction, model training, model evaulation, model preservation, model batch predict functions.

本專案以Pytroch為框架，conll2012_ontonotesv5為默認資料集，輸入指定預測實體(PERSON、NORP、FAC、ORG、GPE、LOC、PRODUCT、DATE、TIME、PERCENT、MONEY、QUANTITY、ORDINAL、CARDINAL、EVENT、WORK_OF_ART、LAW、LANGUAGE)後便能訓練出具特定實體辨識功能的NER模型，另外自行準備相同格式的訓練資料亦可訓練出實用的ner model 

## 環境配置

- pytorch
- numpy
- pickle
- transformers
- opencc
- datasets
- pandas
- itertools
- random
- seqeval
- matplotlib
- seaborn 
- keras
- sklearn
- tqdm

## 運行

可參考example.ipynb範例