# Text Classification

## 项目介绍

本项目基于pytorch，利用TextCNN神经网络实现了文本分类。本项目代码完备，可以直接利用下面提供语料进行分类。

## 环境要求

- Dataset: [yahoo/dbpedia...](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
- Model: CNN
- Pytorch: 1.7.1
- Python: 3.81
- torchtext: 0.8.1.

## 运行

```python
python3 preprocess.py #preprocessing
```

```python
python3 main.py #training
```

```python
python3 test_eval.py #testing evaluation
>>1 #imputing full test dataset for testing
>>2 #inputing one sentence for classification
```

## 参考
https://github.com/catqaq/TextClassification

