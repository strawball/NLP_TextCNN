# Text Classification
- Dataset: [yahoo/dbpedia...](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
- Model: CNN
- Pytorch: 1.7.1
- Python: 3.81
- torchtext: 0.8.1.

## Training

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

## Data

Download from https://bhpan.buaa.edu.cn:443/link/EA97AFD48700F0073E3D1B9A9AB76157

由于语料文件和Glove文件太大，无法直接邮箱发送，所以上传到北航云盘。把下载到的data文件夹放在NLP文件夹下即可运行。