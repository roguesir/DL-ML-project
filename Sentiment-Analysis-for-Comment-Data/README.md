#环境要求
- Unix/Linux系统
- python 2.7
- python包安装： keras,sklearn,gensim,jieba,h5py,numpy,pandas
```
sudo pip install -r requirements.txt
```
# 用法

## 使用SVM分类器进行情感分类：
```
python predict.py svm 这个手机质量很好，我很喜欢，不错不错

```
```
python predict.py svm 这书的印刷质量的太差了吧，还有缺页，以后再也不买了

```

#数据
- ./data/ 原始数据文件夹
  - data/neg.xls 负样本原始数据
  - data/pos.xls 正样本原始数据

- ./svm_data/ svm数据文件夹
  - ./svm_data/\*.npy 处理后的训练数据和测试数据
  - ./svm_data/svm_model/ 保存训练好的svm模型
  - ./svm_data/w2v_model/ 保存训练好的word2vec模型

