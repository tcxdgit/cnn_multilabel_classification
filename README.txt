
# 用于进行意图识别/文本分类的分类器

load_data_for_cnn_multi.py      从mongoDB数据库中下载当前场景的数据,默认对句子进行分词预处理
train.py    训练分类器,需要修改代码中的场景名
config.py   调整分类模型的参数
cnn_attention_model.py  textCNN+attention模型的graph(attention思想参见论文Zhao Z, Wu Y. Attention-Based Convolutional Neural Networks for Sentence Classification[C]// INTERSPEECH. 2016:705-709.)
data_helper.py  数据处理部分
test.py     测试模型性能
cnnclasssify.py     预测代码，输入分词后的句子
mq_cnnclassify.py    serve部分，调用cnnclassify.py
classify.py     client部分

