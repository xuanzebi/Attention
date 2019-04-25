## 收集的一些不同Attention机制的代码

- attention-keras.py          keras实现的attention is all you need        

  > [苏神的代码](https://github.com/bojone/attention/blob/master/attention_keras.py)

- attention-tensorflow.py        attention is all you need 中的attention 函数

- transformer.py                       attention is all you need 中的transformer模块

  > <https://github.com/google-research/bert/blob/master/modeling.py>
  >
  > https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

- attention-Intra-MHA-Inter-MHA.py        

  Multi-Head Attention （MHA） 基于aspect的情感识别中用的Intra-MHA和Inter-MHA的attention代码。

  > <https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aen.py>
  >
  > [Attentional Encoder Network for Targeted Sentiment Classification论文](https://arxiv.org/pdf/1902.09314.pdf)

- AttentionWeightedAverage.py         普通的attention。可以在过LSTM后使用。 pytorch版本和keras版本都有。



