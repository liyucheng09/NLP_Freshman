# NLP 学习路线

## 指南

###################

**竞赛**

查看NLP领域的众多竞赛情况（不断更新中）：[竞赛](https://github.com/nine09/NLP-Syllabus/blob/master/NLP_Competitions.md)

##################

**进阶指南：**

研究生阶段正式开始。NLP研究方向，必修知识指南在这里：[进阶](https://github.com/nine09/NLP-Syllabus/blob/master/researcher.md)

###################

以下为研究生入门指南。

## 基础知识

1. 数学
	- 微积分
	- 概率论
	- 线性代数
2. Python
	- virtualenvironment
	- numpy
	- sklearn
3. 深度学习框架：**用到再看**
	- Tensorflow
	- Pytorch

## 课程学习

**Stanford CS224n**

Download Slides and Pdf lecture in: http://web.stanford.edu/class/cs224n/

Videos in: https://www.bilibili.com/video/av13383754?from=search&seid=5889103122225870394

完成课程学习与课程作业3

## 一：文本分类

文本分类是最基础，最简单的NLP任务形式，即句子分类：给定input句子，预测该句所属的类别。通常，解决该问题通常需要两步：1）将输入文本编码为连续空间中的一个向量，接下来 2）基于该向量进行分类。其中步骤1可以使用word2vec，神经网络等方法，步骤2多使用Linear Layers。

了解该过程可以建立对表示学习的基本认识，并认识到编码过程和预测过程的相对独立性。

必要知识：tensorflow & sklearn; word2vec; CNN & RNN; keras

参考url: https://zhuanlan.zhihu.com/p/26729228

Download Data Set: http://www.sogou.com/labs/resource/cs.php

比较以下模型的分类效果：

- CNN
- LSTM
- Naive Bayes
- SVM

其中，深度学习模型需要使用word2vec初始化。

## 二：序列标注

序列标注是词性标注，实体识别，信息抽取等众多任务的常见解法。其旨在对句子中的每一个词/字进行分类。在了解该问题过程中，我们可以感受实际NLP问题是如何被分解/聚合后套在方便实现的框架下的。例如，实体识别任务是如何通过对逐字分类的过程中完成的。

序列标注可以采用经典概率图方法，例如HMM，CRF等：http://fancyerii.github.io/books/sequential_labeling/

但直接了解基于神经网络的方法更直接（推荐）：https://zhuanlan.zhihu.com/p/34828874

拓展阅读：https://zhuanlan.zhihu.com/p/268579769

## 三：seq2seq

reference url: https://github.com/NELSONZHAO/zhihu/tree/master/basic_seq2seq?1521452873816

使用Tensorflow / Pytorch 实现seq2seq模型

拓展知识：

- Attention mechanism 注意力机制 （重要）
	- BiRNN + Attention 机器翻译模型: https://zhuanlan.zhihu.com/p/37290775
	- 推荐paper：Attention based documents classification: http://www.aclweb.org/anthology/N16-1174
- 推荐项目：OpenNMT
	- url: http://opennmt.net/

## 四：Transformer

Attention is all you need: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

中文：https://zhuanlan.zhihu.com/p/48508221

transformer基本已经统一深度学习届，所以算是必读的论文了。

## 五：BERT & GPT

预训练语言模型是近些年NLP届最大突破。现在预训练+微调已经成为所有研究者的工作模式。预训练预言模型最经典的两个model就是BERT和GPT。两者的核心思路都十分简单，不建议读论文。

BERT：https://zhuanlan.zhihu.com/p/51413773

GPT：https://zhuanlan.zhihu.com/p/125139937

了解BERT的最好方法是使用BERT解决序列标注问题，可以很清晰的了解其优点与缺点。
通过transformers的example可以很好的了解，顺便入门transformers这个library了。
https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py


## 拓展一：知识图谱

知识图谱入门课程：

百度网盘链接: https://pan.baidu.com/s/1NzUdiIkIk330VxbWEGL3MQ 提取码: 32q3

了解知识图谱的基础知识，常用工具与研究方向