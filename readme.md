# NLP 学习路线

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

完成课程学习与课程作业

## 实践一：文本分类

必要知识：tensorflow & sklearn; word2vec; CNN & RNN

参考url: https://zhuanlan.zhihu.com/p/26729228

Download Data Set: http://www.sogou.com/labs/resource/cs.php

比较以下模型的分类效果：

- CNN
- LSTM
- Naive Bayes
- SVM

其中，深度学习模型需要使用word2vec初始化。

## 实践二：seq2seq

reference url: https://github.com/NELSONZHAO/zhihu/tree/master/basic_seq2seq?1521452873816

使用Tensorflow / Pytorch 实现seq2seq模型

拓展知识：

- Attention mechanism 注意力机制 （重要）
	- BiRNN + Attention 机器翻译模型: https://zhuanlan.zhihu.com/p/37290775
	- 推荐paper：Attention based documents classification: http://www.aclweb.org/anthology/N16-1174
- 推荐项目：OpenNMT
	- url: http://opennmt.net/

## 课程学习三：知识图谱

知识图谱入门课程：

百度网盘链接: https://pan.baidu.com/s/1NzUdiIkIk330VxbWEGL3MQ 提取码: 32q3

了解知识图谱的基础知识，常用工具与研究方向

## 拓展方向一：对话生成

推荐论文：

- Generating Informative Responses with Controlled Sentence Function
- Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders
- Commonsense Knowledge Aware Conversation Generation with Graph Attention

## 拓展方向二：关系抽取

了解关系抽取的研究进展，baseline，和state of the art 方法。

[中文综述，包含简单模型和数据介绍]:https://shomy.top/2018/02/28/relation-extraction/

推荐论文：

- 当前所有论文的baseline：Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Network
- 2018年的State-of-the-art：Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism
- 2016年的SOTA-效果依然很好：End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures
- 使用GAN处理远程监督数据噪声-当前的重要研究方向之一2018AAAI：Reinforcement Learning for Relation Classification from Noisy Data
- 联合抽取实体和关系
	- Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme
	- Going out on a limb: Joint Extraction of Entity Mentions and Relations without Dependency Trees
- 多样例，多标签的抽取：Multi-instance Multi-label Learning for Relation Extraction
- An interpretable Generative Adversarial Approach to Classification of Latent Entity Relations in Unstructured Sentences
- Distant supervision for relation extraction without labeled data

## 拓展方向三：KBQA

基于知识库的问答系统。

推荐论文：

- [ACL15]Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base
- [ACL17]Improved Neural Relation Detection for Knowledge Base Question Answering

## 拓展方向四：机器翻译

推荐论文：

- Neural Machine Translation by Jointly Learning to Align and Translate
- A Method for Stochastic Optimization
- Neural Machine Translation of Rare Words with Subword Units
- Attention is All You Need

## 拓展学习：Transfer-Learning

迁移学习的资料合集，包括论文和代码：

reference url: http://transferlearning.xyz/

## 拓展学习：Reinforcement Learning & GAN

GAN for text generation:

推荐论文：

- GANs for Sequences of Discrete Elements with the Gumbel-softmax Distribution
- Generating Text via Adversarial Training
- SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- Adversarial Feature Matching for Text Generation
- Long Text Generation via Adversarial Training with Leaked Information   AAAI2018


强化学习：

- An Introduction to Deep Reinforcement Learning: https://arxiv.org/abs/1811.12560
- 强化学习入门教程：https://simoninithomas.github.io/Deep_reinforcement_learning_Course/