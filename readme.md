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