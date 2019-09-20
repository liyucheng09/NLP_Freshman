# NLP 研究方向 & 基础知识

## 必修

### 网络 & 预训练模型

**Transformer** 当前最风行的NLP神经网络。在大部分任务上效果超过RNN和CNN。相比于RNN，优点在于在大数据上训练时速度大幅度提升，同时允许多GPU并行。被目前几乎所有NLP模型所采用。
- 论文：https://arxiv.org/abs/1706.03762
- Harvard出品Transformer的pytorch版实现：https://nlp.seas.harvard.edu/2018/04/03/attention.html

**BERT** NLP大规模预训练语言模型。区别于unidirectional language model，BERT采用Mask Language Model，以便于得到每个位置上的双向的信息。使用BERT做base model可以提高大多数下游任务的效果。
- 论文：https://arxiv.org/abs/1810.04805
- Google research tensorflow 版实现：https://github.com/google-research/bert

**GPT & GPT-2** OpenAI出品，大规模预训练语言模型。由于BERT的Mask language model设定，使BERT很难应用在生成任务上（最近也有研究BERT做生成的，见下文）。GPT采用经典的unidirectional language model，并在大规模语料上预训练。GPT不止在生成任务上，在许多任务上都取得了很大的进步。
- GPT-2:https://openai.com/blog/better-language-models/
- GPT:https://openai.com/blog/language-unsupervised/

### 强化学习 & GAN

Reinforcement learning在近两年来开始在NLP领域展露头脚，学习Reinforcement Learning是十分有必要的。
- 入门可以观看：https://www.coursera.org/learn/practical-rl?specialization=aml
- 或者阅读综述（不推荐）：http://incompleteideas.net/book/bookdraft2017nov5.pdf
- An Introduction to Deep Reinforcement Learning: https://arxiv.org/abs/1811.12560
- 强化学习入门教程：https://simoninithomas.github.io/Deep_reinforcement_learning_Course/

GAN在NLP的许多任务上都有采用，学习GAN是必要的。建议从GAN在CV上的应用开始了解，最后阅读GAN在NLP领域上的论文。
GAN入门：https://zhuanlan.zhihu.com/p/58812258

GAN for text generation:

- GANs for Sequences of Discrete Elements with the Gumbel-softmax Distribution
- Generating Text via Adversarial Training
- SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- Adversarial Feature Matching for Text Generation
- Long Text Generation via Adversarial Training with Leaked Information   AAAI2018

### 迁移学习：Transfer-Learning

Transfer-learning 的思想在NLP的许多任务的方法上都有所体现。了解Transfer-learning不会让你在读论文时在这方面碰到障碍。

迁移学习的资料合集，包括论文和代码：

reference url: http://transferlearning.xyz/


## 研究方向：文本生成
文本生成有很长的研究历史，并且有众多分支任务，例如：机器翻译，对话生成，文本摘要等等。文本生成也分Unconditional text generation和Conditional text generation.

Conditional text generation指根据特定的条件（例如：问题；英语文本）生成特定的结果（例如：回答；中文文本）。Uncoditional text generation是文本生成的基础方式，可以续写文章等。

了解文本生成，综述：https://arxiv.org/abs/1703.09902


### 子方向一：对话生成

对话生成一般指给出问题生成回复。

推荐论文：

- Generating Informative Responses with Controlled Sentence Function （AAAI2018，清华）
- Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders
- Commonsense Knowledge Aware Conversation Generation with Graph Attention
- Adversarial learning for neural dialogue generation （通过对抗学习）

### 子方向二：机器翻译

推荐论文：

- Neural Machine Translation by Jointly Learning to Align and Translate
- A Method for Stochastic Optimization
- Neural Machine Translation of Rare Words with Subword Units
- Attention is All You Need （Transformer）


## 研究方向：关系抽取

根据ACL2019接受情况，关系抽取是当下最热门的研究方向，同时也是被接收论文最多的方向。

中文综述，包含简单模型和数据介绍：https://shomy.top/2018/02/28/relation-extraction/

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

## 研究方向：KBQA

基于知识库的问答系统。

推荐论文：

- [ACL15]Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base
- [ACL17]Improved Neural Relation Detection for Knowledge Base Question Answering