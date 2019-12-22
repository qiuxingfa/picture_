# Event Detection On Social Media

## 问题定义

* 顺序输入带时间标签的社交媒体文本，X(x1,x2,...,xT), 输出事件簇C(c1,c2,...,ck),假设每一个独立的文本xi都只属于某一事件cj，



## 数据

* TDT（Topic Detection and Tracking）数据集为大约十几年前的新闻报道相关的数据集，且为付费数据集
* 大部分论文使用的数据集并未公开，且很多人都是自己标注数据，大多数据来自于推特API（有限制）和爬虫，标准数据集的产生有三种方法：1）关注热门事件；2）人工标注一部分；3）聚类之后进行标注
* 事件的定义并不统一，事件的粒度难以把握，对于标注和比较造成了一定的困难
* 挑选2019-08-01至2019-08-07一周时间的热门话题（关键词），以此为基础确定热门事件，并对关键词进行一定程度的扩展，最终包括251个事件，1751个关键词，每个关键词匹配至多30条微博（挑选评论数多的），最后去除重复id，去除异常值，得到最终数据28832条

> 徐州教师夫妇
> 徐州女教师绝笔信
> 警方回应徐州女教师绝笔信
> 女教师绝笔信后未归
> 徐州举报人女儿讲述
> 写绝笔信女教师夫妇平安找到
> 写绝笔信女教师维权经过
> 徐州绝笔信女教师首发声
> 绝笔信女教师丈夫停职原因
> 绝笔信女教师回应索赔36万
> 徐州女教师当事民警发声
>
> ....

* 数据特点：包含大量的噪音，较短的文本（平均长度约为128），不规范的文本，由于事件本身会随时间发生变化和热门事件的讨论范围较广等因素，同一事件下的文本可能差异会很大（所包含的观点情绪，所注意的方面可能都不相同），不同事件下的文本可能又会很相似（如北京暴雨和深圳暴雨），事件所包含的微博数不均匀，最少2条，最多1180条，平均115条



## 相关技术

* Event Detection 分为 Specified Event Detection（SED） 和 Unspecified Event Detection（UED）
* SED即提前指定事件，将数据归入提前指定的事件中，和TDT中的Topic Tracking类似，大部分人使用有监督或者半监督的方法作为分类问题，输出每条数据属于不同事件的概率，主要问题集中于事件特征的构建和相似度的计算，监督方法基本上基于事件是静态的假设，而半监督的方法则会通过无标签的数据修正模型
* UED则不提前指定事件，大部分使用无监督的方法，用传统的k-means等算法需指定聚类数目且不断迭代，k难以评估且计算量大，Single-Pass是一种简单高效的增量式聚类方法

* New Event Detection(NED)实际上和SED相似，但只需要判断文档是否属于新事件，即二分类问题
* story link detection，即判断两个文档是否属于同一事件，与一般的QA或语义相似度任务类似，即二分类或者回归问题



## 特征提取和表示

* Keyword-Based（tf-idf）
* Twitter-Based（Hashtags, user information, time-stamps, retweets, and geo-tags）
* Location-Based
* Language-Based（POS）
* 关于事件的表示，参考聚类表示方法，有两种，一种是文档均值或者挑选某一代表文档来表示，另一种是用文本簇所表现的整体特征来表示，如实体图的方法
* 关于句子的表示，论文 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf) 表示bert直接拿最后一层的均值或者拿[CLS]作为句子的表示，直接做cos相似度计算，效果不如Glove向量平均，但后面如果加几层网络训练后，bert的效果会好一些
* ![](https://github.com/qiuxingfa/picture_/blob/master/2019/66c108f8d55764d1115d18ab8e62dc9.png)



## 匹配和排序的思路

* QA、阅读理解和文档检索经常会用匹配或排序的思路来解决
* 论文 [Story Disambiguation: Tracking Evolving News Stories across News and Social Streams](https://arxiv.org/pdf/1808.05906.pdf) 以1：10的正负比例比例构造事件-文本对，**Random Forest**作为分类器，训练一个事件-文本对的分类器，能够适应于不同的事件，但仍需提前指定部分文档表示事件，在分类过程中会对事件特征进行更新
* 匹配分为两种：
  * 一种是representation-focused的模型，尝试用深度神经网络对单独的文本建立好的表示方法，是一种对称的结构
  * 另一种是interaction-focused的模型，首先基于简单的表示方法建立局部的交互信息，然后使用深度神经网络学习匹配的层级的交互模式
    ![](https://github.com/qiuxingfa/picture_/blob/master/2019/cd9ccfc2cdbf95199535e0cf1d176be.png)
  * 论文[A Deep Relevance Matching Model for Ad-hoc Retrieval](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf)提到的Ad-hoc Retrieval任务，作者认为和NLP中的语义匹配任务不同，这一任务更加注重于相关性匹配，这一点和事件聚类相似，即相关性比语义相似更重要
* 主要思路：依时间顺序构建文本-事件对
  * pointwise：输出维度为1，正负均衡
  * pairwise：输出维度为2，同时输入正负样本，一般使用hinge loss

## 强化学习

* 在上述思路中，实际的输入输出其实是确定的，即变成一个确定的匹配问题，我们可以直接在这上面划分训练集和测试集，也可以从聚类的角度去评估
* 在实际的运行过程中，事件所包含的文本不一定是正确的，类似的任务包括序列标注和序列生成，即前一时刻的输出影响后一时刻的输出，前一时刻出错会导致错误传播，但与这些任务不同，事件聚类的文本前后关联并不大，
* 强化学习可利用标准数据集去指导当前状态去产生标签，这个时候输入事件可以是不确定的，可以是任意文本的组合，这和实际可能碰到的情况相符，也使得模型的泛化性更好
* 使用强化学习的方法可能会导致计算量增加

## 评估

![](https://github.com/qiuxingfa/picture_/blob/master/2019/797f99cbc48be2b4d93c65c3f1274fe.png)

![](https://github.com/qiuxingfa/picture_/blob/master/2019/8821563d9be21744f0d73026c9039ea.png)

## 总结

* 从标注数据中学习一种文本-事件的匹配模式
* 从匹配的思路去解决Unspecified的事件聚类的问题，与有监督的specified event的方法相比，不需要提前指定事件，也可以适应不同的未见过的事件，更具有现实意义，与无监督的方法相比，又不需要人为地指定事件数以及阈值等，具有更大的灵活性