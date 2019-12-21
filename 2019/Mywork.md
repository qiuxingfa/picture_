* hello everyone, my name is Qiu Xingfa, I'm very glad to introduce my research to you today. The field I worked on is 'Event Detection in Social Media'. It's a classic task in information retrieval. The meaning of this task is clustering real-time social media texts as events.
* There are several characteristics of social media texts, At first, They are short. Most of them are no more than one  hundred and forty characters. Second, they are irregular, abbreviation and spelling mistakes can often be found in the texts. Third, they are uneven distributed, some events may contain few texts while others can contain many. And they are noisy and numerous. All of these characteristics make this task difficult.
* We decide to use Single-Pass clustering algorithm to deal with this task. The traditional unsupervised clustering is not not suitable for this task because of the need of specifying the number of clusters in advance and the huge amount of calculation. While the Single-Pass clustering algorithm is a kind of simple and effective incremental clustering algorithm. From the flow chart we can know the flow of this algorithm. At first, we input a data T, then we use some NLP tools to extract some features and learn the representation of T, than we calculate the similarity between T and a chosen event E. If the result is greater than the threshold that we set in advance, we decide T belongs to E. Otherwise, system will create a new event which contains T only at this moment. Lastly system will update the event pool and wait to deal with next data.
* As all the present approaches are unsupervised, we decide to use a supervised approach to solve this problem. The main idea is to train a model of matching text-event pair from labeled data. We prepare twenty eight point eight thousand weibos from August 1,2019 to August 7,2019. We labeled 251 events from the corpus. And then we tend to use neural networks to extract text features and calculate them similarity between them.
* From the network structure we can figure that it's a typical Siamese structure which contains two inputs    and one output.
* After some experiments were conduct. We can find that using tf-idf as represent of text perform better than other unsupervised approaches which means that the similarity in characters still have strength when compare whit others. However, we believe that the supervised approach we proposed can perform better than them. The experiments are to be performed in the feature. 
* Here comes the summery. The problem we solve is to detect events in social media，we propose a supervised method to model the similarity between text-event pairs from labeled data. We use Single-Pass clustering algorithm and neural network to create this model. With this method, there is no need to specify parameters in advance and it can  avoid repeated iterations.