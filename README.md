## Text Summarization and Generation on Arxiv Paper Abstracts

#### Project Description
This goal is to train an LSTM on the Abstract and map the paper title to the abstract using seq2seq from a similar idea of machine translation.  
There are two ways of doing this. Can either input keywords and title to predict the abstract or try the inevrse.

#### Dataset Description
The dataset is Arxiv Paper Metadata from 2007 to 2017. 
Link: https://github.com/avenxu/arxiv_meta.git


#### Mile stone
| Time      |     Task  |   Status   |
| :--------: | :--------:| :------:     |
| field1      |   field2  |  field3      |



#### Work Log
<<<<<<< HEAD

=======
| Date      |      Task|      Time taken|  Contributer|
| :-------- | :--------|  :--------     | :--------:  |
|180621     |Retrieve the data from S3 (Over 500 GB data) | 1 day | Yichuan |
|180621     |Search on possible structed paper data       | 3+3 h | Yichuan & Jingyun|
|1800623-0628|Background on text summarization and genenration | 7h | Jingyun|
|180627     |Load the metadata and apply word2vec     <br> desired dataset keeps symbosl whith [Word vector,Onehot string Category, Onehot for subject]   | 7 h    | Yichuan |
|180701-180717|Tried to get the data work with the seq2seq model. Code reference: https://github.com/google/seq2seq.git <br> But training is taking much longer than we thought so we decided to try something else (Char RNN/LSTM)| 23 h | Yichuan|
|180627-0722    | Udnerstand and modify a basic seq2seq with fixed seq length based on https://github.com/NELSONZHAO/zhihu/tree/master/basic_seq2seq?1521452873816. <br> Often problems emerged only after training for several epochs <br> Limited computation power to train| 26 h | Jingyun|
|180718-0723|Used the RNN model to train on arxiv model: https://github.com/martin-gorner/tensorflow-rnn-shakespeare.git <br> Deep Poetry: Word-Level and Character-Level Language Models for Shakespearean Sonnet | 17 h | Yichuan|
|180723-0802 |Learn basic pytorch, tried to modify a CharRNN and get the code running. <br> **Failed** with a configuration problem| 9 h | Jingyun|
|180724-0801|Working on the CharRNN based on the insight from https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm <br> First time needs to fix some bugs and issues from the old tf version. Including stacking LSTM cells <br> After bug fixes tried on training on the 2017 data and the result was terrible. Tried to figure out why| 11 h | Yichuan|
|180804|Figured out a bug in the code where the prediction will be random chars | 4 h | Yichuan|
|180804|Extract the cleaned subject and do one-hot encoding| 4 h | Jingyun | 
|180808|Fixed the bug and work on adding conditional input Also working on CNN models for text generation| 5 h | Yichuan|
|180809-0813|Understand the implemented code, read related papers and draw the architecture| 9 h | Jingyun|
|180812| Presentation slides | 5 h | Yichuan|
|180814| Report writing | 5+12 h | Yichuan & Jingyun|
**NOTE** : Almsot all the models are done be Yichuan both in terms of implementation and experiments. Jingyun tried to modify some model but mostly failed and therefore focus more on reading papers. 
>>>>>>> parent of 590af8c... README update


#### Sources used
| Task       |     Link |  
| :-------- | :--------|
| A Neural Attention Model for Sentence Summarization    |   https://www.aclweb.org/anthology/D/D15/D15-1044.pdf | 
|Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation|https://www.aclweb.org/anthology/D14-1179 | 
|An architectrue of RNN seq2seq model |http://ceur-ws.org/Vol-1769/paper12.pdf |
|Earlier approach for text summarization(a bench mark using statical learning)|ftp://ftp.cse.buffalo.edu/users/azhang/disc/disc01/cd1/out/papers/sigir/p152-chuang.pdf |
| Overview of text generation with various models |  https://noon99jaki.github.io/publication/2017-text-gen.pdf <br>  https://arxiv.org/pdf/1711.09534.pdf <br> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7944061 |
| Understanding convolution and dilated CNN on text | https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f <br> https://arxiv.org/pdf/1509.01626.pdf|
| Understand implementation of Neural Machine Translation (seq2seq) | https://github.com/tensorflow/nmt  <br> https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html <br> https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf <br> https://ai.googleblog.com/2017/04/introducing-tf-seq2seq-open-source.html |
| Understand implementation of CharRNN | http://karpathy.github.io/2015/05/21/rnn-effectiveness/ <br> https://arxiv.org/pdf/1508.06615.pdf| 
| RNN for text genenration | https://arxiv.org/pdf/1308.0850.pdf <br> http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf|
| Understand theory and implementation of Shakespeare CharRNN |https://blog.owulveryck.info/2017/10/29/about-recurrent-neural-network-shakespeare-and-go.html |
|(interesting) An architectrue combines CNN and RNN for news title summarization |https://web.stanford.edu/class/cs224n/reports/2760356.pdf |
| Graph architecture for seq2seq model  |  http://adventuresinmachinelearning.com/keras-lstm-tutorial/ |
|Could possibly be adapted Pointer-Generator Networks| https://arxiv.org/pdf/1704.04368.pdf |
|(interesting)Can be considered to genenrate abstracts with different strength for a subject|https://arxiv.org/pdf/1702.08139.pdf |



