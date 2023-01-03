# Natural Language Processing
It contains details about the different topics in Natural language Processing.
* TF-IDF
* Conditional Random Field
* Word Embeddings
    * Word2vec
    * Glove
* Character level embeddings
    * Fasttext
* Recurrent Neural Network
* LSTM
* ELMO
* Flair
* ULMFIT
* Attention
* Transformer
* BERT
* BERT limitations
* ALBERT
* DistilBERT
* ROBERTA
* Sentence BERT
* XLNET

# TF-IDF
* TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
* This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
* TF-IDF was invented for document search and information retrieval. It works by increasing proportionally to the number of times a word appears in a document, but is offset by the number of documents that contain the word. So, words that are common in every document, such as this, what, and if, rank low even though they may appear many times, since they don’t mean much to that document in particular.
* However, if the word Bug appears many times in a document, while not appearing many times in others, it probably means that it’s very relevant. For example, if what we’re doing is trying to find out which topics some NPS responses belong to, the word Bug would probably end up being tied to the topic Reliability, since most responses containing that word would be about that topic.
* ***How is TF-IDF calculated?***
    * TF-IDF for a word in a document is calculated by multiplying two different metrics:
        * The term frequency of a word in a document. There are several ways of calculating this frequency, with the simplest being a raw count of instances a word appears in a document. Then, there are ways to adjust the frequency, by length of a document, or by the raw frequency of the most frequent word in a document.
        * The inverse document frequency of the word across a set of documents. This means, how common or rare a word is in the entire document set. The closer it is to 0, the more common a word is. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm.
        * So, if the word is very common and appears in many documents, this number will approach 0. Otherwise, it will approach 1.
    * Multiplying these two numbers results in the TF-IDF score of a word in a document. The higher the score, the more relevant that word is in that particular document.
    * The TF-IDF score for the word t in the document d from the document set D is calculated as follows:
    tf_idf(t,d,D) = tf(t,d)*idf(t,D)
    where
    tf(t,d) = log(1+freq(t,d))
    idf(t,D) = log(N/count(d belongs to D: t belongs to d))
* References:
    * https://monkeylearn.com/blog/what-is-tf-idf/
    * https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/

# Conditional Random Field
* Conditional Random Fields are a discriminative model, used for predicting sequences. They use contextual information from previous labels, thus increasing the amount of information the model has to make a good prediction.
* Part of speech tagging:
    * Let’s go into some more detail, using the more common example of part-of-speech tagging. In POS tagging, the goal is to label a sentence (a sequence of words or tokens) with tags like ADJECTIVE, NOUN, PREPOSITION, VERB, ADVERB, ARTICLE. For example, given the sentence “Bob drank coffee at Starbucks”, the labeling might be “Bob (NOUN) drank (VERB) coffee (NOUN) at (PREPOSITION) Starbucks (NOUN)”. So let’s build a conditional random field to label sentences with their parts of speech. Just like any classifier, we’ll first need to decide on a set of feature functions fi.
    * ***Feature Functions in a CRF***
        * In a CRF, each feature function is a function that takes in as input:
            * a sentence s
            * the position i of a word in the sentence
            * the label li of the current word
            * the label li−1 of the previous word
        and outputs a real-valued number (though the numbers are often just either 0 or 1).
        Note: by restricting our features to depend on only the current and previous labels, rather than arbitrary labels throughout the sentence, I’m actually building the special case of a linear-chain CRF. 
    * ***Features to Probabilities***
        * Next, assign each feature function fj a weight λj (I’ll talk below about how to learn these weights from the data). Given a sentence s, we can now score a labeling l of s by adding up the weighted features over all words in the sentence:
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im1.PNG) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im2.PNG) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im3.PNG) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im4.PNG) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im5.PNG) <br/>

* References:
    * http://www.alias-i.com/lingpipe/demos/tutorial/crf/read-me.html
    * https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/
    * https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541

# Word Embeddings
Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. Word embeddings help in generating the distributed representation for text which are lower in dimension as compare to other text representation techniques like TF-IDF.
One of the benefits of using dense and low-dimensional vectors is computational: the majority of neural network toolkits do not play well with very high-dimensional, sparse vectors. … The main benefit of the dense representations is generalization power: if we believe some features may provide similar clues, it is worthwhile to provide a representation that is able to capture these similarities.
## Word2Vec(Skipgram)
* Word2Vec uses a trick in which we train a simple neural network with a single hidden layer to perform a certain task, but then we’re not actually going to use that neural network for the task we trained. nstead, the goal is actually just to learn the weights of the hidden layer–we’ll see that these weights are actually the “word vectors” that we’re trying to learn.
* ***The Fake task***: Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose. Here, "nearby" means there is actually a "window size" parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead (10 in total). The output probabilities are going to relate to how likely it is find each vocabulary word nearby our input word. For example, if you gave the trained network the input word “Soviet”, the output probabilities are going to be much higher for words like “Union” and “Russia” than for unrelated words like “watermelon” and “kangaroo”.
* We’ll train the neural network to do this by feeding it word pairs found in our training documents. The below example shows some of the training samples (word pairs) we would take from the sentence “The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. The word highlighted in blue is the input word.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im6.png) <br/>
The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.
* ***Model Details***
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im7.png) <br/>
The above is the architecture of 1 hidden layer neural network. 
* Let's consider we have 10,000 uinque words in the vocab. Here we represent any input work like "machine" as a one-hot vector. The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, the probability that a randomly selected nearby word is that vocabulary word. There is no activation function on the hidden layer neurons, but the output neurons use softmax.
* When training this network on word pairs, the input is a one-hot vector representing the input word and the training output is also a one-hot vector representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating point values, not a one-hot vector).
* ***Hidden Layer***
    * Let's say we are trying to learn word vectors with 300 dimension. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron). If you look at the rows of this weight matrix, these are actually what will be our word vectors!
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im8.png) <br/>
So the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!. Here the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the “word vector” for the input word.
* ***Output layer***
    * The 1 x 300 word vector for “machine” then gets fed to the output layer. The output layer is a softmax regression classifier. Each output neuron (one per word in our vocabulary!) will produce an output between 0 and 1, and the sum of all these output values will add up to 1. Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, then it applies the function exp(x) to the result. Finally, in order to get the outputs to sum up to 1, we divide this result by the sum of the results from all 10,000 output nodes.
* ***Intuition***
    * If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. And one way for the network to output similar context predictions for these two words is if the word vectors are similar. So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words. And what does it mean for two words to have similar contexts? I think you could expect that synonyms like “intelligent” and “smart” would have very similar contexts. Or that words that are related, like “engine” and “transmission”, would probably have similar contexts as well. This can also handle stemming for you – the network will likely learn similar word vectors for the words “ant” and “ants” because these should have similar contexts.
* ***References***
    * http://ruder.io/word-embeddings-softmax/index.html#noisecontrastiveestimation
    * http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    * http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    * https://machinelearningmastery.com/what-are-word-embeddings/
    * http://jalammar.github.io/illustrated-word2vec/

## Glove
GloVe stands for global vectors for word representation. Unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors. It generates word embeddings by aggregating global word-word co-occurrence matrix from a corpus. It is an approach to marry both the global statistics of matrix factorization techniques like LSA (Latent Semantic Analysis) with the local context-based learning in word2vec. Rather than using a window to define local context, GloVe constructs an explicit word-context or word co-occurrence matrix using statistics across the whole text corpus. 
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im9.PNG) <br/>
* Details
    * https://cran.r-project.org/web/packages/text2vec/vignettes/glove.html
    * https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6 (best)
    * https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010

# Character level embeddings
## Fasttext
FastText is designed to be simple to use for developers, domain experts, and students. It's dedicated to text classification and learning word representations, and was designed to allow for quick model iteration and refinement without specialized hardware. fastText models can be trained on more than a billion words on any multicore CPU in less than a few minutes. Fasttext embeddings handles rare words better than word2vec.
* References
    * https://pypi.org/project/fasttext/
    * https://radimrehurek.com/gensim/models/fasttext.html
    * https://arxiv.org/abs/1607.04606v2 (paper)

# Recurrent Neural Network
RNN is a family of neural network models which is designed to model sequential data. The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back a fixed number of steps (more on this later). Here is what a typical RNN looks like:
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im10.jpg) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im11.png) <br/>
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im12.PNG) <br/>
* ***Vanishing Gradient***
In RNN, the information travels through time which means that information from previous time points is used as input for the next time points. Cost function calculation or error is done at each point of time. Basically, during the training, your cost function compares your outcomes (red circles on the image below) to your desired output. As a result, you have these values throughout the time series, for every single one of these red circles. 
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im13.PNG) <br/>
Let’s focus on one error term et. You’ve calculated the cost function et, and now you want to propagate your cost function back through the network because you need to update the weights.
Essentially, every single neuron that participated in the calculation of the output, associated with this cost function, should have its weight updated in order to minimize that error. And the thing with RNNs is that it’s not just the neurons directly below this output layer that contributed but all of the neurons far back in time. So, you have to propagate all the way back through time to these neurons. The problem relates to updating wrec (weight recurring) – the weight that is used to connect the hidden layers to themselves in the unrolled temporal loop.
For instance, to get from xt-3 to xt-2 we multiply xt-3 by wrec. Then, to get from xt-2 to xt-1 we again multiply xt-2 by wrec. So, we multiply with the same exact weight multiple times, and this is where the problem arises: when you multiply something by a small number, your value decreases very quickly. As we know, weights are assigned at the start of the neural network with the random values, which are close to zero, and from there the network trains them up. But, when you start with wrec close to zero and multiply xt, xt-1, xt-2, xt-3, … by this value, your gradient becomes less and less with each multiplication.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im14.PNG) <br/>
* ***What does this mean for the network?***
The lower the gradient is, the harder it is for the network to update the weights and the longer it takes to get to the final result. For instance, 1000 epochs might be enough to get the final weight for the time point t, but insufficient for training the weights for the time point t-3 due to a very low gradient at this point. However, the problem is not only that half of the network is not trained properly. The output of the earlier layers is used as the input for the further layers. Thus, the training for the time point t is happening all along based on inputs that are coming from untrained layers. So, because of the vanishing gradient, the whole network is not being trained properly. For the vanishing gradient problem, the further you go through the network, the lower your gradient is and the harder it is to train the weights, which has a domino effect on all of the further weights throughout the network.
* References:
    * https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem/
    * https://blog.paperspace.com/recurrent-neural-networks-part-1-2/
    * https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-1/
    * http://karpathy.github.io/2015/05/21/rnn-effectiveness/

# LSTM(Long Short Term Memory networks)
* LSTMs are explicitly designed to avoid the long-term dependency problem or vanishing gradient problem. The key to LSTMs is the cell state, the horizontal line running through the top of the diagram. The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im15.png) <br/>
* The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!”. An LSTM has three of these gates, to protect and control the cell state.
* The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1 and xt, and outputs a number between 0 and 1 for each number in the cell state Ct−1. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.” 
* Let’s go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used. When we see a new subject, we want to forget the gender of the old subject.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im16.png) <br/>
* The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state. In the example of our language model, we’d want to add the gender of the new subject to the cell state, to replace the old one we’re forgetting.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im17.png) <br/>
* It’s now time to update the old cell state, Ct−1, into the new cell state Ct. The previous steps already decided what to do, we just need to actually do it.
We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it∗C~t. This is the new candidate values, scaled by how much we decided to update each state value. In the case of the language model, this is where we’d actually drop the information about the old subject’s gender and add the new information, as we decided in the previous steps.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im18.png) <br/>
* Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to. For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that’s what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that’s what follows next.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im19.png) <br/>
* ***How LSTM solves vanishing gradient problem***
    * https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html
* References:
    * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    * https://kikaben.com/long-short-term-memory/
    * https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html

# ELMO(Embeddings from Language Models) 
ELMO achieves state-of-the-art performance on many popular tasks including question-answering, sentiment analysis, and named-entity extraction
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im20.gif) <br/>
ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass. 
* The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
* These raw word vectors act as inputs to the first layer of biLM
* The forward pass contains information about a certain word and the context (other words) before that word
* The backward pass contains information about the word and the context after it
* This pair of information, from the forward and backward pass, forms the intermediate word vectors
* These intermediate word vectors are fed into the next layer of biLM
* The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors
As the input to the biLM is computed from characters rather than words, it captures the inner structure of the word. For example, the biLM will be able to figure out that terms like beauty and beautiful are related at some level without even looking at the context they often appear in. Sounds incredible!
Unlike traditional word embeddings such as word2vec and GLoVe, the ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts.
* ***Elmo is open source***
    * It's code and datasets are open source. It has a website which includes not only basic information about it, but also download links for the small, medium, and original versions of the model. People looking to use ELMo should definitely check out this website to get a quick copy of the model. Moreover, the code is published on GitHub and includes a pretty-extensive README that lets users know how to use ELMo. I’d be surprised if it took anyone more than a few hours to get a working ELMo model going.
    * ***Code***: https://github.com/allenai/allennlp
    * ***Website***: https://allenai.org/allennlp/software/elmo
* References
    * https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/
    * https://arxiv.org/pdf/1802.05365.pdf (paper)
    * https://towardsdatascience.com/elmo-why-its-one-of-the-biggest-advancements-in-nlp-7911161d44be

# FLAIR
FLAIR library provides powerful state-of-the-art contextual word embeddings. It provides following features:
* Flair supports a number of word embeddings used to perform NLP tasks such as FastText, ELMo, GloVe, BERT and its variants, XLM, and Byte Pair Embeddings including Flair Embedding.
* The Flair Embedding is based on the concept of contextual string embeddings which is used for Sequence Labelling.
* Using Flair you can also combine different word embeddings together to get better results.
Flair supports a number of languages.
In this word embedding each of the letters in the words are sent to the Character Language Model and then the input representation is taken out from the forward and backward LSTMs.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im21.png) <br/>
The input representation for the word ‘Washington’ is been considered based on the context before the word ‘Washington’. The first and last character states of each word is taken in order to generate the word embeddings.
You can see that for the word ‘Washington’ the red mark is the forward LSTM output and the blue mark is the backward LSTM output. Both forward and backward contexts are concatenated to obtain the input representation of the word ‘Washington’.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im22.png) <br/>
After getting the input representation it is fed to the forward and backward LSTM to get the particular task that you are dealing with. In the diagram mentioned we are trying to get the NER.
* References:
    * https://github.com/flairNLP/flair
    * https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/
    * https://www.section.io/engineering-education/how-to-create-nlp-application-with-flair/
# ULMFit(Transfer learning)
* Universal Language Model FIne-Tuning(ULMFIT) is a transfer learning technique which can help in various NLP tasks. It has been state-of-the-art NLP technique for a long time, but then it was dethroned by BERT[which recently got dethroned by XLNet in text classification]
* Deep learning requires a lot of dataset. Specifically when doing transfer learning, we have a large dataset on which our base model is build and we transfer learn the parameters of the neural network to our domain specific dataset. When we have a smaller domain specific dataset, the models overfit. To solve this problem, Jeremy Howard and Sebastian Ruder suggest 3 different techniques in there paper on Universal Language Model Fine-tuning for Text Classification for fine-tuning in transfer learning LMs for NLP specific tasks
    * ***Discriminative fine-tuning***
        * Discriminative fine-tuning means using a larger learning rate for the last layer and decrease the learning rate for each layer, consecutively until the first.
        * For example: use lr=0.01 for the last (most specific) layer, lr=0.005  for the second-last, etc.
    * ***Slanted triangular learning rates***
        * Slanted Triangular Learning Rate is a learning rate schedule; the maximum learning rate (last layer) grows linearly until it maxes out and then starts to be lowered
    * ***Gradual unfreezing***
        * Refers to unfreezing one layer per epoch, starting at the last (most specific) layer. Then, for each new epoch, one extra layer is added to the set of unfrozen layers, and these get to be fine-tuned in that epoch.
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im23.png) <br/>
* There are 3 steps in ULMFit
    1. ***General-domain LM pretraining(Unsupervised)***: The LM is trained on a general-domain corpus to capture general features of the language in different layers.
    2. ***Target task LM fine-tunning(Unsupervised)***: Now that the model has captured the general features of the language. To make it useful for our domain-specific use case, we can fine-tune the paraments of LM using target task data. The distribution of the vocabulary in our data may differ from the pre-trained model. This process is a semi-supervised learning task. 
        * This step uses discriminative fine-tuning and slanted triangular learning rates. 
    3. ***Target task classifier fine-tuning***: In this step, the classifier is fine-tuned on the target task using the same architecture with two additional linear blocks. This is a supervised learning task. The parameters in these task-specific classifier layers are the only ones that can be learned from scratch. For this reason, the Author concatenates the last hidden state with both the max-pooled and the mean-pooled representation of the hidden states.
        * This step uses discriminative fine-tuning, slanted triangular learning rates and gradual unfreezing of classifier layers
* References:
    * https://queirozf.com/entries/paper-summary-ulmfit-universal-language-model-fine-tuning-for-text-classification
    * https://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html
    * https://arxiv.org/abs/1801.06146v5 (paper)
    * https://towardsdatascience.com/understanding-language-modelling-nlp-part-1-ulmfit-b557a63a672b
    * https://medium.com/@j.13mehul/simplified-details-of-ulmfit-452c49294fb8

# Attention
* ***Problems with RNN/LSTM*** 
    * In Encoder-docoder model, the encoder LSTM is used to process the entire input sentence and encode it into a context vector, which is the last hidden state of the LSTM/RNN. This is expected to be a good summary of the input sentence. All the intermediate states of the encoder are ignored, and the final state id supposed to be the initial hidden state of the decoder
    * The decoder LSTM or RNN units produce the words in a sentence one after another
    * RNNs cannot remember longer sentences and sequences due to the vanishing/exploding gradient problem. LSTM tries to solve vanishing gradient but still it's not able to solve completely. Although an LSTM is supposed to capture the long-range dependency better than the RNN, it tends to become forgetful in specific cases. Another problem is that there is no way to give more importance to some of the input words compared to others while translating the sentence. 
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im24.png) <br/>
* ***Attention mechanism***
    * Attention mechanism tries to overcome the information bottleneck of the intermediary state by allowing the decoder model to access all the hidden states, rather than a single vector — aka intermediary state — build out of the encoder’s last hidden state, while predicting each output.
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im25.png) <br/>
    The input to a cell in decoder now gets the following values:
    * The previous hidden state of the decoder model Hₖ-₁.
    * The previous output of decoder model Yₖ-₁.
    * A context vector Cₖ— a weighted sum of all encoder hidden states(hⱼ’s) aka annotations.
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im26.png) <br/>
    The context vector or intermediate vector ci for the output word yi is generated using the weighted sum of the annotations:
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im27.png) <br/>
    The weights αij are computed by a softmax function given by the following equation:
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im28.png) <br/>
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im29.png) <br/>
    eij is the output score of a feedforward neural network described by the function a that attempts to capture the alignment between input at j and output at i.
    * The global alignment weights are important because they tell us which annotations(s) to focus on for the next output. The weights will and should vary in each time steps of the decoder model. They are calculated by using a feed forward neural network.
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im30.png) <br/>
    While predicting the next step, weights are high — shown in white — only for a few words at a time. No more than 3–4 words have high attention for a given output word.
    * References:
        * https://medium.com/analytics-vidhya/https-medium-com-understanding-attention-mechanism-natural-language-processing-9744ab6aed6a
        * https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
        * https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html
        * https://towardsdatascience.com/attaining-attention-in-deep-learning-a712f93bdb1e

# Transformer
* Transformers are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) in recent years. They were introduced in a 2017 paper by Vaswani et al. called "Attention is All You Need," which demonstrated their effectiveness in a variety of tasks such as machine translation and language modeling.
* Self-attention mechanism is introduced, which allow the model to consider the entire input sequence when making predictions, rather than just the previous few tokens as in traditional RNNs (recurrent neural networks). This makes transformers particularly well-suited for tasks that require understanding long-range dependencies, such as translation and summarization.
* Another advantage of transformers is their parallelizability. Unlike RNNs, which must be processed sequentially, the self-attention mechanisms in transformers allow the model to simultaneously attend to all the input tokens, making it possible to parallelize the computation across multiple devices. The feedforward layer computations can also be parallelized because each word is processed separately in the FNN layers. This has made it possible to train very large transformer models, such as BERT and GPT-3, which have achieved state-of-the-art results on a wide range of NLP tasks.
* Another key innovation of the transformer architecture is the use of multi-headed attention. This allows the model to attend to multiple input subspaces at the same time, which can be useful for capturing complex relationships betweed words in the sentence.
* ***Architecture***
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im31.png) <br/>
We will discuss each component in the architecture:
  - ***Self Attention***:
    - Steps:
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im33.png) <br/>
      - The word embedding is transformed into three separate matrices — queries, keys, and values — via multiplication of the word embedding against three matrices with learned weights. These vectors are trained and updated during the training process.
      - Consider this sentence- “transformers are great”. To calculate the self-awareness of the first word "transformers”, calculate the scores of all the words in the phrase related to transformers”. This score determines the importance of other words when encoding a particular word in the input sequence.
      - The score for the first word is calculated by taking the dot product of the Query vector (q1) with the keys vectors (k1, k2, k3) of all the words
      - Then, these scores are divided by 8 which is the square root of the dimension of the key vector:
      - Next, these scores are normalized using the softmax activation function
      - These normalized scores are then multiplied by the value vectors (v1, v2, v3) and sum up the resultant vectors to arrive at the final vector (z1). This is the output of the self-attention layer. It is then passed on to the feed-forward network as input
      - Same process is done for all the words
    - ***Self Attention in Decoder***: In the decoder part, the self attention is mostly same as in encoder except the scope is limited to the words that occur before a given word. This prevents any information leaks during the training of the model. This is done by masking the words that occur after it for each step. So for step 1, only the first word of the output sequence is NOT masked, for step 2, the first two words are NOT masked and so on.
    In the Transformer architecture, self-awareness is calculated independently of each other, not just once, but multiple times in parallel. Therefore, it is called multi-head attention. The outputs are concatenated and transformed linearly, as shown in the following figure.
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im32.png) <br/>
  - ***Encoder-Decoder Attention***: There is an attention layer between encoder and decoder that helps the decoder focus on relevant parts of the input sentence(similary to what attention does in seq2seq models). The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.
  
  - ***Feedforward neural network(in encoder and decoder block)***: Each word is processed in the FNN separately. This allows parallelization. Each layer processes the input data and produces an output, which is then passed on to the next layer. Without normalization, the inputs to each layer can vary widely in scale, which can make it difficult for the model to learn effectively. Layer normalization addresses this issue by normalizing the inputs to each layer across the feature dimensions. This helps to stabilize the learning process and improve the model's ability to generalize to new data.
  - ***Residual connection & Layer Normalization*** : Residual connections mainly help mitigate the vanishing gradient problem. Another effect of residual connections is that the information stays local in the Transformer layer stack. The self-attention mechanism allows an arbitrary information flow in the network and thus arbitrary permuting the input tokens. The residual connections, however, always "remind" the representation of what the original state was.Layer normalization helps in normalizing the inputs in each layer in the model. It helps to stabilize the learning process and improve the model's ability to generalize to new data. Each layer processes the input data and produces an output, which is then passed on to the next layer. Without normalization, the inputs to each layer can vary widely in scale, which can make it difficult for the model to learn effectively. Layer normalization addresses this issue by normalizing the inputs to each layer across the feature dimensions. This helps to stabilize the learning process and improve the model's ability to generalize to new data.
  - ***Positional Embeddings***: Positional Embeddings is added to the word input embeddings before passing to the transformer model to include the information about the position of the word. Positional embeddings follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.
  - ***Final layer and softmax layer(post-processing)***: The goal of the final layer(fully connected linear layer) is to convert the decoder output vector into a much larger vector(size of vocab) called a logit vector. The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.
  - ***Limitations of transformer***
    - Attention can only deal with fixed-length text strings. The text has to be split into a certain number of segments or chunks before being fed into the system as input
    - This chunking of text causes context fragmentation. For example, if a sentence is split from the middle, then a significant amount of context is lost. In other words, the text is split without respecting the sentence or any other semantic boundary
    - Transformer-XL tries to overcome some of the limitations in the original transformer model.
* References:
    - https://jalammar.github.io/illustrated-transformer/
    - https://blog.knoldus.com/what-are-transformers-in-nlp-and-its-advantages/#:~:text=NLP's%20Transformer%20is%20a%20new,relies%20entirely%20on%20self%2Dattention.
    - https://towardsdatascience.com/transformers-89034557de14
    - https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
    - https://stats.stackexchange.com/questions/565196/why-are-residual-connections-needed-in-transformer-architectures#:~:text=The%20reason%20for%20having%20the,derivative%20of%20the%20activation%20function.

# BERT
- BERT, which stands for "Bidirectional Encoder Representations from Transformers," achieves a state-of-the-art performance on a wide range of natural language processing tasks, including language translation, question answering, and text classification. BERT is categorized as autoEncoder(AE) language. The AE language model aims to reconstruct the original data from corrupted input. The corrupted input means we use [MASK] to replace the original token into in the pre-train phase. It is a "bidirectional" model as it is able to consider the context of a word in both directions which allows it to better understand the nuances and relationships between words in a sentence
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im34.png) <br/>
There are 2 steps in BERT:
  1. Semi-supervised training on large amount of text data(books,wikipedia. etc)
  2. Supervised training on domain specific task with a labelled data.
We can also use pretrained BERT model(skip the first step) and finetuned on task specific dataset.
* ***Architecture***
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im35.png) <br/>
  - Since BERT’s goal is to generate a language representation model, it only needs the encoder part
    - BERT-Base, Uncased: 12-layers, 768-hidden, 12-attention-heads, 110M parameters
    - BERT-Large, Uncased: 24-layers, 1024-hidden, 16-attention-heads, 340M parameters
    - BERT-Base, Cased: 12-layers, 768-hidden, 12-attention-heads , 110M parameters
    - BERT-Large, Cased: 24-layers, 1024-hidden, 16-attention-heads, 340M parameters
* ***Preprocessing of text***
The input to the encoder for BERT is a sequence of tokens, which are first converted into vectors and then processed in the neural network. But before processing can start, BERT needs the input to be massaged and decorated with some extra metadata:
  1. ***Token embeddings***: A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.
  2. ***Segment embeddings***: A marker indicating Sentence A or Sentence B is added to each token. This allows the encoder to distinguish between sentences.
  3. ***Positional embeddings***: A positional embedding is added to each token to indicate its position in the sentence.
  ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im36.png) <br/>
  Essentially, the Transformer stacks a layer that maps sequences to sequences, so the output is also a sequence of vectors with a 1:1 correspondence between input and output tokens at the same index. 
  * The first token [CLS] is used in classification tasks as an aggregate of the entire sequence representation. It is ignored in non-classification tasks.
  * For single text sentence tasks, this [CLS] token is followed by the WordPiece tokens and the separator token – [SEP].
  * For sentence pair tasks, the WordPiece tokens of the two sentences are separated by another [SEP] token. This input sequence also ends with the [SEP] token.
  * ***Tokenization***: BERT uses WordPiece tokenization. The vocabulary is initialized with all the individual characters in the language, and then the most frequent/likely combinations of the existing words in the vocabulary are iteratively added.
* ***Pretraining tasks***
  The BERT model is pretrained on two tasks simultaneously:
  1. ***Masked LM(MLM)***: Randomly mask out 15% of the words in the input - replacing them with a
    [MASK] token - run the entire sequence through the BERT attention based encoder and then predict only the masked words, based on the context provided by the other non-masked words in the sequence. Beyond masking 15% of the input, BERT also mixes things a bit in order to improve how the model later fine-tunes. Sometimes it randomly replaces a word with another word and asks the model to predict the correct word in that position.
    ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im37.png) <br/>
  2. ***Next Sentence Prediction(NSP)***: In order to understand relationship between two sentences, BERT training process also uses next sentence prediction. A pre-trained model with this kind of    understanding is relevant for tasks like question answering. During training the model gets as input pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text as well.
   ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im38.png) <br/>
* ***Task Specific Models***: The BERT paper shows a number of ways to use BERT for different tasks.
   ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im39.png) <br/> 
We can use also use BERT for feature extraction just like ELMO.   
* References:
    * https://jalammar.github.io/illustrated-bert/
    * https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c
    * https://www.kaggle.com/code/abhinand05/bert-for-humans-tutorial-baseline
    * https://huggingface.co/blog/bert-101

# BERT limitations
* It struggles to handle negation
* It is very compute intensive while training and inferencing because of many parameters.
* 

# ALBERT(A lite BERT)
The main motivation behind ALBERT was to improve the training(training time) and results of BERT architecture by using different techniques such as factorization of embedding matrix, parameter sharing, and Inter sentence Coherence loss.
  1. ***Cross-layer Parameter sharing***: There are multiple ways to share parameters(in transformer network), e.g., only sharing FFN parameters across layers, or only sharing attention parameters. The default decision for ALBERT is to share all parameters across layers. So, we can say that ALBERT have one encoder layer with different weight and apply that layer 12 times on the input. As a result, the large ALBERT model has about 18x fewer parameters compared to BERT-large. 
  2. ***Embedding Factorization*** : In BERT, as well as later modelling advancements like XLNet and RoBERTa, the WordPiece embedding size E and the hidden layer size H are tied together, i.e., E ≡ H, which is sub-optimal. If E=H, then increasing H increases the size of the embedding matrix, which has size V×E.
  From a modeling perspective, WordPiece embeddings are meant to learn context-independent representations, whereas hidden-layer embeddings are meant to learn context-dependent representations.
    - A more efficient usage is to have H≫E.
    - A factorization of the embedding parameters is proposed in ALBERT, decomposing them into two smaller matrices.
    - By using this decomposition, the embedding parameters are reduced from O(V×H) to O(V×E+E×H). This parameter reduction is significant when H≫E.
  3. ***Inter-Sentence Coherence Loss***: BERT uses NSP loss, later many studies found that NSP’s impact is unreliable and decided to eliminate it. NSP’s ineffectiveness is its lack of difficulty as a task, as compared to MLM. In ALBERT, sentence-order prediction (SOP) is used which only looks for sentence coherence and avoids topic prediction. The SOP loss uses positive examples (two consecutive segments from the same document) and negative examples (the same two consecutive segments but with their order swapped). This forces the model to pick up minute nuances about discourse-level coherence properties.  
  * References:
    - https://sh-tsang.medium.com/review-albert-a-lite-bert-for-self-supervised-learning-of-language-representations-14e1fcc05ba9
    - https://iq.opengenus.org/albert-nlp/
    - https://www.analyticsvidhya.com/blog/2022/10/albert-model-for-self-supervised-learning/
    - https://arxiv.org/pdf/1909.11942.pdf(paper)

# DistilBERT
  DistillBERT is a smaller, faster, and lighter version of BERT which is designed to be more resource-efficient, easier to train and faster inference than BERT, while still maintaining a high level of performance. Knowledge distillation is leveraged during the pre-training phase and it has 40% fewer parameters than BERT, while retaining 97% of its language understanding capabilities and being 60% faster and more efficient to run and deploy in a production environment
  - ***Knowledge distillation and training loss***: In the teacher-student training, we train a student network to mimic the full output distribution of the teacher network (its knowledge). Rather than training with a cross-entropy over the hard targets (one-hot encoding of the gold class), we transfer the knowledge from the teacher to the student with a cross-entropy over the soft targets (probabilities of the teacher). 
  ![](https://github.com/tikna123/natural-language-processing/blob/main/images/im40.png) <br/>
  This loss is a richer training signal since a single example enforces much more constraint than a single hard target. It is also call distillation loss(L_ce).
  The final training objective is a linear combination of the distillation loss L_ce with the supervised training loss, in our case the masked language modeling loss L_mlm. We found it beneficial to add a cosine embedding loss (Lcos) which will tend to align the directions of the student and teacher hidden states vectors.
  - ***Student architecture and Initialization***: It has the same general architecture as BERT. The token-type embeddings and the pooler are removed while the number of layers is reduced by a factor of 2. Most of the operations used in the Transformer architecture (linear layer and layer normalisation) are highly optimized in modern linear algebra frameworks and our investigations showed that variations on the last dimension of the tensor (hidden size dimension) have a smaller impact on computation efficiency (for a fixed parameters budget) than variations on other factors like the number of layers. Thus Number of layers are reduced in the architecture.
  Student architecture is initialized from original BERT architecture by taking one layer out of two.
  - ***Pretraining***: It is trained on very large batches leveraging gradient accumulation(upto 4K examples  perbatch) using dynamic masking and without the next sentence prediction objective. The same corpus as the original BERT model is used for the training.