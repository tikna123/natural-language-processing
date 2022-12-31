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
* Attention
* Transformer
* BERT
* ALBERT
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
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im6.PNG) <br/>
The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.
* ***Model Details***
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im7.PNG) <br/>
The above is the architecture of 1 hidden layer neural network. 
* Let's consider we have 10,000 uinque words in the vocab. Here we represent any input work like "machine" as a one-hot vector. The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, the probability that a randomly selected nearby word is that vocabulary word. There is no activation function on the hidden layer neurons, but the output neurons use softmax.
* When training this network on word pairs, the input is a one-hot vector representing the input word and the training output is also a one-hot vector representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating point values, not a one-hot vector).
* ***Hidden Layer***
* Let's say we are trying to learn word vectors with 300 dimension. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron). If you look at the rows of this weight matrix, these are actually what will be our word vectors!
![](https://github.com/tikna123/natural-language-processing/blob/main/images/im8.PNG) <br/>
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

## Recurrent Neural Network

