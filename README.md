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
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im1.png) <br/>
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im2.png) <br/>
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im3.png) <br/>
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im4.png) <br/>
    ![](https://github.com/tikna123/machine-learning-concepts/blob/main/images/im5.png) <br/>
