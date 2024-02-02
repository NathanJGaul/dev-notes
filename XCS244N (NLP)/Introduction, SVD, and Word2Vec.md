Word Vectors: used to determine similarity and distance between words
- using distance measures such as Jaccard, Cosine, Euclidian, etc.

Denotational Semantics: The concept of representing an idea as a symbol (word or a one-hot vector). It is sparse and cannot capture similarity. This is a "localist" representation.

## SVD Based Methods
1. first loop over a massive dataset and accumulate word co-occurrence counts in some form of a matrix $X$
2. then perform Singular Value Decomposition on $X$ to get a $USV^T$ decomposition
3. then use the rows of $U$ as the word embeddings for all words in our dictionary

### Word-Document Matrix
- bold conjecture: words that are related will often appear in the same documents
$X$ in the following manner:
1. loop over billions of documents and for each time word $i$ appears in document $j$, we add one entry $X_{ij}$ 
- very large matrix $\mathbb{R}^{|V| \times M}$ and scales to the number of documents

### Window based Co-Occurrence Matrix
- count the number of times each word appears inside a window of a particular size around the word of interest
- count for all words in the corpus
- $|V| \times |V|$ co-occurrence matrix $X$
- apply SVD on $X$ to get $X = USV^T$
- select the first $k$ columns of $U$ to get $k$-dimensional word vectors

## Iteration Based Methods - Word2Vec
- design a model whose parameters are the word vectors
- then train the model on a certain objective
- at every iteration run our model
	- evaluate the errors
	- follow an update rule that has some notion of penalizing the model parameters that caused the error
- the model learns the word vectors

### Word2Vec
- 2 algorithms
	- continuous bag-of-words (CBOW): predict a center word from the surrounding context
	- skip-gram: predicts the distribution (probability) of context words from a center word
- 2 training methods
	- negative sampling: defines an objective by sampling negative examples
	- hierarchical softmax: defines the object using a efficient tree structure to compute probabilities for all the vocabulary

### Language Models
- create such a mode that will assign a probability to a sequence of tokens
	- "The cat jumped over the puddle." should be given a high probability
	- "stock boil fish is toy" should have a low probability
#### Unigram model:
- $P(w_1,w_2,\dots,w_n) = \displaystyle\prod^n_{i=1}P(w_i)$
- break apart the probability by assuming the word occurrences are completely independent
	- this is a bit ludicrous because we know the next word is highly contingent upon the previous sequence of words
#### Bigram model
- $P(w_1,w_2,\dots,w_n) = \displaystyle\prod^n_{i=1}P(w_i|w_{i-1})$
- probability of a word depends on itself and the word next to it
- bit naive approach since it depends on pairs of neighboring words

#### Continuous Bag of Words Model (CBOW)
- known values: sentence represented by one-hot word vectors
	- input: one hot vectors (context) $x^{(c)}$
	- output: one hot vectors $y$
- unknowns: matrices $V \in \mathbb{R}^{n \times |V|}$ and $U \in \mathbb{R}^{|V| \times n}$ where $n$ is an arbitrary size which defines the size of our embedding space
	- $V$ is the input word matrix such that the $i$-th column of $V$ is the $n$-dimensional embedded vector for word $w_i$
		- this $n \times 1$ vector is denoted as $v_i$
	- $U$ is the output word matrix where the $j$-th row of $U$ is an $n$-dimensional embedded vector for word $w_j$
		- denoted as $u_j$
	- the model learns these two vectors for every word $w_i$
- objective function: cross-entropy
	- $H(\hat y,y) = -y_i log(\hat y_i)$
	- perfect case: ($\hat y_c = 1$ and $y_c = 1$) $H(\hat y, y) = -1 log(1) = 0$
	- bad case: ($\hat y_c = 0.01$ and $y_c = 1$) $H(\hat y, y) = -1 log(0.01) \approx 4.605$

### Skip-Gram Model
- given the center word, the model will be able to predict the surrounding words
- the jumped word is know as the context

### Sampling Methods

#### Negative Sampling
- instead of looping over the entire vocabulary, we can just sample several negative examples
	- sample from a noise distribution ($P_n(w)$) whose probabilities match the ordering of the frequency of the vocabulary
- works better for frequent words and lower dimensional vectors

#### Hierarchical Softmax
- tends to be better for infrequent words
- uses a binary tree to represent all words in the vocabulary
	- no output representation for words
- each node of the graph (except the root and the leaves) is associated to a vector that the model is going to learn