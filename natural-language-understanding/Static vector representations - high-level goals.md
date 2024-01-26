given a lexicon matrix where each word is classified and have different dimensions, we can classify and vectorize words not existing in the lexicon
![[Pasted image 20240125111947.png]]

## High-level goals
1. Begin think about how vectors can encode the meaning of linguistic units.
2. Foundational concepts for vector-space models (VSMs) a.k.a. embeddings.
3. A foundation for deep learning NLU models.

## Guiding hypotheses
Firth (1957)
> "You shall know a word by the company it keeps."

Harris (1954)
> "distributional statements can cover all of the materiel of a language without requiring support from other types of information"

Wittgenstein (1953)
> "the meaning of a word is its use in the language"

Turney and Pantel (2010)
> "If units of text have similar vectors in a text frequency matrix. then they tend to have similar meanings"

## Latent Design
- tokenization
- annotation
- tagging
- parsing
- feature selection
- cluster texts by date/author/discourse context/...
- matrix design
	- word x document
	- word x word
	- word x search proximity
	- adj. x modified noun
	- word x dependency rel.
- reweighting
	- probabilities
	- length norm.
	- TF-IDF
	- PMI
	- positive pmi
	- ...
- dimensionality reduction
	- lsa
	- plsa
	- lda
	- pca
	- nnmf
	- ...
- vector comparison
	- euclidean
	- cosine
	- dice
	- jaccard
	- kl
	- ...
Nearly the full cross-product to explore; only a handful of the combinations are ruled out mathematically. Models like GloVe and word2vec offer packaged solutions to design/weighting/reduction and reduce the importance of the choice of comparison method. Contextual embedding dictate many preprocessing choices.