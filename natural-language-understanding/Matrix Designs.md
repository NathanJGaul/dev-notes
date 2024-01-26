## word x word
![[Pasted image 20240125115221.png]]

Words appearing near other words
### Properties
1. Very dense
2. Dimensionality will remain fixed

## word x document
![[Pasted image 20240125115320.png]]

Words occurring within a document
### Properties
1. Sparse (most words don't appear in most documents)
2. Dimensionality will change with new document

### word x discourse context
![[Pasted image 20240125115551.png]]

Word occurrences during dialog category

## Other designs
- adj. x modified noun
- word x syntactic context
- word x search query
- person x product
- word x person
- word x word x pattern
- verb x subject x object
- ...
- how does your design align with the modeling problem you are exploring?

## Feature representations of data
- how are you modeling your sentences into a vector

## Windows and scaling: What is a co-occurence?
![[Pasted image 20240125121115.png]]
- window: what distance from the focus word is considered co-occurence
- scaling: how do we scale the distance from the focus word in the window
	- flat: equal weight
	- 1/n
- larger, flatter windows capture more semantic information
- small, more scaled windows capture more syntactic (collocational) information
- Textual boundaries can be separately controlled; core unit as the sentence/paragraph/document will have major consequences for downstream tasks involving the representations you create
