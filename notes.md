
IF a feature is split too wide,
then the importance calculating becomes 0?



Story:
Increasing rank of k, is actually not very sensitive to hyperparameters on the baseline model;
	-Goes up to rank 6 and still performs well, this is however where all values are summed up
	- Instead of being passed into a k dimensional vector.
It does however not cause the clustering behaviour that we seek;
	- but it does make them more sparse, so its maybe towards the right direction?
	- It does however not give the clustering behaviour that we intend, nothign tells it to group these 3 specific things for example.

Also: Whole layer2 is causually important for everything..

softmax didnt work, very well. It pushes things towards one hot encoding
do i want that?

Something to try, attend attention only to same layer.

Mention experiments with what happens with global attention?

attention moves it to diagonal, so does softmax
two alternative methods:
	-Structural clustering
	-Finding the part of the weight matrix that encodes a given feature

perhaps section with contrast to the two forms
what is the best form of clustering, one per independent or one that encourages clustering?

Second part of story -> is this enough to say we want to do softmaxing?



Full story revised:

Show SPD on the toy model of gemoetry
	-It gives good clustering, however the features themselves are not sparse
		-Gives to thinking what the true decomposition should be, if each feature is part of a point, then it by itself is not very telling
			-Or an even better one would maybe be the entire simplex?
				-This way we dont need to know the structure in advance to find it.
		- In order to capture this sparsely we need to increase the rank of k
			-Increasing rank of k, allows capturing more features
				- However it does not gurantee clustering same features
		- We try to encourage clustering within the same component, one way of doing so is through softmaxing?
			-Softmax encourages one hot, however is that the signal we want?
			- Lasso as alternative approach
		- Lastly we want to make k an upper bound, per feature ablation
	
	-Somewhere inside of this we rewrite the whole thing as "SVD", then we can take D * D^T for visualization, and the other one for output visualization.


Use the word atom in there somewhere.

we hypothesize that the lack of sparsity could be a consequence of the individual atoms not being large enough? -> multi dimensionality 

Also want to try something with passing in smaller and smaller components.
-
Also the issue that we decompose 4 identical things but only get 4 similar returns.

Also add a section on how the very wide layer in SPD turns off, when its large enough
also check how SPD works on different losses, does it keep the clustering on geometry.


code seems very sensitive to hyperparameters,
0.0001 gives good result, 0.0005 does not.

add a section to replacing minimality to something more stable
    -Only get a decomposition at 0.0001...


softmax is super promising here on the two pair TMS, it decomposes the underlying data into the same direction, show cosine similarity and thus they can be decomposed itno rnak 1, with little loss
but we get co-occurances