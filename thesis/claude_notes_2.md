Change the distance function to be multi parameter,
both the layer distance and the cosine similarity.
This way we get different results for neurons in the same layer.


Add rank constraint, to make it try to push it into rank 1 components,
where fitting. (Analogus to the spectral norm from APD)
Or add per column masking?


https://arxiv.org/pdf/2502.04878
Sparsity loss in SAE, encourages combined features
-> Same problem foru s?


Note that if we can train a larger k feature, concept
then we can atleast cluster within the same layer
without any issues.