1. It seems to learn something (TMS has a clear residual)
2. Show the other model, has a clear signal that dissapears after 5k, need to find causes
3. Currently the gnan only has 1 feature, the inner activation, likewise gnan must be modified to handle batches?
    3.1 hence i'm summing over the batch right now, until next week atleast.
4. Show the visualizations from gnan, it seems to not learn much?
    4.1 Likewise fewer features, again mean that its a bit uninteresting....
5. crosscoder???
6. Set batch size to 1?
7. Have a look at the batched version from GNAN
8. What am i doing with AB * x, lukas said to increase the rank this gives more dimensions.
9. i.e Take A * x, and B*x, increasing the rank of A and B would create a higher dimensional feature.
10. It ONLY USES THE A NOT AB -> Increase the rank. Less interpretable but maybe better?



implement gnan
statistical framework deadsalmon
bring down time complexiy gnan adjacency
playing with the k value in the rank dimension.
playing with forward only edges or edges in both direction.







maybe????
try a toy model that uses the original activations, to avoid low rank?