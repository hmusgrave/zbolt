I skimmed 1706.10283.pdf (arxiv) and thought the results sounded meaningful and
the techniques up my alley. I know basically nothing about vector compression,
but how hard could it be :) The plan is to bounce back and forth between Python
and here to figure out exactly what's going on and then write the whole thing
in Zig.

Given
=====

q: (J,)
query vector

X: (N,J)
database vectors

d: (q, x) -> f(sum(starmap(δ, zip(q, x))))
"distance" function form

Task
====

g: R^J -> G

h: R^J -> H

d': GxH -> R

For a given L2 distance between d'(g(q), h(x)) and d(q, x), minimize the
computation time
    T = Tg + Th + Td

Tg: time to encode received queries using g

Th: time to encode the database X using h

Td: time to compute the approximate distances between the encoded queries and
encoded database vectors

Notes
=====

- It looks like we're explicitly ignoring the time to arrive at g, h, and d'.
    - That assumption is confirmed in the next paragraph.
- Indexing (Inverted Multi-Index, Locality-Sensitive Hashing tables) looks to
  be orthogonal to this work. That probably means G and H are also vector
  implicitly spaces.

Related Work
============

Main approaches:
(1) Binary Embedding
(2) Vector Quantization

Binary:
- map to a Hamming space
- fast for obvious reasons
- sub-optimal error for given code length

Quantization:
- several approaches (outlined below)
- slower
- more efficient error/length

Popular quantization approaches:
- map to k-means centroid
- Product Quantization generalizes that to operate on disjoint slices
- Many generalizations of Product Quantization
- In general those all rotate the vectors or relax the "disjoint" constraint,
  and the name for those techniques is "Multi-Codebook Quantization"
- A noteworthy hybrid allows Binary embeddings to be refined to Vector
  Quantization, designed to allow for fast pre-filtering while retaining high
  accuracy (presumably at some storage cost?)

Other "approaches":
- ML usually applies custom insights to the model structure, and those are
  supposedly complementary with this paper
- This paper is most similar to a light extension of PQ (reference [3] of their
  paper...I'll descend that rabbit-hole later if needed)
- Structured matrices can be extremely valuable. The paper didn't talk about it
  much, but the idea is to recast something that looks like a dense matrix
  multiplication (usually O(N^2.7ish), a vectorized O(N^3), maybe as low as
  O(N^2) eventually but nobody knows, really slow O(N^2.35ish) algorithms
  exist, ....) as some combination of operations on structured matrices (FFT,
  Hadamard, ...) which represent a lot of problems and can have their matrix
  multiplies implemented in O(N) with some log factors. There was a recent
  paper talking about how all the "fast" ML transformer architectures were some
  specific instance of a certain set of structured matrices.
- Model compression basically truncates part or all of various weights.

Method
======

(1) Partition the indices j into M disjiont subsets

(2) Each x(m) is stored as i, representing ci(m): the codeword corresponding to
    i in that codebook

Let C = {C1 ... CM} be the set of M codebooks, where each codebook is itself a
set of K vectors {c1(m) ... cK(m)}

notes: That's then a rank 3 tensor? (M, K, Z) -- Z seems unintroduced and might
vary across K. The location in the vector determines which codebook to use, and
which code_word_ you're closest to determines your index.

(3) So how do we map the queries somewhere reasonable?

G = R^(KxM)
g(q): (i,m) -> δ(q(m), ci(m))

I.e., the i'th row of g(q) is the coordinate-wise distance between the query in
that partition and the codeword for that row and partition. We're just
precomputing all the distances to the entire codebook.

notes: The paper has bolding to indicate vectors. δ is actually bolded in that
g(q) definition, as are its arguments, and this indicates that its arguments
are vectors and that δ horizontally sums them. There still isn't any evidence
that Z need necessarily be constant-width.

(4) Then the estimated distance function is just to go through each i in h(x),
add the associated entry in g(q) to a running total, and return the result.

Extended Method
===============

The above description is just product quantization. Bolt (the referenced paper)
has two additions:

(1) Use smaller codebooks than normal (16 centroids)

(2) Approximate the distance matrix D

The first is just a design parameter so I won't touch on it much (though it's
important in how it enables easy vectorization), but the distance matrix
approximation seems interesting.

- We're learning 8-bit quantizations of the entries of D. I'll wait to see what
  happens when we get into the code, but I think AVX2 still reigns supreme for
  most CPU workloads, and it seems like Bolt might be optimized moreso for SSE.
  Alternative parameters might perform better.
- (quote) We can leverage vector instructions to perform V lookups for V consecutive
  h(x) where V = 16, 32, 64 depending on the platform. It looks like this is
  considering analyzing a single input x at a time and considering all its
  constituent parts.
- The quantization function is learned at training time. For a given column of
  D across multiple query vectors qm, we wish to minimize the quantization
  error (MSE of the estimated distances vs real distances, assuming a linear
  column-wise reconstruction).
- Bm(y) = max(0, min(255, [ay-b]))
- y' = (bm(y)+b)/a
- Proposal to set b=F^-1(alpha), a=255/(F^-1(1-alpha)-b) for some alpha. F^-1
  is the inverse CDF of Y, estimated emperically (Y is the distribution of
  distances when looking at a column of D for a set of query subvectors and the
  distribution X of database vectors).
- Note that Y can be estimated extremely cheaply. There aren't many ci vectors,
  so you can maintain a frequency for each of those 16 vectors across X and
  analyze all the query vectors crossed with those 16. A random sampling should
  suffice (this isn't an exact science anyway), and the literature is full of
  fast quantile estimation.
- They used a grid search over {0, .001, .002, .005, .01, .02, .05, .1} for
  alpha.
- The alpha values are shared across D tables (not across subvectors). The b
  values are learned individually.

notes: I don't understand something about this. There's a fixed alpha we can
find with a grid search, and that's moderately expensive. Then b and a are
defined precisely with respect to that alpha. Supposedly though, b values are
table-specific. And there's a table per query.

notes: I looked at their python source. Z is fixed via padding.

notes: Their source also _only_ learns offsets at training time and doesn't
require the same exact queries at serving time to use those. I apparently have
no idea how query encoding works.

notes: Their code is doing something awful. There's some _other_ lookup table
thingamajig somewhere else only called deep in some spaghetti code whose result
is never used. I don't know this is a great way to figure out what's going on.

notes: They indicate that a has to be shared whereas b can't because the error
from b can be corrected later but it doesn't make sense to rescale distances
(weighting each table differently). I don't think that makes sense though. The
intent of all of these is to approximate the distance matrix as well as
possible. If some columns have lots of distances and others don't have many
you'll get better representational accuracy if those are spread out.

notes: All right...I think I understand a little bit. Having the same `a`
scaling allows you to mix and match distances _before_ applying the scaling and
save some vectorized multiplies. Then you can have different `b` offsets
trivially because they can be pre-accumulated into a correction factor applied
to the final result.

notes: I was concerned about the beta/alpha computations and how those seem to
rely on global information across all queries while
simultaneously...not...doing that. Their source code does something very
different from what their paper does:

(1) Each table has its distances (for a single subvector location) percentiled to
choose the floor values, and the results are clipped.

(2) _Those_ percentiles are percentiled to produce the ceiling. I think this is
part of why the authors found lower percentiles to work best -- there's a
compounding effect of sequential percentiles that makes the ceiling further
than one might guess.

(3) The returned values are: floors for each LUT, a single scalar for all LUTs,
and also the chosen alpha (presumably used for other incoming queries?).

(3a) alpha is ignored...

(4a) There's a set_data function that persists the data we're encoding,
separately from what was "fit".

(4b) Oh god, that's all done in cpp.

(4c) Oh god, it's just spaghetti globals.

(4d) There's no way that's super important.

notes: I don't see any good reason why `a` shouldn't be different for each LUT
(shared across subvectors, disjoint for separate queries). It's a negligible
memory increase compared to the LUT sizes, and it's hardly an additional
computational cost at all (perhaps more expensive data loading) since _some_
multiplication still has to happen per query anyway.

notes: For many use cases I'm not sure how beneficial the percentile grid
search is. I'm personally often working with matrices smaller than 2^10 or so,
so even with quite a few centroids there would literally be no difference
between {0, .001, .005, .01}. Taking the smallest and largest values is
probably more than sufficient. That then encodes, for each query, a single
scalar, a single offset, and its discretization. The process is not even a
little bit reversible in the sense that you can't compute the distances of
subvectors anymore, but it's lightning fast and almost as accurate.

I think we're going to do the first implementation in Zig. I see no good reason
to play with Python first. This seems simple enough.

K-Means
=======

We probably ought to figure out how this works and devise a suitable
initialization scheme.

- It looks like kmeans++ is pretty good.

(1) Choose a center uniformly at random

(2) For each data point not chosen yet, compute D(x), the distance between x
and the nearest center that has already been chosen

(3) Choose one new data point at random as a new center, using a weighted
probability distribution where a point x is chosen with probability
proportional to D(x)^2

(4) Repeat (2-3) till k centers have been chosen

(5) Proceed with some kmeans iteration procedure

notes: 

(6) If the vectors are even a tiny bit nontrivial in size we probably want to
store computed distances to centers.

(7) There's a meta-procedure for initialization where you apply a random
partition, cluster inside there, and pairwise-nearest-neighbor the clusters.
Especially when applied recursively, this basically gives you more chances at
reasonable initializations but only on a small subset of the data so you retain
linear costs.

(7a) IMO you'd probably want to run a few optimization passes after doing that.

- And what exactly do we do for the update scheme? Supposedly there are things
  much faster than the naive approach.

To start with, apparently you get decent results by starting with a subset of
the data. Actually in hindsight, this is basically what the meta-procedure is
doing.

Loyd's Algorithm

(1) Calculate centroids

(2) Assign to nearest centroid

(3) Repeat (1-2) till assignments don't change.

Supposedly you can get significant speedups via KD-Trees. For 16 centroids I'm
not sure it's a huge win. At a minimum you have 5 distance computations just
for naive checks on the tree. This on average roughly doubles as you
recursively check the other plane halves, so on average you save roughly 6
distance computations and pick up a fair bit of other overhead. It might be a
little faster for us, but it's a lot of code I don't really want to write.

Random Seeds
============

Fine. I guess we need a way to generate nice random numbers. Jax has a nice
idea based on seed splitting with an array syntax. TODO.

Pairwise Nearest Neighbor
=========================

Can we avoid the quadratic cost?

Yeah, probably. For each point we can easily compute the nearest neighbor in N
log(N) time. N is only 32, so that's probably dominated by overhead and whatnot
if we're not careful. Might have to hand-tune this one.

Nearest Neighbor isn't symmetric, so we can't just blindly merge those.

Grr. Apparently PNN is lower-bounded by N^2. Approximations might suffice, but
we have to bite the bullet I think.

Blah blah, "Practical methods for speeding-up the pairwise nearest neighbor
method"

d(a,b) = n(a)n(b)/(n(a)+n(b))||c(a)-c(b)||^2

i.e., merging two clusters is roughly as expensive as the distance between
their centers scaled by a sort of mean of the number of interactions divided by
the number of elements.

(1) Each cluster has a pointer to its nearest neighbor (minimizing the merge
cost)

(2) Greedily merge the two closest clusters

(3) Repeat (2) as necessary, updating invariants

(4) Distances never decrease from a merge, so you can delay updating pointers
so long as it never becomes a candidate for being the smallest distance.

(5) 

Eh, this is all garbage. Why PNN? Let's just KNN the centroids. Partitions are
approximately equally sized, so this roughly partitions the space into the
number of elements we want while not magnifying the effects from merging
disparate clusters. Initialization isn't super important because the average
cluster size will be 2, so the space will be reasonably well covered by any
random selection.
