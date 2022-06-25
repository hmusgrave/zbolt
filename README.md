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


