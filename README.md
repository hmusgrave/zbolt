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
