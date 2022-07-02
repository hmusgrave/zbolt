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
idea based on seed splitting with an array syntax.

Jax:
(1) Threefry counter PRNG
(2) A functional array-oriented splitting model

Threefry:
random123sc11.pdf

The threefry paper includes several other PRNG's (we should probably implement
them all?) and its own splitting model. Threefry is fast but not
cryptographically secure per se. It does pass a lot of tests indicating monte
carlo suitability. Other PRNG's are secure and reasonably fast. I know Jax's
source (2) spends a lot of time covering the API and a little describing why
their splitting model generates better random numbers than alternatives, so
we'll need to check if that differs from this source substantially.

The model is
- f: S -> S
- g: K x Zj x S -> U

S -- internal state space
U -- key space
K -- integer output multiplicity
Zj -- extract a small number of output values from each internal state

f -- transition function
g -- output function

Traditional PRNG's have a complicated f and trivial g. The design space here
uses a simple f and complicated g. In particular,

g(k,j) = h(j) o b(k)

for a keyed bijection b(k) (i.e., b is a family of functions indexed by k, each
of which is a bijection of some suitable range of integers), and a simple
selector h(j).

S will consist of p-bit integers. The selector h(j) chooses the jth r-bit block
with r<=floor(p/J). The state transition function can be as simple as

f(s) = (s+1)mod2^p

and so is referred to as a counter, though it may be more complicated

A natural API for counter-based PRNGs provides the keyed bijection b(k) but
leaves the application in complete control of the key k and the counter n.
Often applications will have multiple objects, multiple object types, multiple
threads, or other natural ways to generate keys, and with a 128-bit key space
it's trivial to reserve 32+ bits for the dedicated purpose of ensuring
uniqueness. Applications are in control of avoiding key/seed reuse.

ciphers:

AES:
- Supported natively for a decade
- uses a pre-computed set of 11 "round keys"
- 10-round iterated block cipher

Threefish:
- Like AES
- Software-friendly P blocks
- 72 rounds

ARS:
- Round keys are generated using 64-bit addition on the separate high and low
  halves of the 128-bit ARS key
- (Eqn 6.) rk0 = user_key, rki = rk(i-1) + constant
- constant is fairly arbitrary, verified with (sqrt(3)-1) and the golden ratio.
- ARS-5 is crush resistant, but ARS-4 is not.

Threefry:
- Threefry-NxW-R has the same pattern as Threefish with R rounds and N W-bit
  inputs and outputs.
- Threefry ignores the "tweak" (an extra 128-bits of key-like input)
- For W=64 and N>=4 Threefry takes the rotation constants from Threefish.
- For other parameters, we generate Threefry's rotation constants using the
  code provided with the Skein reference materials.

Round 4x32 unknown 2x64
0     10   26      16
1     11   21      42
2     13   27      12
3     23    5      31
4      6   20      16
5     17   11      32
6     25   10      24
7     18   20      21

The rounds are mod 8. All of 2x64-13, 4x32-12, 4x64-12, and 4x64-72 are crush
resistant. Presumably the middle column is 4x64?

I suppose we can figure out how to generate this table on our own.

Philox:
L' = Bk(R) = mullow(R, M)
R' = Fk(R) +%2 L = mulhi(R,M) +%2 k +%2 L

- For N=2, Philox-2xW-R performs R rounds of the Philox S-box on a pair of
  W-bit inputs.
- For larger N, the inputs are permuted using the Threefish N-word P-box
  before being fed, two-at-a-time, into N/2 Philox S-boxes, each with its own
  multiplier M, and key k.
- Then N/2 multipliers are constant from round to round, while the N/2 keys are
  updated for each round according to the weyl sequence defined in Eqn 6.

4x32: 0xCD9E8D57, 0xD2511F53
2x64: 0xD2B74407B1CE6E93
4x64: 0xCA5A826395121157, 0xD2E7470EE14C6C93

Splitting Model
===============

I guess the synopsis at the end of the Threefry paper settles the question of
whether the splitting paper does anything substantially different. The answer
is definitely yes. The Threefry paper leaves control over state completely to
the application developer, and we're going to build on top of that with an
appropriate splitting model.

I'm pretty sure oopsla14.pdf is different from what Jax themselves use, but it
looks friendly enough.

To create a new PRNG, simply use an existing PRNG instance to generate a new
seed and a new gamma value pseudorandomly. In practice they tweak this idea
slightly. First let's back up and see what gamma is, then let's see their
tweaks.

notes: They have some measure of proof that such a simple idea
probabilistically generates independent streams.

So the gamma comes from DOTMIX:
- When PRNGs are split, you can conceptually keep track of an index associated
  to each branch
- Each PRNG is uniquely determined by that set of indices
- We want to dot those indices with a random value gamma, add a seed sigma, and
  run it through a mixing function.

notes: It looks like the mixing function produces effectively a starting point
for our pseudorandom cycle.

notes: they talk about everything happening modulo a large prime. Later they
indicate that you lose 10 or so bits of security if you use a power of 2
instead. We'll do that for the shift efficiency.

SplittableRandom (SPLITMAX):
- Two 64-bit fields, seed and gamma
- Gamma must be odd to keep a high period.
- The sequence of seeds is (seed + n * gamma)
- A mix function produces a result
- We can actually do away with the dotmix strategy. This gamma is something
  different.
- The root default gamma value is the closest odd integer to 2^64/phi (phi ==
  golden ratio)

Both Random APIs
================

Combining those two ideas is relatively simple. The state update function is
(s, g) -> (s+g, g)%p. We choose J=1. The mixer functions work exactly as in the
Threefry paper (and novel hashing functions were one of their signature
contributions, we have a todo above use for handling that). Then we
additionally have a splitting model for probabilistically generating
independent streams.

To keep with jax's API, we'll never explicitly update the seed. We either use
the state to produce random values, or we split it into n values by creating
the sequence of n seeds and using each of those to generate new PRNGs.

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

Addendum
========
- For ML, I imagine that the transform of a set of centroids is a great
  starting point for the next layer's centroids to accelerate kmeans.
- For a few layers, you can probably (sometimes) get away with literally just
  transforming the centroids and keeping the same input data.
- In the above example, the idea is that the input matrix would be encoded (as
  opposed to weight matrices), because it tends to remain fixed over a training
  period.
