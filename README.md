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
