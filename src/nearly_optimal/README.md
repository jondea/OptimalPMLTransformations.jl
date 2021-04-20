# Nearly optimal transformations

Optimal transformations (transforming the field into a linear variation) are
expensive to calculate and can be expensive to integrate, particularly if they
have rips.
Furthermore, if used in an iterative scheme, they are only optimal for the
approximation to the field on the previous iteration.
In this sense, they are unlikely to be truly optimal for the exact field, or
the discretised field we are currently solving for.

With this in mind, can we take a more pragmatic approach?
By limiting ourselves to a basis of well behaved (easy to integrate) functions,
can we find a nearly optimal transformation which will perform nearly as well
at a much smaller computational cost?
We could do this by minimising some cost function.
Inspired by Z2 error estimation, and optimal transformations, we could minimise
the second derivative of the transformed solution through the PML.
