
# Optimal PML Transformations

An optimal transformation for a given field and PML geometry is one which
transforms the field to have a linear variation through the PML.
This repo is a Julia implementation of this idea, which was described in my
[PhD thesis](https://www.research.manchester.ac.uk/portal/en/theses/optimal-pml-transformations-for-the-helmholtz-equation(2617fdfb-06e9-4fbf-9bfc-934f6b361572).html) submitted to the University of Manchester.

## Get started
To install this package, call
```julia
import Pkg
Pkg.add("https://github.com/jondea/OptimalPMLTransformations.jl")
```
or alternatively type `] add https://github.com/jondea/OptimalPMLTransformations.jl`
in the REPL.
