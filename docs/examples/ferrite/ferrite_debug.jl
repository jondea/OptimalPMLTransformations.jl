
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(@__DIR__)
    # instantiate, i.e. make sure that al packages are downloaded
    Pkg.instantiate()
    using Tensors
    using Tau
    using SparseArrays
	using SpecialFunctions
    using LinearAlgebra
	# using OptimalPMLTransformations
	using StaticArrays
	using IterTools
	using OffsetArrays
end

using Ferrite

using OptimalPMLTransformations

using PMLFerriteExtensions

# Single case for debugging
# solve_and_save(;k=0.1, N_θ=3, N_r=1, N_pml=1, cylinder_radius=1.0, R=2.0, δ_pml=1.0, n_h=3, order=2, folder=tempname("./"))

include("pml_helmholtz_polar_annulus_assembly.jl")

k = 0.1
n_h = 0
R = 2.0
N_θ = 2
N_pml = 0
N_r = 2

int_order = 2
δ_pml=1.0
cylinder_radius = 1.0
nqr_1d = 2*(int_order + 1)
bulk_qr=QuadratureRule{2, RefCube}(nqr_1d)
pml_qr=QuadratureRule{2, RefCube}(nqr_1d)
# u_ana=HankelSeries(k, OffsetVector([1.0], n_h:n_h))
# u_ana=two_mode_pole_series(k, R + (1.0+1.0im))
a = scattered_coef(-10:10, k)
u_ana = HankelSeries(k, a)

pml = InvHankelSeriesPML(AnnularPML(R, δ_pml), u_ana)
result = solve(PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order=int_order, pml, bulk_qr, pml_qr))
@show result

# (θ_min, θ_max) = (0.0, τ)
# intrp = interpolation(PMLFieldFunction(pml), range(θ_min, θ_max, length=21))

