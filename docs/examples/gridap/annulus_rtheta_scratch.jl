begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    # Pkg.instantiate()
	using PlutoUI, PlutoLinks
	using LinearAlgebra
	using SpecialFunctions
	using Gridap
	import Gridap: Point
    using Gridap.FESpaces
    using Gridap.ReferenceFEs
    using Gridap.Arrays
    using Gridap.Geometry
    using Gridap.Fields
    using Gridap.CellData
    using Tau
    using Colors
	using GridapGmsh
	using StaticArrays
	using GLMakie,GridapMakie
end

using OptimalPMLTransformations: AnnularPML, SFB

k = 1.0

λ = 2π/k

R = 4.0          # Width of the area
r_c = 1.0          # Radius of the cylinder

δ_pml = 1.0      # Thickness of the PML

N_H = 1          # Order of hankel function/angular fourier mode

resol = 10.0      # Number of points per wavelength in mesh

l_c = λ/resol      # Characteristic length

model = CartesianDiscreteModel(CartesianDescriptor(Point(r_c,0.0), Point(R+δ_pml, τ), (5,3); isperiodic=(false, true)))

# let
# 	Ωp = Triangulation(model) # No method yet to plot CartesianGrid?
# 	fig = plot(Ωp)
# 	wireframe!(Ωp, color=RGBA(0,0,0,0.4), linewidth=0.5)
# 	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
# 	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
# 	fig
# end

order = 1

reffe = ReferenceFE(lagrangian,Float64,order)

V = TestFESpace(model,reffe;conformity=:H1, dirichlet_tags=["boundary"], vector_type=Vector{ComplexF64})

u_D(x) = exp(im*N_H*x[2]) * (x[1] < r_c*1.1)

U = TrialFESpace(V,u_D)

degree = order*2

Ω = Triangulation(model)

dΩ = Measure(Ω,degree)

pml = SFB(AnnularPML(R, δ_pml),k)

polarJ = x -> begin
	r = x[1]
	return TensorValue{2,2,Float64}(1,0,0,1/r)
end

polarDet = x -> x[1]

b(v) = 0

# h0 = hankelh1(N_H,k*r_c)

# uh_t = CellField(x->exp(im*N_H*x[2])*hankelh1(N_H,k*x[1])/h0, Ω)

# uh_t.cell_field

# print(uh_t)

# ### Compare the difference in the bulk
function Bulk(x)
    if x[1]<=R
        return 1
    else
        return 0
    end
end

import OptimalPMLTransformations: jacobian, tr_and_jacobian

function J(x)
	if x[1] <= R
		return TensorValue{2,2,ComplexF64}(1,0,0,1)
	else
		rθ = PolarCoordinates(x[1], x[2])
		tr, j = tr_and_jacobian(pml, rθ)
		return TensorValue{2,2,ComplexF64}(j[1,1], j[1,2], j[2,1], j[2,2])
	end
end

# Plain Helmholtz: a(u,v) = ∫( ∇(v)⋅∇(u) - k^2*(v*u))*dΩ
# a(u,v) = ∫( ( ((inv∘J)⋅polarJ⋅∇(v))⋅((inv∘J)⋅polarJ⋅∇(u)) - (k^2*(v*u)) )*polarDet*(det∘J))*dΩ
a(u,v) = ∫(
    ( ((inv∘J)⋅(polarJ⋅∇(v)))⋅((inv∘J)⋅(polarJ⋅∇(u))) - (k^2*(v*u)) )*polarDet*(det∘J))*dΩ


# # This block is from AffineFEOperator
# du = get_trial_fe_basis(U)
# dv = get_fe_basis(V)

# We just need to modifiy this bit
# matcontribs = a(du,dv)
# veccontribs = b(dv)

# uhd = zero(U)

# data = Gridap.FESpaces.collect_cell_matrix_and_vector(U,V,matcontribs,veccontribs,uhd)
# A,b_vec = assemble_matrix_and_vector(assem,data)
# #GC.gc()

# op = AffineFEOperator(U,V,A,b_vec)

# Try to do it all a bit lower level
Tₕ = Ω
Qₕ = CellQuadrature(Tₕ,4*order)
qₖ = get_data(get_cell_points(Qₕ))

du = get_trial_fe_basis(U)
dv = get_fe_basis(V)

# Overload this integrate function?
poison_integrand = ∇(dv)⋅∇(du)
jac = integrate(poison_integrand,Qₕ)

uh = solve(op)

# let
# 	fig = plot(Triangulation(model), real(uh), aspectratio=1.0)
# 	# wireframe!(Ω, color=RGBA(1,1,1,0.1), linewidth=0.5)
# 	# cam2d!(fig.figure.scene, reset = Keyboard.left_control)
# 	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
# 	fig
# end

# writevtk(Ω,"annulus_rtheta",cellfields=["Real"=>real(uh),
#         "Imag"=>imag(uh),
#         "Norm"=>abs2(uh),
#         # "Arg"=>angle(uh),
#         ])

# Difference=sqrt(sum(∫(abs2(uh_t-uh)*Bulk)*dΩ)/sum(∫(abs2(uh_t)*Bulk)*dΩ))

# let
# 	fig = plot(Ω, abs(uh - uh_t)*Bulk, colormap=:heat)
# 	# Colorbar(fig[1,2], plt)
# 	# wireframe!(Ω, color=RGBA(1,1,1,0.1), linewidth=0.5)
# 	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
# 	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
# 	fig
# end
