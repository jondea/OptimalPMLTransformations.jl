### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 2b1f6cbc-7f38-4573-a722-22668837de03
using Gridap

# ╔═╡ 5acadaf2-0c4c-494c-a305-27b1c41cc7a2
using GridapGmsh

# ╔═╡ e10c5855-3713-4a81-b071-8fe14c3aa0af
using Gridap.Fields

# ╔═╡ 1d772e87-2143-498f-bdf0-b1594b8d31e3
using Gridap.Geometry

# ╔═╡ 0de7404b-46bd-4462-b6f9-413a86cde6a2
using WGLMakie,GridapMakie

# ╔═╡ 83db2f34-7089-42bc-8e2e-d4cdd4e696fd
using Colors

# ╔═╡ 7ba09689-ffd5-4c58-8a81-a1089b4fd00d
using SpecialFunctions

# ╔═╡ cb382b58-16e1-45ab-8099-6f812c956071
md"""
# Electromagnetic scatering in 2D
In this tutorial, we will learn:

  * How to formulate the weak form for a scalar time-harmonic electromagnetic problem
  * How to implement a perfectly matched layer (PML) to absorb outgoing waves
  * How to impose periodic boundary conditions in Gridap
  * How to discretize PDEs with complex-valued solutions

## Problem statement

We are going to solve a scalar electromagnetic wave scattering problem: a plane wave (Hz-polarized $H_{inc}$) scattering of a dielectric cylinder (of radius $R$ and permittivity $\varepsilon$), as illustrated below. The computational cell is of height $H$ and length $L$, and we employ a perfectly matched layer (PML) thickness of $d_{pml}$ to implement outgoing (radiation) boundary conditions for this finite domain.

![](../assets/emscatter/Illustration.png)

From Maxwell's equations, considering a time-harmonic electromagnetic field, we can derive the governing equation of this problem in 2D (Helmholtz equation) [1]:

```math
\left[-\nabla\cdot\frac{1}{\varepsilon(x)}\nabla -k^2\mu(x)\right] H = f(x),
```

where $k=\omega/c$ is the wave number in free space and $f(x)$ is the source term (which corresponds to a magnetic current density in Maxwell's equations).

In order to simulate this scattering problem in a finite computation cell, we need outgoing (radiation) boundary conditions such that all waves at the boundary would not be reflected back since we are simulating an infinite space.
One commonly used technique to simulate such infinite space is through the so called "perfectly matched layers" (PML) [2]. Actually, PML is not a boundary condition but an artificial absorbing "layer" that absorbs waves with minimal reflections (going to zero as the resolution increases).
There are many formulations of PML. Here, we use one of the most flexible formulations, the "stretched-coordinate" formulation, which takes the following replace in the PDE [3]:

```math
\frac{\partial}{\partial x}\rightarrow \frac{1}{1+\mathrm{i}\sigma(u_x)/\omega}\frac{\partial}{\partial x},
```

```math
\frac{\partial}{\partial y}\rightarrow \frac{1}{1+\mathrm{i}\sigma(u_y)/\omega}\frac{\partial}{\partial y},
```

where $u_{x/y}$ is the depth into the PML, $\sigma$ is a profile function (here we chose $\sigma(u)=\sigma_0(u/d_{pml})^2$) and different derivative corresponds to different absorption directions.
Note that at a finite mesh resolution, PML reflects some waves, and the standard technique to mitigate this is to "turn on" the PML absorption gradually—in this case we use a quadratic profile. The amplitude $\sigma_0$ is chosen so that in the limit of infinite resolution the "round-trip" normal-incidence is some small number.

Since PML absorbs all waves in $x/y$ direction, the associated boundary condition is then usually the zero Dirichlet boundary condition. Here, the boundary conditions are zero Dirichlet boundary on the top and bottom side $\Gamma_D$ but periodic boundary condition on the left ($\Gamma_L$) and right side ($\Gamma_R$).
The reason that we use a periodic boundary condition for the left and right side instead of zero Dirichlet boundary condition is that we want to simulate a plane wave exicitation, which then requires a periodic boundary condition.

Consider $\mu(x)=1$ (which is mostly the case in electromagnetic problems) and denote $\Lambda=\operatorname{diagm}(\Lambda_x,\Lambda_y)$ where $\Lambda_{x/y}=\frac{1}{1+\mathrm{i}\sigma(u_{x/y})/\omega}$, we can formulate the problem as

```math
\left\{ \begin{aligned}
\left[-\Lambda\nabla\cdot\frac{1}{\varepsilon(x)}\Lambda\nabla -k^2\right] H &= f(x) & \text{ in } \Omega,\\
H&=0 & \text{ on } \Gamma_D,\\
H|_{\Gamma_L}&=H|_{\Gamma_R},&\\
\end{aligned}\right.
```
For convenience, in the weak form and Julia implementation below we represent $\Lambda$ as a vector instead of a diagonal $2 \times 2$ matrix, in which case $\Lambda\nabla$ becomes the elementwise product.

## Numerical scheme

Similar to the previous tutorials, we need to construct the weak form for the above PDEs. After integral by part and removing the zero boundary integral term, we get:

```math
a(u,v) = \int_\Omega \left[\nabla(\Lambda v)\cdot\frac{1}{\varepsilon(x)}\Lambda\nabla u-k^2uv\right]\mathrm{d}\Omega
```

```math
b(v) = \int_\Omega vf\mathrm{d}\Omega
```
Notice that the $\nabla(\Lambda v)$ is also a element-wise "product" of two vectors $\nabla$ and $\Lambda v$.

## Setup

We import the packages that will be used, define the geometry and physics parameters.
"""

# ╔═╡ e32971fc-14c0-4ed6-a563-8ba5633147ed
λ = 1.0          # Wavelength (arbitrary unit)

# ╔═╡ be580a34-c070-481c-ad6f-ffed29e6fc3b
L = 4.0          # Width of the area

# ╔═╡ b848e3c7-3b46-41e0-aa78-ff4fcfcb0857
H = 6.0          # Height of the area

# ╔═╡ d0bb4403-176e-4212-8a52-161324ca8d28
xc = [0 -1.0]    # Center of the cylinder

# ╔═╡ 3d2e2bff-4d55-42e6-8ff8-540be800dca0
r = 1.0          # Radius of the cylinder

# ╔═╡ 034816e3-8e5b-4409-b52b-e06fe8cd2933
δ_pml = 0.8      # Thickness of the PML

# ╔═╡ 033fa122-d80e-4df5-b451-64aba1be8a54
k = 2*π/λ        # Wave number

# ╔═╡ b556b8be-9c58-43bb-b0df-29513147ab4d
const ϵ₁ = 3.0   # Relative electric permittivity for cylinder

# ╔═╡ 85abed5a-5cd2-4b89-8928-07c8f48de368
md"""
## Discrete Model

We import the model from the `geometry.msh` mesh file using the `GmshDiscreteModel` function defined in `GridapGmsh`. The mesh file is created with GMSH in Julia (see the file ../assets/emscatter/MeshGenerator.jl). Note that this mesh file already contains periodic boundary information for the left and right side, and that is enough for gridap to realize a periodic boundary condition should be implemented.
"""


# ╔═╡ 4b8e79e3-3841-4f58-bbe6-4b7f72c2739f
import GridapGmsh: gmsh,GmshDiscreteModel

# ╔═╡ 4146c37f-e088-473c-b11d-fb926ea4a5c7
function generate_and_save_mesh(X,Y,a,δ_pml,lc)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.clear()
    gmsh.model.add("geometry")

    function addAndSetPhyicalGroupName(dims, tags, name)
        gmsh.model.setPhysicalName(dims, gmsh.model.addPhysicalGroup(dims, tags), name)
    end

    function addBox(x1, x2, y1, y2, lc, name)
        p11 = gmsh.model.geo.addPoint(x1, y1, 0, lc)
        p21 = gmsh.model.geo.addPoint(x2, y1, 0, lc)
        p22 = gmsh.model.geo.addPoint(x2, y2, 0, lc)
        p12 = gmsh.model.geo.addPoint(x1, y2, 0, lc)

        l1 = gmsh.model.geo.addLine(p11, p21)
        l2 = gmsh.model.geo.addLine(p21, p22)
        l3 = gmsh.model.geo.addLine(p22, p12)
        l4 = gmsh.model.geo.addLine(p12, p11)

        curve_loop_id = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        addAndSetPhyicalGroupName(1, [l1, l2, l3, l4], name)
        return curve_loop_id
    end

    function addCircle(x_centre, y_centre, radius, lc, name)
        p_l = gmsh.model.geo.addPoint(x_centre-radius, y_centre, 0, lc)
        p_c = gmsh.model.geo.addPoint(x_centre,        y_centre, 0, lc)
        p_r = gmsh.model.geo.addPoint(x_centre+radius, y_centre, 0, lc)

        arc1 = gmsh.model.geo.addCircleArc(p_l, p_c, p_r)
        arc2 = gmsh.model.geo.addCircleArc(p_r, p_c, p_l)

        curve_loop_id = gmsh.model.geo.addCurveLoop([arc1, arc2])
        addAndSetPhyicalGroupName(1, [arc1, arc2], name)
        return curve_loop_id
    end

    cylinder_curve_loop_id = addCircle(0.0, 0.0, a, lc, "Cylinder")
    bulk_curve_loop_id = addBox(-X, X, -Y, Y, lc, "BulkOuter")
    pml_curve_loop_id = addBox(-X-δ_pml, X+δ_pml, -Y-δ_pml, Y+δ_pml, lc, "PMLOuter")

    pml_plane_surface_id = gmsh.model.geo.addPlaneSurface([pml_curve_loop_id,bulk_curve_loop_id])
    bulk_plane_surface_id = gmsh.model.geo.addPlaneSurface([bulk_curve_loop_id,cylinder_curve_loop_id])
    addAndSetPhyicalGroupName(2, [pml_plane_surface_id, bulk_plane_surface_id], "WholeMesh")
    addAndSetPhyicalGroupName(2, [pml_plane_surface_id], "PML")
    addAndSetPhyicalGroupName(2, [bulk_plane_surface_id], "Bulk")

    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # ... and save it to disk
    gmsh.write("geometry.msh")
    gmsh.finalize()
end

# ╔═╡ 3d698126-45b0-4b6c-95db-7c284961e2e0
let
	# Geometry parameters
	λ = 1.0          # Wavelength (arbitrary unit)
	Y = 4.0          # Width of the area
	X = 6.0          # Height of the area
	a = 1.0          # Radius of the cylinder
	δ_pml = 0.8      # Thickness of the PML

	resol = 1.0      # Number of points per wavelength
	lc = λ/resol      # Characteristic length

	generate_and_save_mesh(X,Y,a,δ_pml,lc)
end

# ╔═╡ 356a4ad6-4d28-479d-8a59-468bb59529bf
model = GmshDiscreteModel("geometry.msh")

# ╔═╡ 6d0dc467-df67-4404-a7b3-748ed2008325
typeof(model.grid)

# ╔═╡ 49ee6e73-ccf3-47cf-b99c-60f79b5727c2
md"""
## FE spaces

We use the first-order lagrangian as the finite element function space basis. The dirihlet edges are labeld with `DirichletEdges` in the mesh file. Since our problem involves complex numbers (because of PML), we need to assign the `vector_type` to be `Vector{ComplexF64}`.


### Test and trial finite element function space
"""

# ╔═╡ 821a015b-e8e1-47a9-9124-d2431ac6ac75
order = 1

# ╔═╡ 371d409a-447a-46da-88bf-bc6e24c48c7d
reffe = ReferenceFE(lagrangian,Float64,order)

# ╔═╡ 9ec6187a-1393-4496-b508-fc54294a32f0
V = TestFESpace(model,reffe,dirichlet_tags="DirichletEdges",vector_type=Vector{ComplexF64})

# ╔═╡ 58e98595-6015-44ef-8f20-f44cc954b2fe
U = V # mathematically equivalent to TrialFESpace(V,0)

# ╔═╡ ddead707-c0f6-4e9e-a12c-0b69eae32d74
md"""
## Numerical integration

We generate the triangulation and a second-order Gaussian quadrature for the numerial integration. Note that we create a boundary triangulation from a `Source` tag for the line excitation. Generally, we do not need such additional mesh tags for the source, we can use a delta function to approximate such line source excitation. However, by generating a line mesh, we can increase the accuracy of this source excitation.


### Generate triangulation and quadrature from model
"""

# ╔═╡ c4b1c3ae-1e5c-43aa-97ac-33eb17cbaa6a
degree = 2

# ╔═╡ 6ce7d52b-b7a3-46d7-8966-86b93228b04d
Ω = Triangulation(model)

# ╔═╡ 3a00e5aa-957e-40f6-a444-7f076ca4c4e9
let
	fig = plot(Ω)
	wireframe!(Ω, color=RGBA(0,0,0,0.4), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ 5fb2241c-3bf0-4a02-8234-b165d6259c9c
dΩ = Measure(Ω,degree)

# ╔═╡ 63ac2bba-b66b-4eb4-86ad-fd8fbb7f7b76
# ### Source triangulation
Γ = BoundaryTriangulation(model;tags="Source")

# ╔═╡ ca6f374b-b098-4123-97ed-61529f4ca54c
dΓ = Measure(Γ,degree)

# ╔═╡ 5e790bdf-ab97-4fe4-9af5-3670474bb604
md"""
## PML formulation

Here we first define a `s_PML` function: $s(x)=1+\mathrm{i}\sigma(u)/\omega,$ and its derivative `ds_PML`. The parameter `LH` indicates the size of the inner boundary of the PML regions. Finally, we create a function-like object `Λ` that returns the PML factors and define its derivative in gridap.
Note that here we are defining a "callable object" of type `Λ` that encapsulates all of the PML parameters. This is convenient, both because we can pass lots of parameters around easily and also because we can define additional methods on `Λ`, e.g. to express the `∇(Λv)` operation.


### PML parameters
"""

# ╔═╡ d09981ef-e435-4367-9be1-cfd2d0e91312
Rpml = 1e-12      # Tolerence for PML reflection

# ╔═╡ ebe86f17-6c65-4d6a-820f-ecf8eb208a0f
σ = -3/4*log(Rpml)/δ_pml # σ_0

# ╔═╡ fe2cd95b-2a5e-4d73-80a2-d6b5201549d4
LH = [L,H] # Size of the PML inner boundary (a rectangular centere at (0,0))

# ╔═╡ 721e979e-e102-408e-b2c8-0f00531b15d0
# ### PML coordinate streching functions
function s_PML(x,σ,k,LH,δ_pml)
    u = abs.(Tuple(x)).-LH./2  # get the depth into PML
    return @. ifelse(u > 0,  1+(1im*σ/k)*(u/δ_pml)^2, $(1.0+0im))
end

# ╔═╡ c7390e93-3a66-42a0-8971-0078dea083bc
function ds_PML(x,σ,k,LH,δ_pml)
    u = abs.(Tuple(x)).-LH./2 # get the depth into PML
    ds = @. ifelse(u > 0, (2im*σ/k)*(1/δ_pml)^2*u, $(0.0+0im))
    return ds.*sign.(Tuple(x))
end

# ╔═╡ 0813217c-0b06-4991-99c1-7f70706fd726
begin
	struct Λ<:Function
	    σ::Float64
	    k::Float64
	    LH::Vector{Float64}
	    δ_pml::Float64
	end
	function (Λf::Λ)(x)
	    s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)
	    return VectorValue(1/s_x,1/s_y)
	end
end

# ╔═╡ 146b3f7b-dcce-4e39-928d-a88a04284dfb
Fields.∇(Λf::Λ) = x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)[2])

# ╔═╡ 8cb354e5-448d-47c3-bb67-f5ac5992144d
md"""
## Weak form

In the mesh file, we labeled the cylinder region with `Cylinder` to distinguish it from other regions. Using this tag, we can assign material properties correspondingly (basically a function with different value in different regions). The weak form is very similar to its mathematical form in gridap.


### Intermediate variables
"""

# ╔═╡ 4409d494-5e90-420e-9ea4-20ec0c7f28d1
labels = get_face_labeling(model)

# ╔═╡ 4014a378-6f31-4a60-b609-e221893a7590
dimension = num_cell_dims(model)

# ╔═╡ d782b505-858e-4a26-b7c5-70054a6a3dcf
tags = get_face_tag(labels,dimension)

# ╔═╡ b677ede0-8d35-41d3-b9a0-dd09f67dd3fd
const cylinder_tag = get_tag_from_name(labels,"Cylinder")

# ╔═╡ 0016ed55-b3ec-4bb5-aa5d-de8cbd6b3b40
function ξ(tag)
    if tag == cylinder_tag
        return 1/ϵ₁
    else
        return 1.0
    end
end

# ╔═╡ 4c06bfb0-66b5-4c2a-ac25-0ea6d5a4ebdb
τ = CellField(tags,Ω)

# ╔═╡ d341bef6-5d72-427e-9818-c65cf2c2d74c
Λf = Λ(σ,k,LH,δ_pml)

# ╔═╡ 5a20ed81-bbb2-4564-a0b6-f2b207a6d3e6
begin
	struct PMLTransformation<:Function
	    k::Float64
		X::SVector{2,Float64}
		δ::Float64
	end
	function (pml::PMLTransformation)(x)
	    s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)
	    return VectorValue(1/s_x,1/s_y)
	end
	function Fields.∇(pml::PMLTransformation)

		x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.δ_pml)[2])
	end
end

# ╔═╡ 1175432a-682e-418c-a9b0-230e800721f5
md"""
### Bi-linear term (from weak form)
Note that we use a element-wise product .* here for the vector-vector product $\Lambda \nabla$
"""

# ╔═╡ 1879c441-509b-4506-a0ea-6b38e0105ece


# ╔═╡ eb27ab8a-ad56-451f-9568-c97690e844ce
# a(u,v) = ∫( ( J\∇(v)⋅J\∇(u) - (k^2*(v*u))  )*det(J))*dΩ
a(u,v) = ∫( ∇(v)⋅∇(u) - k^2*(v*u))*dΩ

# ╔═╡ 542439b4-d337-4d9d-8323-089a8c33fe96
# ### Source term (uniform line source)
b(v) = 0

# ╔═╡ 81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
md"""
## Solver phase

We can assemble the finite element operator in Gridap with the bi-linear and linear form, then solve for the field.

"""

# ╔═╡ 709ed3be-b0f3-42b5-930b-a136c6803165
op = AffineFEOperator(a,b,U,V)

# ╔═╡ ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
uh = solve(op)

# ╔═╡ 3e36acc8-c468-4c11-a3cb-0fe0f9cece4d
let
	fig = plot(Ω, real.(first.(uh.cell_dof_values)))
	wireframe!(Ω, color=RGBA(1,1,1,0.1), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ 7cfe7698-c04e-4b6a-bbde-9de54b93df3e
md"""
## Analytical solution
### Theoretical analysis
In this section, we construct the semi-analytical solution to this scattering problem, for comparison to the numerical solution. This is possible because of the symmetry of the cylinder, which allows us to expand the solutions of the Helmoltz equation in Bessel functions and match boundary conditions at the cylinder interface. (In 3d, the analogous process with spherical harmonics is known as "Mie scattering".) For more information on this technique, see Ref [4].
In 2D cylinder coordinates, we can expand the plane wave in terms of Bessel functions (this is the Jacobi–Anger identity [5]):

```math
H_0=\sum_m i^mJ_m(kr)e^{im\theta},
```

where $m=0,\pm 1,\pm 2,\dots$ and $J_m(z)$ is the Bessel function of the fisrt kind.

For simplicity, we start with only the $m$-th component and take it as the incident part:

```math
H_{inc}=J_m(kr).
```

For the scattered field, since the scattered wave should be going out, we can then expand it in terms of the Hankel function of the first kind (outgoing and incoming cylindrical waves are Hankel functions of the first and second kind [6]):

```math
H_1=\alpha_mH_m^1(kr).
```

For the fields inside the cylinder, we require the field to be finite at $r=0$, which then constrains the field to be only the expansion of the Bessel functions:

```math
H_2=\beta_mJ_m(nkr),
```

where $n=\sqrt{\varepsilon}$ is the refractive index.

Applying the boundary conditions (tangential part of the electric and magnetic field to be continuous):

```math
H_{inc}+H_1=H_2|_{r=R},
```

```math
\frac{\partial H_{inc}}{\partial r}+\frac{\partial H_1}{\partial r}=\frac{1}{\epsilon}\frac{\partial H_2}{\partial r}|_{r=R}.
```

After some math, we get:

```math
\alpha_m=\frac{J_m(nkR)J_m(kR)^\prime-\frac{1}{n}J_m(kR)J_m(nkR)^\prime}{\frac{1}{n}H_m^1(kR)J_m(nkr)^\prime-J_m(nkr)H_m^1(kr)^\prime},
```

```math
\beta_m = \frac{H_m^1(kR)J_m(kR)^\prime-J_m(kR)H_m^1(kR)^\prime}{\frac{1}{n}J_m(nkR)^\prime H_m^1(kR)-J_m(nkR)H_m^1(kR)^\prime},
```

where $^\prime$ denotes the derivative, and the derivatives of the Bessel functions are obtained with the recurrent relations:

```math
Y_m(z)^\prime=\frac{Y_{m-1}(z)-Y_{m+1}(z)}{2}
```

where $Y_m$ denotes any Bessel functions (Hankel functions).


Finally, the analytical field is ($1/2k$ is the amplitude that comes from the unit line source excitation):
```math
H(r>R)=\frac{1}{2k}\sum_m\left[\alpha_mi^mH_m^1(kr)+J_m(kr)\right]e^{im\theta}
```

```math
H(r\leq R)=\frac{1}{2k}\sum_m\beta_mi^mJ_m(nkr)e^{im\theta}
```


### Define the analytical functions
"""

# ╔═╡ a79d510b-c0cd-4523-b848-9e143092f9a1
dbesselj(m,z) = (besselj(m-1,z)-besselj(m+1,z))/2

# ╔═╡ 7482d4a3-cb3d-48b3-adb3-62ddd0ef865a
dhankelh1(m,z)= (hankelh1(m-1,z)-hankelh1(m+1,z))/2

# ╔═╡ 8e58b553-20c3-470b-88ce-f53ff5affd76
α(m,n,z) = (besselj(m,n*z)*dbesselj(m,z)-1/n*besselj(m,z)*dbesselj(m,n*z))/(1/n*hankelh1(m,z)*dbesselj(m,n*z)-besselj(m,n*z)*dhankelh1(m,z))

# ╔═╡ 6fb3676b-1a66-4504-b0ea-a68fa70ed163
β(m,n,z) = (hankelh1(m,z)*dbesselj(m,z)-besselj(m,z)*dhankelh1(m,z))/(1/n*dbesselj(m,n*z)*hankelh1(m,z)-besselj(m,n*z)*dhankelh1(m,z))

# ╔═╡ d1d79565-d71a-4c40-bd5b-432f1cb14765
function H_t(x,xc,r,ϵ,λ)
    n = √ϵ
    k = 2*π/λ
    θ = angle(x[1]-xc[1]+1im*(x[2]-xc[2]))+π
    M = 40 # Number of Bessel function basis used
    H0 = 0
    if norm([x[1]-xc[1],x[2]-xc[2]])<=r
        for m=-M:M
            H0 += β(m,n,k*r)*cis(m*θ)*besselj(m,n*k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    else
        for m=-M:M
            H0 += α(m,n,k*r)*cis(m*θ)*hankelh1(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))+cis(m*θ)*besselj(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    end
    return 1im/(2*k)*H0
end

# ╔═╡ 203294ed-feb7-458f-b22f-6b4bac230a43
#
# ### Construct the analytical solution in finite element basis
# uh_t = CellField(x->H_t(x,xc,r,ϵ₁,λ),Ω)

# ╔═╡ 1cacb0f1-7874-4eee-95da-afb7d62ef64f
md"""
## Output and compare results

The simulated field is shown below. We can see that the simulated fields and the analytical solution matched closed except for the top and PML regions. This is because the simulated source generate plane waves in two directions but we only consider the downward propagating wave in the analytical solution and the PML effect is also not considered in the analytical solution. Therefore, we just need to focus on the "center" regions which excludes the PML and top region above the source, the difference is within 6% of the field amplitude integral. As we increase the resolution, this difference should decrease (until it becomes limited by the PML reflection coefficient from $\sigma_0$, the number of Bessel function basis $M$ or by floating-point error.)
![](../assets/emscatter/Results.png)

### Save to file and view
"""

# ╔═╡ eda67024-2652-49a3-8008-f4d87611cb64
# writevtk(Ω,"demo",cellfields=["Real"=>real(uh),
#         "Imag"=>imag(uh),
#         "Norm"=>abs2(uh),
#         "Real_t"=>real(uh_t),
#         "Imag_t"=>imag(uh_t),
#         "Norm_t"=>abs2(uh_t),
#         "Difference"=>abs(uh_t-uh)])

# ╔═╡ 2c4db0e3-1e2d-4f85-87e8-e52c243e252a
# ### Compare the difference in the "center" region
function AnalyticalBox(x) # Get the "center" region
    if abs(x[1])<L/2 && abs(x[2]+0.5)<2.5
        return 1
    else
        return 0
    end
end

# ╔═╡ 89204ee5-b7cb-4828-aed1-c625fab3d4fd
Difference=sqrt(sum(∫(abs2(uh_t-uh)*AnalyticalBox)*dΩ)/sum(∫(abs2(uh_t)*AnalyticalBox)*dΩ))

# ╔═╡ 55161119-2673-4c04-b872-3f18796a72dd
@assert Difference < 0.1

# ╔═╡ 4201e390-b137-4f81-a9a1-7c623d3e3fd1
md"""
## References
[1] [Wikipedia: Electromagnetic wave equation](https://en.wikipedia.org/wiki/Electromagnetic_wave_equation)

[2] [Wikipedia: Perfectly matched layer](https://en.wikipedia.org/wiki/Perfectly_matched_layer)

[3] A. Oskooi and S. G. Johnson, “[Distinguishing correct from incorrect PML proposals and a corrected unsplit PML for anisotropic, dispersive media](http://math.mit.edu/~stevenj/papers/OskooiJo11.pdf),” Journal of Computational Physics, vol. 230, pp. 2369–2377, April 2011.

[4] Stratton, J. A. (1941). Electromagnetic Theory. New York: McGraw-Hill.

[5] [Wikipedia: Jacobi–Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion)

[6] [Wikipedia: Bessel function](https://en.wikipedia.org/wiki/Bessel_function)
"""

# ╔═╡ be59001e-b529-49ff-beeb-d901d1bc11cb
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
Gridap = "56d4f2e9-7ea1-5844-9cf6-b9c51ca7ce8e"
GridapGmsh = "3025c34a-b394-11e9-2a55-3fee550c04c8"
GridapMakie = "41f30b06-6382-4b60-a5f7-79d86b35bf5d"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
Colors = "~0.12.8"
Gridap = "~0.17.9"
GridapGmsh = "~0.6.0"
GridapMakie = "~0.1.1"
SpecialFunctions = "~2.1.4"
WGLMakie = "~0.4.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "9f8186bc19cd1c129d367cb667215517cc03e144"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "5.0.1"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "56c347caf09ad8acb3e261fe75f8e09652b7b05b"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.7.10"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BSON]]
git-tree-sha1 = "306bb5574b0c1c56d7e1207581516c557d105cad"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.5"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "21490270d1fcf2efa9ddb2126d6958e9b72a4db0"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "0.16.11"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "dcf8021447cd178d611b910e8c8a162f51095f1d"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.6.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c43e992f186abaf9965cc45e372f4693b7754b22"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.52"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "90b158083179a6ccbce2c7eb1446d5bf9d7ae571"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.7"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipsisNotation]]
git-tree-sha1 = "18ee049accec8763be17a933737c1dd0fdf8673a"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.0.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "56956d1e4c1221000b7781104c58c34019792951"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "d51e69f0a2f8a3842bca4183b700cf3d9acce626"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "35dbc482f0967d8dceaa7ce007d16f9064072166"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.1"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "Printf", "Serialization", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "5a1cb5efff725ebb6b9040eacd24044784459380"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.4.5"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78e2c69783c9753a91cdae88a8d432be85a2ab5e"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Match", "Observables"]
git-tree-sha1 = "d44945bdc7a462fa68bb847759294669352bd0a4"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.5.7"

[[deps.Gridap]]
deps = ["AbstractTrees", "BSON", "BlockArrays", "Combinatorics", "DocStringExtensions", "FastGaussQuadrature", "FileIO", "FillArrays", "ForwardDiff", "JLD2", "JSON", "LineSearches", "LinearAlgebra", "NLsolve", "NearestNeighbors", "QuadGK", "Random", "SparseArrays", "SparseMatricesCSR", "StaticArrays", "Test", "WriteVTK"]
git-tree-sha1 = "14396f1ef81c4eb21ac270cc66e6c2fbb12f115d"
uuid = "56d4f2e9-7ea1-5844-9cf6-b9c51ca7ce8e"
version = "0.17.9"

[[deps.GridapDistributed]]
deps = ["FillArrays", "Gridap", "LinearAlgebra", "MPI", "PartitionedArrays", "SparseArrays", "SparseMatricesCSR", "WriteVTK"]
git-tree-sha1 = "aa4d313f1dbb32f9fb0077e280a0625fa980dcb1"
uuid = "f9701e48-63b3-45aa-9a63-9bc6c271f355"
version = "0.2.5"

[[deps.GridapGmsh]]
deps = ["Gridap", "GridapDistributed", "Libdl", "Metis", "PartitionedArrays", "gmsh_jll"]
git-tree-sha1 = "d7e4e6b11b44a99aa400871e5f89be1be23814d2"
uuid = "3025c34a-b394-11e9-2a55-3fee550c04c8"
version = "0.6.0"

[[deps.GridapMakie]]
deps = ["CairoMakie", "FileIO", "FillArrays", "GLMakie", "GeometryBasics", "Gridap", "Makie"]
git-tree-sha1 = "37f1c816d088fa64851a9b0227317ebca5cf7b13"
uuid = "41f30b06-6382-4b60-a5f7-79d86b35bf5d"
version = "0.1.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[deps.ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f025b79883f361fa1bd80ad132773161d231fd9f"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "28b114b3279cdbac9a61c57b3e6548a572142b34"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.21"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "8c1f668b24d999fb47baf80436194fdccec65ad2"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.4"

[[deps.JSServe]]
deps = ["Base64", "CodecZlib", "Colors", "HTTP", "Hyperscript", "JSON3", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "Test", "UUIDs", "WebSockets", "WidgetsBase"]
git-tree-sha1 = "e8c3434c3e880e15760821a9eac00deb35ab6ea9"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f9"
version = "1.2.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1d31872bb9c5e7ec1f618e8c4a56c8b0d9bddc7e"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.1+0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MPI]]
deps = ["Distributed", "DocStringExtensions", "Libdl", "MPICH_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "Pkg", "Random", "Requires", "Serialization", "Sockets"]
git-tree-sha1 = "d56a80d8cf8b9dc3050116346b3d83432b1912c0"
uuid = "da04e1cc-30fd-572f-bb4f-1f8673147195"
version = "0.19.2"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8eed51eb836c8f47781cdb493ffd5f56370c0496"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.0.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "d03c5a4056707bb8d43e349bc2cb49fc1cfa8b9f"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.15.1"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "7bcc8323fb37523a6a51ade2234eee27a11114c8"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.1.3"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "Test"]
git-tree-sha1 = "69b565c0ca7bf9dae18498b52431f854147ecbf3"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.1.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "6d92d825d3834ecad23ffd5582dc67da7e6f020c"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.7"

[[deps.Metis]]
deps = ["Graphs", "LightGraphs", "LinearAlgebra", "METIS_jll", "SparseArrays"]
git-tree-sha1 = "a20753003f8664921929ef9f719ae91bf47a37c1"
uuid = "2679e427-3c69-5b7f-982b-ece356f1e94b"
version = "1.1.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "a16aa086d335ed7e0170c5265247db29172af2f9"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.3+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "344f8896e55541e30d5ccffcbf747c98ad57ca47"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "a8cbf066b54d793b9a48c5daa5d586cf2b5bd43d"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.1.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "6340586e076b2abd41f5ba1a3b9c774ec6b30fde"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.2+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "eb4dbb8139f6125471aa3da98fb70f02dc58e49c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.14"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "f4049d379326c2c7aa875c702ad19346ecb2b004"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "13468f237353112a01b2d6b32f3d0f80219944aa"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.2"

[[deps.PartitionedArrays]]
deps = ["Distances", "IterativeSolvers", "LinearAlgebra", "MPI", "Printf", "SparseArrays", "SparseMatricesCSR"]
git-tree-sha1 = "88ff2293fd57089a4036a3056ba058ae9806111b"
uuid = "5a9dfac6-5c52-46f7-8278-5e2210713be9"
version = "0.2.10"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "de893592a221142f3db370f48290e3a2ef39998f"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "63c6b8796d28a1f942c29659e5519e2ef9ef4a59"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.2.7"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseMatricesCSR]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4870b3e7db7063927b163fb981bd579410b68b2d"
uuid = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
version = "0.6.6"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "87e9954dfa33fd145694e42337bdd3d5b07021a6"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.6.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "74fb527333e72ada2dd9ef77d98e4991fb185f04"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "aaa19086bc282630d82f818456bc40b4d314307d"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.4"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.WGLMakie]]
deps = ["Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "ImageMagick", "JSServe", "LinearAlgebra", "Makie", "Observables", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "56d9ca8f1e1c2c09c1ccdf19c4ce5d45efd66097"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.4.5"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "c1ef6e02bc457c3b23aafc765b94c3dcd25f174d"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WriteVTK]]
deps = ["Base64", "CodecZlib", "FillArrays", "LightXML", "TranscodingStreams"]
git-tree-sha1 = "bff2f6b5ff1e60d89ae2deba51500ce80014f8f6"
uuid = "64499a7a-5c06-52f2-abe2-ccb03c286192"
version = "1.14.2"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.gmsh_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9554bb1cad1926e7d3afb68b0ab117d0b9bb73ee"
uuid = "630162c2-fc9b-58b3-9910-8442a8a132e6"
version = "4.9.3+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─cb382b58-16e1-45ab-8099-6f812c956071
# ╠═2b1f6cbc-7f38-4573-a722-22668837de03
# ╠═5acadaf2-0c4c-494c-a305-27b1c41cc7a2
# ╠═e10c5855-3713-4a81-b071-8fe14c3aa0af
# ╠═1d772e87-2143-498f-bdf0-b1594b8d31e3
# ╠═e32971fc-14c0-4ed6-a563-8ba5633147ed
# ╠═be580a34-c070-481c-ad6f-ffed29e6fc3b
# ╠═b848e3c7-3b46-41e0-aa78-ff4fcfcb0857
# ╠═d0bb4403-176e-4212-8a52-161324ca8d28
# ╠═3d2e2bff-4d55-42e6-8ff8-540be800dca0
# ╠═034816e3-8e5b-4409-b52b-e06fe8cd2933
# ╠═033fa122-d80e-4df5-b451-64aba1be8a54
# ╠═b556b8be-9c58-43bb-b0df-29513147ab4d
# ╟─85abed5a-5cd2-4b89-8928-07c8f48de368
# ╠═4b8e79e3-3841-4f58-bbe6-4b7f72c2739f
# ╠═4146c37f-e088-473c-b11d-fb926ea4a5c7
# ╠═3d698126-45b0-4b6c-95db-7c284961e2e0
# ╠═356a4ad6-4d28-479d-8a59-468bb59529bf
# ╠═3a00e5aa-957e-40f6-a444-7f076ca4c4e9
# ╠═0de7404b-46bd-4462-b6f9-413a86cde6a2
# ╠═6d0dc467-df67-4404-a7b3-748ed2008325
# ╟─49ee6e73-ccf3-47cf-b99c-60f79b5727c2
# ╠═821a015b-e8e1-47a9-9124-d2431ac6ac75
# ╠═371d409a-447a-46da-88bf-bc6e24c48c7d
# ╠═9ec6187a-1393-4496-b508-fc54294a32f0
# ╠═58e98595-6015-44ef-8f20-f44cc954b2fe
# ╟─ddead707-c0f6-4e9e-a12c-0b69eae32d74
# ╠═c4b1c3ae-1e5c-43aa-97ac-33eb17cbaa6a
# ╠═6ce7d52b-b7a3-46d7-8966-86b93228b04d
# ╠═5fb2241c-3bf0-4a02-8234-b165d6259c9c
# ╠═63ac2bba-b66b-4eb4-86ad-fd8fbb7f7b76
# ╠═ca6f374b-b098-4123-97ed-61529f4ca54c
# ╟─5e790bdf-ab97-4fe4-9af5-3670474bb604
# ╠═d09981ef-e435-4367-9be1-cfd2d0e91312
# ╠═ebe86f17-6c65-4d6a-820f-ecf8eb208a0f
# ╠═fe2cd95b-2a5e-4d73-80a2-d6b5201549d4
# ╠═721e979e-e102-408e-b2c8-0f00531b15d0
# ╠═c7390e93-3a66-42a0-8971-0078dea083bc
# ╠═0813217c-0b06-4991-99c1-7f70706fd726
# ╠═146b3f7b-dcce-4e39-928d-a88a04284dfb
# ╠═5a20ed81-bbb2-4564-a0b6-f2b207a6d3e6
# ╟─8cb354e5-448d-47c3-bb67-f5ac5992144d
# ╠═4409d494-5e90-420e-9ea4-20ec0c7f28d1
# ╠═4014a378-6f31-4a60-b609-e221893a7590
# ╠═d782b505-858e-4a26-b7c5-70054a6a3dcf
# ╠═b677ede0-8d35-41d3-b9a0-dd09f67dd3fd
# ╠═0016ed55-b3ec-4bb5-aa5d-de8cbd6b3b40
# ╠═4c06bfb0-66b5-4c2a-ac25-0ea6d5a4ebdb
# ╠═d341bef6-5d72-427e-9818-c65cf2c2d74c
# ╟─1175432a-682e-418c-a9b0-230e800721f5
# ╠═1879c441-509b-4506-a0ea-6b38e0105ece
# ╠═eb27ab8a-ad56-451f-9568-c97690e844ce
# ╠═542439b4-d337-4d9d-8323-089a8c33fe96
# ╟─81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
# ╠═709ed3be-b0f3-42b5-930b-a136c6803165
# ╠═ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
# ╠═83db2f34-7089-42bc-8e2e-d4cdd4e696fd
# ╠═3e36acc8-c468-4c11-a3cb-0fe0f9cece4d
# ╟─7cfe7698-c04e-4b6a-bbde-9de54b93df3e
# ╠═7ba09689-ffd5-4c58-8a81-a1089b4fd00d
# ╠═a79d510b-c0cd-4523-b848-9e143092f9a1
# ╠═7482d4a3-cb3d-48b3-adb3-62ddd0ef865a
# ╠═8e58b553-20c3-470b-88ce-f53ff5affd76
# ╠═6fb3676b-1a66-4504-b0ea-a68fa70ed163
# ╠═d1d79565-d71a-4c40-bd5b-432f1cb14765
# ╠═203294ed-feb7-458f-b22f-6b4bac230a43
# ╟─1cacb0f1-7874-4eee-95da-afb7d62ef64f
# ╠═eda67024-2652-49a3-8008-f4d87611cb64
# ╠═2c4db0e3-1e2d-4f85-87e8-e52c243e252a
# ╠═89204ee5-b7cb-4828-aed1-c625fab3d4fd
# ╠═55161119-2673-4c04-b872-3f18796a72dd
# ╟─4201e390-b137-4f81-a9a1-7c623d3e3fd1
# ╠═be59001e-b529-49ff-beeb-d901d1bc11cb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
