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
d_pml = 0.8      # Thickness of the PML

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
import Gmsh: gmsh

# ╔═╡ 4146c37f-e088-473c-b11d-fb926ea4a5c7
function MeshGenerator(L,H,xc,r,hs,d_pml,lc)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.clear()
    gmsh.model.add("geometry")

    # Add points
    gmsh.model.geo.addPoint(-L/2-d_pml, -H/2-d_pml, 0, lc, 1)
    gmsh.model.geo.addPoint( L/2+d_pml, -H/2-d_pml, 0, lc, 2)
    gmsh.model.geo.addPoint( L/2+d_pml,  hs       , 0, lc, 3)
    gmsh.model.geo.addPoint(-L/2-d_pml,  hs       , 0, lc, 4)
    gmsh.model.geo.addPoint( L/2+d_pml,  H/2+d_pml, 0, lc, 5)
    gmsh.model.geo.addPoint(-L/2-d_pml,  H/2+d_pml, 0, lc, 6)
    gmsh.model.geo.addPoint( xc[1]-r  ,  xc[2]    , 0, lc, 7)
    gmsh.model.geo.addPoint( xc[1]    ,  xc[2]    , 0, lc, 8)
    gmsh.model.geo.addPoint( xc[1]+r  ,  xc[2]    , 0, lc, 9)
    # Add lines
    gmsh.model.geo.addLine( 1,  2,  1)
    gmsh.model.geo.addLine( 2,  3,  2)
    gmsh.model.geo.addLine( 3,  4,  3)
    gmsh.model.geo.addLine( 1,  4,  4)
    gmsh.model.geo.addLine( 3,  5,  5)
    gmsh.model.geo.addLine( 5,  6,  6)
    gmsh.model.geo.addLine( 4,  6,  7)
    gmsh.model.geo.addCircleArc( 7, 8, 9, 8)
    gmsh.model.geo.addCircleArc( 9, 8, 7, 9)
    # Construct curve loops and surfaces
    gmsh.model.geo.addCurveLoop([1, 2, 3, -4], 1)
    gmsh.model.geo.addCurveLoop([5, 6,-7, -3], 2)
    gmsh.model.geo.addCurveLoop([8, 9], 3)
    gmsh.model.geo.addPlaneSurface([1,3], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.addPlaneSurface([3], 3)
    # Physical groups
    #gmsh.model.addPhysicalGroup(0, [1,2,3,4,5,6], 1)
    #gmsh.model.setPhysicalName(0, 1, "DirichletNodes")
    gmsh.model.addPhysicalGroup(1, [1,6], 2)
    gmsh.model.setPhysicalName(1, 2, "DirichletEdges")
    gmsh.model.addPhysicalGroup(0, [7,9], 3)
    gmsh.model.setPhysicalName(0, 3, "CylinderNodes")
    gmsh.model.addPhysicalGroup(1, [8,9], 4)
    gmsh.model.setPhysicalName(1, 4, "CylinderEdges")
    gmsh.model.addPhysicalGroup(2, [3], 5)
    gmsh.model.setPhysicalName(2, 5, "Cylinder")
    gmsh.model.addPhysicalGroup(2, [1,2], 6)
    gmsh.model.setPhysicalName(2, 7, "Air")
    gmsh.model.addPhysicalGroup(1, [3], 7)
    gmsh.model.setPhysicalName(1, 7, "Source")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setPeriodic(1, [2], [4],
                            [1, 0, 0, L+2*d_pml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.model.mesh.setPeriodic(1, [5], [7],
                            [1, 0, 0, L+2*d_pml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

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
	L = 4.0          # Width of the area
	H = 6.0          # Height of the area
	xc = [0 -1.0]    # Center of the cylinder
	r = 1.0          # Radius of the cylinder
	hs = 2.0         # y-position of the source (plane wave)
	d_pml = 0.8      # Thickness of the PML

	resol = 10.0      # Number of points per wavelength
	lc = λ/resol      # Characteristic length

	MeshGenerator(L,H,xc,r,hs,d_pml,lc)
end

# ╔═╡ 356a4ad6-4d28-479d-8a59-468bb59529bf
model = GmshDiscreteModel("geometry.msh")

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
σ = -3/4*log(Rpml)/d_pml # σ_0

# ╔═╡ fe2cd95b-2a5e-4d73-80a2-d6b5201549d4
LH = [L,H] # Size of the PML inner boundary (a rectangular centere at (0,0))

# ╔═╡ 721e979e-e102-408e-b2c8-0f00531b15d0
# ### PML coordinate streching functions
function s_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2  # get the depth into PML
    return @. ifelse(u > 0,  1+(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

# ╔═╡ c7390e93-3a66-42a0-8971-0078dea083bc
function ds_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2 # get the depth into PML
    ds = @. ifelse(u > 0, (2im*σ/k)*(1/d_pml)^2*u, $(0.0+0im))
    return ds.*sign.(Tuple(x))
end

# ╔═╡ 0813217c-0b06-4991-99c1-7f70706fd726
begin
	struct Λ<:Function
	    σ::Float64
	    k::Float64
	    LH::Vector{Float64}
	    d_pml::Float64
	end
	function (Λf::Λ)(x)
	    s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)
	    return VectorValue(1/s_x,1/s_y)
	end
end

# ╔═╡ 146b3f7b-dcce-4e39-928d-a88a04284dfb
Fields.∇(Λf::Λ) = x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[2])

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
Λf = Λ(σ,k,LH,d_pml)

# ╔═╡ 1175432a-682e-418c-a9b0-230e800721f5
md"""
### Bi-linear term (from weak form)
Note that we use a element-wise product .* here for the vector-vector product $\Lambda \nabla$
"""

# ╔═╡ eb27ab8a-ad56-451f-9568-c97690e844ce
a(u,v) = ∫(  (∇.*(Λf*v))⊙((ξ∘τ)*(Λf.*∇(u))) - (k^2*(v*u))  )dΩ

# ╔═╡ 542439b4-d337-4d9d-8323-089a8c33fe96
# ### Source term (uniform line source)
b(v) = ∫(v)*dΓ

# ╔═╡ 81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
md"""
## Solver phase

We can assemble the finite element operator in Gridap with the bi-linear and linear form, then solve for the field.

"""

# ╔═╡ 709ed3be-b0f3-42b5-930b-a136c6803165
op = AffineFEOperator(a,b,U,V)

# ╔═╡ ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
uh = solve(op)

# ╔═╡ 7c28e147-8d1c-460d-ac0d-05ab361d815e
size(uh.cell_dof_values)

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
uh_t = CellField(x->H_t(x,xc,r,ϵ₁,λ),Ω)

# ╔═╡ eda67024-2652-49a3-8008-f4d87611cb64
# ## Output and compare results
#
# The simulated field is shown below. We can see that the simulated fields and the analytical solution matched closed except for the top and PML regions. This is because the simulated source generate plane waves in two directions but we only consider the downward propagating wave in the analytical solution and the PML effect is also not considered in the analytical solution. Therefore, we just need to focus on the "center" regions which excludes the PML and top region above the source, the difference is within 6% of the field amplitude integral. As we increase the resolution, this difference should decrease (until it becomes limited by the PML reflection coefficient from $\sigma_0$, the number of Bessel function basis $M$ or by floating-point error.)
# ![](../assets/emscatter/Results.png)

# ### Save to file and view
writevtk(Ω,"demo",cellfields=["Real"=>real(uh),
        "Imag"=>imag(uh),
        "Norm"=>abs2(uh),
        "Real_t"=>real(uh_t),
        "Imag_t"=>imag(uh_t),
        "Norm_t"=>abs2(uh_t),
        "Difference"=>abs(uh_t-uh)])

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Gmsh = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
Gridap = "56d4f2e9-7ea1-5844-9cf6-b9c51ca7ce8e"
GridapGmsh = "3025c34a-b394-11e9-2a55-3fee550c04c8"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
Gmsh = "~0.1.1"
Gridap = "~0.17.9"
GridapGmsh = "~0.6.0"
SpecialFunctions = "~2.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

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

[[deps.BSON]]
git-tree-sha1 = "306bb5574b0c1c56d7e1207581516c557d105cad"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.5"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "21490270d1fcf2efa9ddb2126d6958e9b72a4db0"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "0.16.11"

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

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

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

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.Gmsh]]
deps = ["BinaryProvider", "Libdl"]
git-tree-sha1 = "1204b5592a195d71569889c29670c72ca9118a87"
uuid = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
version = "0.1.1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

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

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

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

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

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

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "6340586e076b2abd41f5ba1a3b9c774ec6b30fde"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

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

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "de893592a221142f3db370f48290e3a2ef39998f"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.gmsh_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9554bb1cad1926e7d3afb68b0ab117d0b9bb73ee"
uuid = "630162c2-fc9b-58b3-9910-8442a8a132e6"
version = "4.9.3+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
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
# ╟─8cb354e5-448d-47c3-bb67-f5ac5992144d
# ╠═4409d494-5e90-420e-9ea4-20ec0c7f28d1
# ╠═4014a378-6f31-4a60-b609-e221893a7590
# ╠═d782b505-858e-4a26-b7c5-70054a6a3dcf
# ╠═b677ede0-8d35-41d3-b9a0-dd09f67dd3fd
# ╠═0016ed55-b3ec-4bb5-aa5d-de8cbd6b3b40
# ╠═4c06bfb0-66b5-4c2a-ac25-0ea6d5a4ebdb
# ╠═d341bef6-5d72-427e-9818-c65cf2c2d74c
# ╟─1175432a-682e-418c-a9b0-230e800721f5
# ╠═eb27ab8a-ad56-451f-9568-c97690e844ce
# ╠═542439b4-d337-4d9d-8323-089a8c33fe96
# ╟─81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
# ╠═709ed3be-b0f3-42b5-930b-a136c6803165
# ╠═ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
# ╠═7c28e147-8d1c-460d-ac0d-05ab361d815e
# ╟─7cfe7698-c04e-4b6a-bbde-9de54b93df3e
# ╠═7ba09689-ffd5-4c58-8a81-a1089b4fd00d
# ╠═a79d510b-c0cd-4523-b848-9e143092f9a1
# ╠═7482d4a3-cb3d-48b3-adb3-62ddd0ef865a
# ╠═8e58b553-20c3-470b-88ce-f53ff5affd76
# ╠═6fb3676b-1a66-4504-b0ea-a68fa70ed163
# ╠═d1d79565-d71a-4c40-bd5b-432f1cb14765
# ╠═203294ed-feb7-458f-b22f-6b4bac230a43
# ╠═eda67024-2652-49a3-8008-f4d87611cb64
# ╠═2c4db0e3-1e2d-4f85-87e8-e52c243e252a
# ╠═89204ee5-b7cb-4828-aed1-c625fab3d4fd
# ╠═55161119-2673-4c04-b872-3f18796a72dd
# ╟─4201e390-b137-4f81-a9a1-7c623d3e3fd1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
