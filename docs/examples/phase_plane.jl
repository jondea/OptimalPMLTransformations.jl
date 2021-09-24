### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a4c9b6cf-fade-4297-9c48-bd99cd1d8bff
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 1cf4d120-29fe-41b6-83a4-f935fa46a6bf
begin
	using Revise
	Pkg.add(url="https://github.com/jondea/InverseHankelFunction.jl")
	Pkg.develop(url="https://github.com/jondea/OptimalPMLTransformations.jl")
	using OptimalPMLTransformations
end

# ╔═╡ c27e7cdc-4e9f-4cf5-93f0-cfff97216806
begin
    Pkg.add(["Plots", "PlotlyJS"])
    using Plots
	plotlyjs()
end

# ╔═╡ 3533c492-052b-419d-8c2b-24efccf87a32
md"""
# Transformation as ODE through complex plane
"""

# ╔═╡ 074dceba-1d44-11ec-12b8-df9845c59094
md"""
## Technical bits 
"""

# ╔═╡ 9ff5581c-6f3f-47b4-9356-11890bfc79c6
function complex_heatmap!(f, complex_to_real_fnc;
        xlims=(-50,50), ylims=(-50,50), n_samples=100,
        kwargs...)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    heatmap!(x, y, (x,y)->complex_to_real_fnc(f(x+im*y)), kwargs...)
end

# ╔═╡ e491d97d-a7c9-41e3-925f-91e3639ca3ae
log10abs(z) = log10(abs(z))

# ╔═╡ fb9e039a-5067-4470-b061-befaaad5fe45
logsigmoid(z) = z/(1+abs(z))

# ╔═╡ 819e6e67-116b-4ea8-bd40-bfeed3e9048d
function complex_quiver!(f; xlims=(-50,50), ylims=(-50,50),
        arrow_scale=0.1, arrow_fnc=z->arrow_scale*(xlims[2]-xlims[1])*logsigmoid(z),
        n_arrows=20, kwargs...
    )

    xh = (xlims[2] - xlims[1])/n_arrows
    yh = (ylims[2] - ylims[1])/n_arrows

    x = (xlims[1]+xh/2):xh:(xlims[2]-xh/2)
    y = (ylims[1]+yh/2):yh:(ylims[2]-yh/2)

    grid = x .+ im.*reshape(y,1,:)

    complex_to_2tuple(z) = (real(z),imag(z))
    arrows = (complex_to_2tuple∘arrow_fnc∘f).(grid)
    quiver!(real.(grid), imag.(grid), quiver=arrows, arrow=arrow(:closed, :head, 0.2, 0.1), color=:black)
end

# ╔═╡ 0d962ea4-b0f9-46a8-8377-442e214e1e4c
function plot_phase_plane(;
    R = 2.0,
    δ = 1.0,
    θ = 0.0,
    k = 1.0,
    a = two_mode_pole_coef(0:1, 2.0+1.0im),
    xlims=(-50,50),
    ylims=(-50,50),
    n_samples=200,
    n_arrows=20,)

    function ode_rhs(tr)
        u, ∂u_∂tν = HankelSeries(k, a)(NamedTuple{(:u, :∂u_∂tr)}, PolarCoordinates(tr,θ))
        return -u/∂u_∂tν
    end

    plot()

    complex_heatmap!(ode_rhs, log10abs; xlims, ylims, n_samples)

    complex_quiver!(ode_rhs; xlims, ylims, n_arrows)
end

# ╔═╡ 6abc4132-48e1-4dfb-a341-f87bf9fba916
plot_phase_plane(n_samples=400, k=1.0)

# ╔═╡ Cell order:
# ╟─3533c492-052b-419d-8c2b-24efccf87a32
# ╠═6abc4132-48e1-4dfb-a341-f87bf9fba916
# ╠═074dceba-1d44-11ec-12b8-df9845c59094
# ╠═a4c9b6cf-fade-4297-9c48-bd99cd1d8bff
# ╠═1cf4d120-29fe-41b6-83a4-f935fa46a6bf
# ╠═c27e7cdc-4e9f-4cf5-93f0-cfff97216806
# ╠═9ff5581c-6f3f-47b4-9356-11890bfc79c6
# ╠═819e6e67-116b-4ea8-bd40-bfeed3e9048d
# ╠═0d962ea4-b0f9-46a8-8377-442e214e1e4c
# ╠═e491d97d-a7c9-41e3-925f-91e3639ca3ae
# ╠═fb9e039a-5067-4470-b061-befaaad5fe45
