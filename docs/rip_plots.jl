include("exact_mapping.jl")
# push!(LOAD_PATH, "./")
# using ExactMapping

using SpecialFunctions
using LinearAlgebra
using OffsetArrays
using Plots
using LaTeXStrings
using ProgressMeter
using FastGaussQuadrature

using JondeaUtils

function plot_around_zeta(field_fnc::Function, ν, tν₀=0.0; Nζ=100, ζ₋=0.0, ζ₊=2π)
    ζ_vec = range(ζ₋, stop=ζ₊, length=Nζ)

    k = 1.0 # Define k = 1, it makes no difference
    tν_vec = map(ζ -> exact_mapping_solve_adaptive_steps(tν->field_fnc(tν₀+tν, ζ), ν)[1], ζ_vec)
    plot!(ζ_vec, abs.(tν_vec), grid=false)
end

logsigmoid(z) = z/(1+abs(z))

plot_f(field_fnc::Function, ζ::Number; kwargs...) = plot_f(tν->field_fnc(tν, ζ); kwargs...)

function plot_f(field_fnc::Function; color_fnc=z->log10(abs(z)), xlims=(-50,50), ylims=(-5,95),
        arrow_scale=0.1, arrow_fnc=z->arrow_scale*(xlims[2]-xlims[1])*logsigmoid(z),
        n_arrows=20, n_samples=100, clims=:auto, title="Gradient field", kwargs...
    )

    f(z) = dtν_dt(field_fnc(z))

    # Plot gradient field
    x = range(xlims[1], stop=xlims[2], length=n_samples)
    y = range(ylims[1], stop=ylims[2], length=n_samples)
    heatmap(x, y, (x,y)->color_fnc(f(x+im*y)), clims=clims)

    # Add arrows
    x_sparse = range(xlims[1], stop=xlims[2], length=n_arrows)[1:end-1]
    x_sparse = x_sparse .+ step(x_sparse)/2
    y_sparse = range(ylims[1], stop=ylims[2], length=n_arrows)[1:end-1]
    y_sparse = y_sparse .+ step(y_sparse)/2
    grid = (reshape([x for x in x_sparse, y in y_sparse],:), reshape([y for x in x_sparse, y in y_sparse],:))
    complex_grid = grid[1] .+ im*grid[2]
    complex_to_2tuple(z) = (real(z),imag(z))
    arrows = (complex_to_2tuple∘arrow_fnc∘f).(complex_grid)
    quiver!(grid[1], grid[2], quiver=arrows, arrow=arrow(:closed, :head, 0.2, 0.1), color=:black)

    plot!(xlims=xlims, ylims=ylims, aspectratio=1.0, title=title, xlab="Real(transformed nu)", ylab="Imaginary(transformed nu)", kwargs...)
end


function add_path_to_plot(field_fnc::Function, tν₀::Number, tν_vec=Vector{Complex{Float64}}(undef,1); h=1e-3, N_steps=10000, color=:white)

    tν = complex(tν₀)
    tν_vec[1] = tν
    f = 1.0 + 0.0im
    ν = zero(h)
    inside_plot(tν)::Bool = (   (xlims()[1] <= real(tν) <= xlims()[2])
                             && (xlims()[1] <= imag(tν) <= ylims()[2]))
    steps=0

    while inside_plot(tν) && steps < N_steps

        field = field_fnc(tν)
        f = dtν_dt(field)
        tν = tν + f*h
        push!(tν_vec, tν)
        steps += 1
    end

    plot!(real.(tν_vec), imag.(tν_vec), label="", line=(1.0, color))
end

function add_path_to_plot_nu_steps(field_fnc::Function, tν₀::Number, tν_vec=Vector{Complex{Float64}}(undef,1); h=1e-3, ν_abs_max=1, N_steps=10000)

    tν = complex(tν₀)
    tν_vec[1] = tν
    ν = zero(h)
    steps=0

    U_field = field_fnc(tν₀)

    while steps < N_steps && abs(ν) < ν_abs_max

        field = field_fnc(tν)
        f = dtν_dν(field, U_field)
        tν = tν + f*h
        ν += h
        push!(tν_vec, tν)
        steps += 1
    end

    plot!(real.(tν_vec), imag.(tν_vec), label="", line=(1.0, :white))
end

function plot_f_with_paths(f::Function, tν₀_vec; kwargs...)

    plot_f(f; kwargs...)

    for tν₀ in tν₀_vec
        add_path_to_plot(f, tν₀; forward=true, h=1e-3)
    end

    plot!()
end

function plot_objective(field_fnc::Function; color_fnc=z->log10(z),
        xlims=(-50,50), ylims=(-5,95),
        n_samples=100, clims=:auto, title="Objective field", colorbar=true,
        size=(800,700)
    )

    U = field_fnc(0.0+0.0im).u

    function objective(z)
        q = field_fnc(z).u/U
        nu_zero_deriv = 1.0 - (real(q)*real(q) + imag(q)*imag(q))/real(q);
        return abs(q/(1.0-nu_zero_deriv) - 1.0)
    end

    # Plot gradient field
    x = range(xlims[1], stop=xlims[2], length=n_samples)
    y = range(ylims[1], stop=ylims[2], length=n_samples)
    heatmap(x, y, (x,y)->color_fnc(objective(x+im*y)), clims=clims, colorbar=colorbar)

    plot!(xlims=xlims, ylims=ylims, aspectratio=1.0, title=title, xlab=L"Real($\tilde \nu$)", ylab=L"Imag($\tilde \nu$)", colorbar_title=L"\log_{10}(q)", size=size, colorbar=colorbar)
end

function plot_abs_residual_fixed_nu(field_fnc::Function, ν; color_fnc=z->log10(z),
        xlims=(-50,50), ylims=(-5,95),
        n_samples=100, clims=:auto, title="Objective field", colorbar=true,
        size=(800,700)
    )

    U = field_fnc(0.0+0.0im).u

    abs_res(z) = abs(field_fnc(z).u/U - (1-ν))

    # Plot gradient field
    x = range(xlims[1], stop=xlims[2], length=n_samples)
    y = range(ylims[1], stop=ylims[2], length=n_samples)
    heatmap(x, y, (x,y)->color_fnc(abs_res(x+im*y)), clims=clims, colorbar=colorbar)

    plot!(xlims=xlims, ylims=ylims, aspectratio=1.0, title=title, xlab=L"Real($\tilde \nu$)", ylab=L"Imag($\tilde \nu$)", colorbar_title=L"\log_{10}(|\mathcal{R}(\tilde \nu)|)", size=size, colorbar=colorbar)
end

function plot_toy_dr_dnu(;fnc=z->log(abs(z)), arrow_scale=0.5, arrow_fnc=z->arrow_scale*logsigmoid(z), xlims=(-5,5), ylims=(-5,5), n_arrows=20, n_samples=200, clims=:auto)
    f(z) = - im/(im-z) + exp(im*z)
    x = range(xlims[1], stop=xlims[2], length=n_samples)
    y = range(ylims[1], stop=ylims[2], length=n_samples)
    heatmap(x, y, (x,y)->fnc(f(x.+im.*y)), clims=clims)
    x_sparse = range(xlims[1], stop=xlims[2], length=n_arrows)[2:end-1]
    y_sparse = range(ylims[1], stop=ylims[2], length=n_arrows)[2:end-1]
    grid = (reshape([x for x in x_sparse, y in y_sparse],:), reshape([y for x in x_sparse, y in y_sparse],:))
    complex_grid = grid[1] .+ im*grid[2]
    complex_to_2tuple(z) = (real(z),imag(z))
    arrows = complex_to_2tuple.(arrow_fnc.(f.(complex_grid)))
    quiver!(grid[1], grid[2], quiver=arrows, arrow=arrow(:closed, :head, 0.2, 0.1), color=:black)
    plot!(xlims=xlims, ylims=ylims, aspectratio=1.0)
end

function plot_coef_ratio_against_pole_location()
    kr_ims = 2.0.^(0:0.1:10)
    kr_res = 10.0.^range(-1, stop=3, length=1000)
    plot(grid=false, yscale=:log10, ylab="Re(pole location)", xlab="Ratio of coefficients", xlims=(0.7,1.0), xticks=([0.7,0.8,0.9,1.0],["0.7","0.8","0.9","1.0"]))
    for (i, kr_im) in enumerate(kr_ims)
        plot!(abs.([two_mode_pole_coef(1,kr_re+kr_im*im) for kr_re in kr_res]), kr_res, color=RGB(0.8,0.1,0.1), label="")
    end
    plot!()
end

function plot_hankel_around_theta(coefs, ν, kR; Nθ=100, θ_start=0.0, θ_end=2π)
    θ_vec = range(θ_start, stop=θ_end, length=Nθ)

    k = 1.0 # Define k = 1, it makes no difference
    kr_vec = map(θ -> k*exact_mapping_solve_with_steps_single_nu(coefs, ν, θ, k, kR/k)[1], θ_vec)
    plot!(θ_vec, abs.(kr_vec), grid=false)
end

function plot_f_with_path_multiple_ys(ys; x_pole=0.0+1.0*im, y_pole=0.0, Xs=[0.0])

    for (i,y) in enumerate(ys)

        plot_f_with_paths(x->two_planar_pole(x,y,x_pole,y_pole), Xs, xlims=(-1,1), ylims=(0,2))

        plot!(title="Log magnitude of gradient field for $(rpad("y=$y",6,'0'))")
        plot!(clims=(-5,3))
        plot!(size=(1100,1000))
        savefig("images/f_plot_$(lpad("$i",3,'0'))")
    end

    return
end

function plot_rips(scale=:identity; R=2.0, N=10)
    # For N = 10 this works
    ks = 10.0.^(-0.75:-0.25:-4.75)
    # For N = 1 this works
    ks = 10.0.^(-1.0:-0.25:-4.5)
    catas = map(k->find_rip(scattered_coef(-N:N,k), k*R, 2π/8, 2π/4), ks)
    if scale == :log10 || scale == :log
        plot(ks,[c[1] for c in catas], scale=:log10, xlims=(1e-5,1.0), ylims=(1e-3,1.0))
    else
        plot(ks,[c[1] for c in catas], ylims=(0.0,1.0), xlims=(0.0,0.2),
            xticks=([0.0,0.05,0.1,0.15,0.2],[L"0.00",L"0.05",L"0.10",L"0.15",L"0.20"]),
            yticks=([0.0,0.2,0.4,0.6,0.8,1.0],[L"0.0",L"0.2",L"0.4",L"0.6",L"0.8",L"1.0"])
        )
    end
    plot!(legend=false, grid=false, xlab=L"\textrm{wavenumber } (k)", ylab=L"\nu_{\textrm{rip}}", guidefont=Plots.font(14))
end

function plot_derivative(θ; fnc=z->log(abs(z)), clims=(-18,0))
    k = 1.0
    du_dr(kr)   = sum(n -> coefs[n]*exp(im*n*θ)*k*(hankelh1(n-1,k*r)-hankelh1(n+1,k*r))/2, indices(coefs,1))
    x = range(-50, stop=50, length=100)
    y = range(-5, stop=95, length=100)
    heatmap(x, y, (x,y)->fnc(du_dr(x.+im.*y)))
    # scatter!([real(r_crit)], [imag(r_crit)], label="")
    plot!(xlims=(minimum(x), maximum(x)), ylims=(minimum(y),maximum(y)), clims=clims)
end

function plot_single_mode_with_paths(n, clims=(-1.2,0.2))

    f(r) = hankel_field(r, 0.0, OffsetArray([1.0+0.0im],n:n), 1.0)
    plot_f(f; n_samples=500, xlims=(0,10), ylims=(0,10), arrow_fnc=z->0.2*z, clims=clims)
    plot!(xlab=L"Real($\tilde r$)", ylab=L"Imag($\tilde r$)", title="")
    plot!(colorbar_title=L"\log_{10}\left|\frac{\mathrm{d} \tilde r}{\mathrm{d}t}\right|")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    add_path_to_plot(f,0.1,N_steps=20000)
    add_path_to_plot(f,1.0,N_steps=20000)
    add_path_to_plot(f,2.0,N_steps=20000)
    add_path_to_plot(f,4.0,N_steps=20000)
    add_path_to_plot(f,8.0,N_steps=20000)

    savefig("images/single_mode_phase_diagram_$n.png")

    plot!()
end

function plot_two_mode_hankel_pole(θ)
    coefs = two_mode_pole_coef(0:1, 3.0+3.0*im)
    f(r,θ) = hankel_field(r, θ, coefs, 1.0)
    plot_f(f, 0.076; xlims=(0,10), n_samples=500, ylims=(0,10), arrow_fnc=z->0.2*z, clims=:auto)
    plot!(size=[800,600])
    add_path_to_plot(r->f(r,θ),2.0, N_steps=20000)

    plot!(xlab=L"Real($\tilde r$)", ylab=L"Imag($\tilde r$)", title="")
    plot!(colorbar_title=L"\log_{10}\left|\frac{\mathrm{d}\tilde r}{\mathrm{d}t}\right|")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    savefig("images/two_mode_pole_phase_diagram_theta_$(replace(string(θ), '.'=>'_')).png")

    plot!()
end

function plot_two_mode_pole_trajectories(θ₋, θ_crit, θ₊, N=11, δ = 1e-6)
    coefs = two_mode_pole_coef(0:1, 3.0+3.0*im)

    plot(xlims=(0,10), ylims=(0,10), grid=false, aspectratio=1.0)

    for (θ, color) in zip(range(θ₋, θ_crit-δ, length=N), cvec(:blues, 2N)[1:N])
        f(r,θ) = hankel_field(r, θ, coefs, 1.0)
        add_path_to_plot(r->f(r,θ), 2.0, N_steps=20000, color=color)
    end
    for (θ, color) in zip(range(θ_crit+δ, θ₊, length=N), cvec(:blues, 2N)[N+1:2N])
        f(r,θ) = hankel_field(r, θ, coefs, 1.0)
        add_path_to_plot(r->f(r,θ), 2.0, N_steps=20000, color=color)
    end

    plot!(xlab=L"Real($\tilde r$)", ylab=L"Imag($\tilde r$)", title="")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    plot!([1.8, 2.2], [5.2, 5.8], arrow=true, color=:black, label="")
    annotate!(1.8, 5.0, text(L"\theta=0.076", :top, :right, 20))
    annotate!(2.2, 6.0, text(L"\theta=0.078", :bottom, :left, 20))

    savefig("images/two_mode_pole_trajectories.png")

    plot!()
end

function plot_two_mode_pole_rip_3d(;r_pole=3.0+3.0*im, θ_pole=0.5,color=:viridis,ν_max = 1.0 - 1e-6, θ₋=0.0, θ₊=0.5)
    coefs = two_mode_pole_coef(0:1, r_pole, θ_pole)
    R = 2.0

    f(tν,θ) = hankel_field(R+tν, θ, coefs, 1.0)

    plot()

    δ = 1e-5
    θ_end = θ₊
    rips = find_rips(f, θ₋, θ₊, Nζ=21, ν=ν_max, ε=1e-3)

    for (i,rip) in enumerate(rips)
        θ₊ = rip.ζ-δ
        θs = range(θ₋, stop=θ₊, length=50)
        ν_vec = linlogspace(0.005, 0.5, stop=ν_max)
        r = Matrix{Complex{Float64}}(undef, length(ν_vec), length(θs))
        for (i,θ) in enumerate(θs)
            r[:,i] .= R .+ exact_mapping_solve_steps(tν->f(tν,θ), ν_vec=ν_vec)[1]
        end
        surface!(θs, ν_vec, imag.(r),color=color)
        θ₋ = rip.ζ+δ
    end

    θ₊ = θ_end
    θs = range(θ₋, stop=θ₊, length=50)
    ν_vec = linlogspace(0.005, 0.5, stop=ν_max)
    r = Matrix{Complex{Float64}}(undef, length(ν_vec), length(θs))
    for (i,θ) in enumerate(θs)
        r[:,i] .= R .+ exact_mapping_solve_steps(tν->f(tν,θ), ν_vec=ν_vec)[1]
    end
    surface!(θs, ν_vec, imag.(r),color=color)

    plot!(xlab=L"\theta", ylab=L"\nu", zlab=L"Imag($\tilde \nu$)", colorbar=false, title="")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    savefig("images/two_mode_pole_rip_3d.png")

    plot!()

end

function plot_hankel_rip_radial_3d(;color=:viridis,k=0.1)

    coefs = scattered_coef(-5:5, k)
    R = 2.0

    f(tν,θ) = hankel_field(R+tν, θ, coefs, k)

    plot()

    ν_max = 1.0 - 1e-5
    δ = 1e-5
    θ_start = 0.0
    θ_end = 2π
    rips = find_rips(f, θ_start, θ_end, Nζ=21, ν=ν_max, ε=1e-3)

    function plotbetween!(θ₋, θ₊)
        θ_delta_max = (θ_end - θ_start)/100
        θs_length = max((θ₊ - θ₋)/θ_delta_max |> abs |> ceil |> Int, 5)
        θs = range(θ₋, stop=θ₊, length=θs_length)
        νs = linlogspace(0.005, 0.5, stop=ν_max)
        rs_prev = Vector{Complex{Float64}}(undef, length(νs))
        rs_current = Vector{Complex{Float64}}(undef, length(νs))

        rs_prev .= R .+ exact_mapping_solve_adaptive_steps(tν->f(tν,θs[1]), νs)
        θ_prev = θs[1]
        for θ in θs[2:end]
            rs_current .= R .+ exact_mapping_solve_adaptive_steps(tν->f(tν,θ), νs)

            x = vcat([(R+ν)*cos(θ_prev) for ν in νs], [(R+ν)*cos(θ) for ν in νs])
            y = vcat([(R+ν)*sin(θ_prev) for ν in νs], [(R+ν)*sin(θ) for ν in νs])

            # x = vcat(real.(rs_prev).*cos(θ_prev), real.(rs_current).*cos(θ))
            # y = vcat(real.(rs_prev).*sin(θ_prev), real.(rs_current).*sin(θ))

            z = vcat(imag.(rs_prev), imag.(rs_current))
            surface!(x, y, z, color=color)

            rs_prev .= rs_current
            θ_prev = θ
        end
    end

    if length(rips) == 0
        plotbetween!(θ_start, θ_end)
    else
        θ₋ = θ_start
        for (i,rip) in enumerate(rips)
            θ₊ = rip.ζ-δ
            plotbetween!(θ₋, θ₊)
            θ₋  = rip.ζ+δ
        end
        plotbetween!(θ₋, θ_end)
    end

    plot!()

    # plot!(xlab=L"Real($\tilde x$)", ylab=L"Real($\tilde y$)", zlab=L"Imag($\tilde r$)", colorbar=false, title="")
    plot!(xlab=L"x", ylab=L"y", zlab=L"Imag($\tilde r$)", colorbar=false, title="")

    plot!(size=[1200,600], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

end


function plot_scattered_rip_3d_pml_coords(;color=:viridis,k=0.1)

    coefs = scattered_coef(-5:5, k)
    R = 2.0

    f(tν,θ) = hankel_field(R+tν, θ, coefs, k)

    plot()

    ν_max = 1.0 - 1e-2
    δ = 1e-6
    θ_start = 1.41
    θ_end = 1.43
    θ₋ = θ_start
    θ₊ = θ_end
    rips = find_rips(f, θ₋, θ₊, Nζ=21, ν=ν_max, ε=1e-3)

    ν_vec = linlogspace(0.001, 0.5, stop=ν_max)
    plot_inds = findfirst(ν-> ν>=0.65, ν_vec):findfirst(ν-> ν>=0.75, ν_vec)
    ν_vec_plot = ν_vec[plot_inds]

    for (i,rip) in enumerate(rips)
        θ₊ = rip.ζ-δ
        θs = range(θ₋, stop=θ₊, length=50)
        r = Matrix{Complex{Float64}}(undef, length(ν_vec), length(θs))
        for (i,θ) in enumerate(θs)
            r[:,i] .= R .+ exact_mapping_solve_adaptive_steps(tν->f(tν,θ), ν_vec)
        end
        surface!(θs, ν_vec[plot_inds], imag.(r[plot_inds,:]),color=color)
        θ₋ = rip.ζ+δ
    end

    θ₊ = θ_end
    θs = range(θ₋, stop=θ₊, length=50)
    r = Matrix{Complex{Float64}}(undef, length(ν_vec), length(θs))
    for (i,θ) in enumerate(θs)
        r[:,i] .= R .+ exact_mapping_solve_adaptive_steps(tν->f(tν,θ), ν_vec)
    end
    surface!(θs, ν_vec[plot_inds], imag.(r[plot_inds,:]),color=color)

    plot!(xlab=L"\theta", ylab=L"\nu", zlab=L"Imag($\tilde \nu$)", colorbar=false, title="")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")
    # Get other side angle, for some reason this turns to azim=-135, elev=30 in pyplot??
    plot!(camera=(-45,30))
    savefig("images/scattered_rip_3d_pml_coords_zoomed.png")

    plot!()

end

"Explains why Hankel mode 2 at k=1 does not converge at ν=1, hits a pole in complex plane"
function plot_radius_of_convergence_study()
    coefs = OffsetArray([1.0+0.0im],2:2)

    f(r) = hankel_field(r, 0.0, coefs,  1.0)

    plot_f(f, xlims=(0,3), ylims=(-2,1), n_arrows=0, n_samples=500)

    function fnc!(F, r)
        F_c = f(r[1] + r[2]*im).du_dtν
        F[1] = real(F_c)
        F[2] = imag(F_c)
    end
    rt_pole_vec = nlsolve(fnc!, [1.5, -1.0]).zero
    rt_pole = rt_pole_vec[1] + rt_pole_vec[2]*im

    u_pole = f(rt_pole).u
    U = f(1.0+0.0im).u
    ν_pole = 1 - u_pole/U

    add_path_to_plot_nu_steps(f, 1.0+0.0im, h=exp(angle(ν_pole)*im)*1e-4, ν_abs_max=abs(ν_pole))
    annotate!(real(rt_pole), imag(rt_pole), text("\$\\nu=$(round(abs(ν_pole),digits=2))e^{$(round(angle(ν_pole),digits=2))i}\$", :left, :top, 20))
    scatter!([real(rt_pole)], [imag(rt_pole)], marker=(:circle, :white), label="")

    annotate!(1.0, 0.0+0.02, text(L"\nu=0",:bottom, 20))
    scatter!([1.0], [0.0], marker=(:circle, :white), label="")

    tν_vec = Vector{Complex{Float64}}(undef,1)
    add_path_to_plot_nu_steps(f, 1.0+0.0im, tν_vec, h=exp(-0.69*im)*1e-3)
    annotate!(real(tν_vec[end])-0.1, imag(tν_vec[end])-0.05, text(L"\nu=e^{-0.69i}\;",:top, 20))
    scatter!([real(tν_vec[end])], [imag(tν_vec[end])], marker=(:circle, :white), label="")

    tν_vec = Vector{Complex{Float64}}(undef,1)
    add_path_to_plot_nu_steps(f, 1.0+0.0im, tν_vec, h=exp(-0.71*im)*1e-3)
    annotate!(real(tν_vec[end]), imag(tν_vec[end])+0.02, text(L"\nu=e^{-0.71i}\;",:bottom, 20))
    scatter!([real(tν_vec[end])], [imag(tν_vec[end])], marker=(:circle, :white), label="")

    plot!(xlab=L"Real($\tilde r$)", ylab=L"Imag($\tilde r$)", title="")
    plot!(colorbar_title=L"\log_{10}\left|\frac{\mathrm{d} \tilde r}{\mathrm{d}t}\right|")

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    savefig("radius_of_convergence_study.png")
    plot!()

end

function plot_example_transformation()

    coefs = OffsetArray([1.0+0.0im],2:2)

    R = 2.0
    field_fnc(tν) = hankel_field(R+tν, 0.0, coefs,  1.0)

    ν_vec = Float64[]; tν_vec = Complex{Float64}[];
    exact_mapping_solve_adaptive_steps(field_fnc, 0.9999, ν_vec, tν_vec, h_max=0.01)

    u(tν) = field_fnc(tν).u
    U = u(0.0+0.0im)
    u_trans_normalised = u.(tν_vec)./U

    plot(
        plot(
            ν_vec, [real.(R .+ tν_vec) imag.(tν_vec)], label=["Re Optimal" "Im Optimal"],
             xlab=L"\nu", ylab=L"\tilde r", grid=false, xlims=(0,1),
             legend=:topleft, linestyle=[:solid :dash], color=[:blue :orange]
        ),
        plot(
            ν_vec, [real.(u_trans_normalised) imag.(u_trans_normalised)], label=["Re Optimal" "Im Optimal"],
             xlab=L"\nu", ylab=L"u/U", grid=false, xlims=(0,1),
             legend=:topright, linestyle=[:solid :dash], color=[:blue :orange]
        )
    )
    plot!(size(1000,500))

    filename="example_transformation"
    savefig("$filename.tex")
    run(`sed -i 's|{152.4mm}|{0.9\\textwidth}|g' $filename.tex`)
    run(`sed -i 's|{101.6mm}|{0.4\\textwidth}|g' $filename.tex`)

    savefig("$filename.pdf")


    return nothing

end

function plot_scattered_example_transformation()

    k = 1.0
    coefs = scattered_coef(-10:10, k)

    R = 2.0
    θ = 0.0
    field_fnc(tν) = hankel_field(R+tν, θ, coefs,  k)

    ν_vec = Float64[]

    tν_opt_vec = Complex{Float64}[]
    exact_mapping_solve_adaptive_steps(field_fnc, 0.9999, ν_vec, tν_opt_vec, h_max=0.01)

    δ = 0.5
    tν_sfberm_vec =             (-im/k).*log.(1 .- ν_vec)
    tν_berm_vec   = δ.*ν_vec .+ (-im/k).*log.(1 .- ν_vec)

    u(tν) = field_fnc(tν).u
    U = u(0.0+0.0im)

    u_opt_error    = u.(tν_opt_vec)./U    .- (1 .- ν_vec)
    u_sfberm_error = u.(tν_sfberm_vec)./U .- (1 .- ν_vec)
    u_berm_error   = u.(tν_berm_vec)./U   .- (1 .- ν_vec)

    plot(
        plot(
            ν_vec, [real.(R .+ tν_opt_vec) imag.(tν_opt_vec) real.(R .+ tν_sfberm_vec) imag.(tν_sfberm_vec) real.(R .+ tν_berm_vec) imag.(tν_berm_vec)],
            label=["Re Optimal" "Im Optimal" "Re SFB" "Im SFB" "Re Bermudez" "Im Bermudez"],
            linestyle=[:solid :solid :dash :dash :dot :dot],
            color=[:blue :orange :blue :orange :blue :orange],
            xlab=L"\nu", ylab=L"\tilde r", grid=false, xlims=(0,1),
            legend=:topleft,
        ),
        plot(
            ν_vec, [real.(u_opt_error) imag.(u_opt_error) real.(u_sfberm_error) imag.(u_sfberm_error) real.(u_berm_error) imag.(u_berm_error)],
            label=["Re Optimal" "Im Optimal" "Re SFB" "Im SFB" "Re Bermudez" "Im Bermudez"],
            linestyle=[:solid :solid :dash :dash :dot :dot],
            color=[:blue :orange :blue :orange :blue :orange],
            xlab=L"\nu", ylab=L"u(\tilde r)/U - (1-\nu)", grid=false, xlims=(0,1),
            legend=:bottomleft,
        )
    )

    plot!(size(1000,500))

    filename="scattered_example_transformation_theta"
    savefig("$filename.tex")
    run(`sed -i 's|{152.4mm}|{0.9\\textwidth}|g' $filename.tex`)
    run(`sed -i 's|{101.6mm}|{0.4\\textwidth}|g' $filename.tex`)

    savefig("$filename.pdf")

    return nothing

end

function plot_error_against_n_points_integrating_through()

    coefs = OffsetArray([1.0+0.0im],2:2)

    R = 2.0
    field_fnc(tν) = hankel_field(R+tν, 0.0, coefs,  1.0)

    n_max = 1000

    integrand(ν, dtν_dν) = 1.0/dtν_dν + ((1-ν)^2)*dtν_dν
    # integrand(ν, dtν_dν) = (1-ν)/dtν_dν

    function integrate(n)

        # Get Gauss-Legendre knot points and weights, and transform to [0,1]
        nodes, weights = gausslegendre(n)
        nodes .= (nodes .+ 1)/2
        weights .= weights./2

        # Initialise stepping through and integrating
        integral = 0.0 + 0.0im
        ν_prev = 0.0
        tν_prev = 0.0 + 0.0im
        U_field = field_fnc(0.0+0.0im)
        field_prev = U_field

        # Step through PML, adding contributions to integral
        for i = 1:n
            ν = nodes[i]
            tν_prev, dtν_dν, dtν_dζ, ν_prev, field_prev = exact_mapping_solve_adaptive_steps(field_fnc, ν;ν0=ν_prev, tν0=tν_prev, field0=field_prev, U_field=U_field, householder_order=3)
            integral += weights[i]*integrand(ν, dtν_dν)
        end
        return integral
    end

    println("Calculating exact with $(2n_max) points")
    exact = integrate(2n_max)

    function integrate_trap(n)

        h = 1/n

        # Initialise stepping through and integrating
        integral = 0.0 + 0.0im
        ν_prev = 0.0
        tν_prev = 0.0 + 0.0im
        U_field = field_fnc(0.0+0.0im)
        field_prev = U_field

        # Step through PML, adding contributions to integral
        ν = 0.0
        tν_prev, dtν_dν, dtν_dζ, ν_prev, field_prev = exact_mapping_solve_adaptive_steps(field_fnc, ν;ν0=ν_prev, tν0=tν_prev, field0=field_prev, U_field=U_field, householder_order=3)
        integral += h/2*(1-ν)/dtν_dν
        for i = 1:n-1
            ν = i*h
            tν_prev, dtν_dν, dtν_dζ, ν_prev, field_prev = exact_mapping_solve_adaptive_steps(field_fnc, ν;ν0=ν_prev, tν0=tν_prev, field0=field_prev, U_field=U_field, householder_order=3)
            integral += h*integrand(ν, dtν_dν)
        end
        return integral
    end

    n_trap = 10^7
    println("Calculating trapezoidal exact with $(n_trap) points")
    exact_trap = integrate_trap(n_trap)

    println("How can we be sure the exact solution is good?")
    println("Compare it to a simpler and more robust method: the trapezoidal rule")
    println("exact = $exact  exact_trap = $exact_trap  abs diff = $(abs(exact_trap-exact))")

    println("Calculating integrals with different number of knots")
    errors = abs.(integrate.(1:n_max) .- exact)./abs(exact) .+ 1e-16

    plot(1:n_max, errors, xlab="Number of knot points", ylab="Relative error",
        scale=:log10, grid=false, legend=:bottomleft, xticks=[1,10,100,1000],
        label="Gauss-Legendre", color=:purple, ylims=(1e-15, 1)
    )
    plot!(1:100, 1e-3.*(1:100).^(-5), linestyle=:dash, label=L"O(N^{-5})", color=:purple)

    filename="rel_error_against_n_points_integrating_through"
    savefig("$filename.tex")
    run(`sed -i 's|{152.4mm}|{0.8\\textwidth}|g' $filename.tex`)
    run(`sed -i 's|{101.6mm}|{0.6\\textwidth}|g' $filename.tex`)

    savefig("$filename.pdf")

    return plot!()
end


function plot_intro_abs_residual_fixed_nu()

    coefs = OffsetArray([1.0+0.0im],2:2)

    R = 2.0
    field_fnc(tν) = hankel_field(R+tν, 0.0, coefs,  1.0)

    plot_abs_residual_fixed_nu(field_fnc, 0.1; color_fnc=z->log10(z),
        xlims=(-2,2), ylims=(-2,2),
        n_samples=500, clims=:auto, title="", colorbar=true,
        size=(800,700)
    )
    scatter!([0.0], [0.0], marker=:star, markersize=8.0, color=:white, markerstrokecolor=:transparent, label="")
    annotate!(0.0, 0.0, text(L"\tilde \nu = 0", :top, :left, :white, 20))

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    savefig("images/intro_abs_residual_fixed_nu.png")
    plot!()
end

function plot_abs_residual_fixed_nu_scat(;k=0.1, plotq=false)

    coefs = scattered_coef(-5:5, k)

    R = 2.0
    θ = 1.4202
    field_fnc(tν) = hankel_field(R+tν, θ, coefs,  k)

    ν = 0.705
    ν0 = 0.700

    if plotq
        plot_objective(field_fnc; color_fnc=z->log10(z),
            xlims=(-3,3), ylims=(-2,4), n_samples=1000, clims=:auto, title="",
            colorbar=true, size=(800,700)
        )
    else
        plot_abs_residual_fixed_nu(field_fnc, ν; color_fnc=z->log10(z),
            xlims=(-3,3), ylims=(-2,4), n_samples=500, clims=:auto, title="",
            colorbar=true, size=(800,700)
        )
    end


    tν0 = exact_mapping_solve_adaptive_steps(field_fnc, ν0)[1]

    scatter!([real(tν0)], [imag(tν0)], marker=:star, markersize=8.0, color=:white, markerstrokecolor=:transparent, label="")
    annotate!(real(tν0), imag(tν0), text("\$\\tilde \\nu($ν0)\$", :top, :left, :white, 20))

    plot!(size=[1200,800], tickfontsize=18, guidefontsize=20, dpi=200)
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end
    plot!(fontfamily="Latin Modern Math")

    if plotq
        savefig("images/abs_q_scat.png")
    else
        savefig("images/abs_residual_fixed_nu_scat.png")
    end

    plot!()
end

function plot_intro_abs_q()

    # coefs = OffsetArray([1.0+0.0im],2:2)
    coefs = scattered_coef(-5:5, 0.1)

    R = 2.0
    field_fnc(tν) = hankel_field(R+tν, 0.0, coefs,  1.0)

    plot_objective(field_fnc; color_fnc=z->log10(z),
        xlims=(-5,5), ylims=(-5,5),
        n_samples=500, clims=:auto, title="", colorbar=true,
        size=(800,700)
    )
    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end

    plot!(tickfontsize=27, guidefontsize=30, dpi=200)
    plot!(fontfamily="Latin Modern Math")

    savefig("images/intro_abs_q_zoomed.png")
    plot!()
end

function plot_objective_near_bifurcation(; n_samples=200)

    k = 0.1
    R = 2.0
    coefs = scattered_coef(-5:5, k)
    field_fnc(tν,θ) = hankel_field(R+tν, θ, coefs, k)

    ν_max = 1.0 - 1e-6

    θ₋ = 1.4
    θ_end = 1.5
    θ₊ = θ_end
    rips = find_rips((tν,ζ)->field_fnc(tν,ζ), θ₋, θ₊, Nζ=21, ν=ν_max, ε=1e-3)
    rip = rips[1]

    clims=(-9,-1)
    # clims=:auto

    xlims=(-5,5)
    ylims=(-5,5)

    δ_zoomed = 0.2
    xlims_zoomed=(real(rip.tν)-δ_zoomed,real(rip.tν)+δ_zoomed)
    ylims_zoomed=(imag(rip.tν)-δ_zoomed,imag(rip.tν)+δ_zoomed)

    plot_objective(tν->field_fnc(tν,rip.ζ), clims=clims,
        xlims=xlims, ylims=ylims, n_samples=n_samples, title="", colorbar=false, size=(1000,800))


    if backend() == Plots.PyPlotBackend()
        PyPlot.matplotlib.rc("mathtext",fontset="cm")
    end

    plot!(tickfontsize=27, guidefontsize=30, dpi=200)
    plot!(fontfamily="Latin Modern Math")

    # Plot a square where we are going to zoom in
    plot!(Plots.Shape([(xlims_zoomed[1],ylims_zoomed[1]),
                       (xlims_zoomed[2],ylims_zoomed[1]),
                       (xlims_zoomed[2],ylims_zoomed[2]),
                       (xlims_zoomed[1],ylims_zoomed[2]),
                       (xlims_zoomed[1],ylims_zoomed[1]) ]), color=:transparent, label="")

    savefig("images/objective_on_bifurcation.png")

    plot_objective(tν->field_fnc(tν,rip.ζ), clims=clims,
        xlims=xlims_zoomed, ylims=ylims_zoomed, n_samples=n_samples, title="", size=(1000,800))


    plot!(tickfontsize=27, guidefontsize=30, dpi=200, right_margin=Plots.Length(:mm,50))
    plot!(fontfamily="Latin Modern Math")

    savefig("images/objective_on_bifurcation_zoomed.png")


    δ = 1e-6

    ζ₋ = rip.ζ-δ

    plot_objective(tν->field_fnc(tν,ζ₋), clims=clims,
        xlims=xlims_zoomed, ylims=ylims_zoomed, n_samples=n_samples, title="", colorbar=false, size=(1000,800))

    ν_vec = 0.0:0.0005:0.9
    tν_vec, _ = exact_mapping_solve_steps(tν->field_fnc(tν,ζ₋), ν_vec=ν_vec)
    plot!(real.(tν_vec), imag.(tν_vec), marker=(:white, :star, 16), markerstrokecolor=:transparent, color=:white, line=(:dot,4), label="")
    plot!(tickfontsize=27, guidefontsize=30, dpi=200)
    plot!(fontfamily="Latin Modern Math")
    savefig("images/objective_below_bifurcation_zoomed_with_basic_method.png")

    ζ₊ = rip.ζ+δ

    plot_objective(tν->field_fnc(tν,ζ₊), clims=clims,
        xlims=xlims_zoomed, ylims=ylims_zoomed, n_samples=n_samples, title="", colorbar=false, size=(1000,800))

    ν_vec = 0.0:0.0005:0.9
    tν_vec, _ = exact_mapping_solve_steps(tν->field_fnc(tν,ζ₊), ν_vec=ν_vec)
    plot!(real.(tν_vec), imag.(tν_vec), marker=(:white, :star, 16), markerstrokecolor=:transparent, color=:white, line=(:dot,4), label="")
    plot!(tickfontsize=27, guidefontsize=30, dpi=200)
    plot!(fontfamily="Latin Modern Math")
    savefig("images/objective_above_bifurcation_zoomed_with_basic_method.png")

    plot_objective(tν->field_fnc(tν,ζ₋), clims=clims,
        xlims=xlims_zoomed, ylims=ylims_zoomed, n_samples=n_samples, title="", colorbar=false, size=(1000,800))

    ν_vec₋ = [0.0]; tν_vec₋ = [0.0+0.0im]; dtν_dν_vec₋ = [0.0+0.0im]
    exact_mapping_solve_adaptive_steps(tν->field_fnc(tν,ζ₋), ν_max, ν_vec₋, tν_vec₋, dtν_dν_vec₋)
    plot!(real.(tν_vec₋), imag.(tν_vec₋), marker=(:white, :star, 16), markerstrokecolor=:transparent, color=:white, line=(:dot,4), label="")
    plot!(tickfontsize=27, guidefontsize=30, dpi=200)
    plot!(fontfamily="Latin Modern Math")
    savefig("images/objective_below_bifurcation_zoomed.png")


    plot_objective(tν->field_fnc(tν,ζ₊), clims=clims,
        xlims=xlims_zoomed, ylims=ylims_zoomed, n_samples=n_samples, title="", size=(1000,800))

    ν_vec₊ = [0.0]; tν_vec₊ = [0.0+0.0im]; dtν_dν_vec₊ = [0.0+0.0im]
    exact_mapping_solve_adaptive_steps(tν->field_fnc(tν,ζ₊), ν_max, ν_vec₊, tν_vec₊, dtν_dν_vec₊)
    plot!(real.(tν_vec₊), imag.(tν_vec₊), marker=(:white, :star, 16), markerstrokecolor=:transparent, color=:white, line=(:dot,4), label="")
    plot!(tickfontsize=27, guidefontsize=30, dpi=200, right_margin=Plots.Length(:mm,50))
    plot!(fontfamily="Latin Modern Math")
    savefig("images/objective_above_bifurcation_zoomed.png")


end
