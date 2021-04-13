using Plots

using LaTeXStrings

include("plotutils.jl")

f_naughty(x;a=0.5,x_crit=0.0) = real(complex(x-x_crit)^-a) * (1 + 10x - 8x^2)

int_f_naughty(x;a=0.5) = x^-a * (x/(1-a) + 10x^2/(2-a) - 8x^3/(3-a))

f_naughty(x,y;a=0.5,x_crit=(0.0,0.0)) = sqrt((x-x_crit[1])^2+(y-x_crit[2])^2)^-a * (1 + 10x - 8x^2) * (2 - 5x + 2x^2)

function f_rip(x,y;x_crit=(0.0,0.0))
    z = -(x-x_crit[1]) + (y-x_crit[2])*im
    -1/sqrt(z)
end

f_rip_re(x,y;kwargs...) = real(f_rip(x,y;kwargs...))
f_rip_im(x,y;kwargs...) = imag(f_rip(x,y;kwargs...))

function plot_gausslegendretrans_mid_knots(x_crit=(0.7,0.2))
    scatter([(gausslegendretrans_mid(n,nodes,weights,(0.7,0.2))[1].*2 .-(1,1)) for n in 1:400], markersize=1)
    plot!(grid=false, legend=false, xlab=L"s_1", ylab=L"s_2")
    return backend_save_and_return(backend(), "../images/plot_gausslegendretrans_mid_knots")
end

"Plot comparing error with and without transformed scheme in 1d with singularity at 0"
function plot_error_against_n_points_1d()
    N = 1:1000
    int_gauss_err = (abs.(int_gauss.(f_naughty,N) .- int_f_naughty(1.0)) .+ 1e-16)./abs(int_f_naughty(1.0))
    int_gauss_trans_err = (abs.(int_gauss_trans.(f_naughty,N) .- int_f_naughty(1.0)) .+ 1e-16)./abs(int_f_naughty(1.0))
    plot(N, [int_gauss_err, int_gauss_trans_err], label=["Gauss-Legendre" "Transformed Gauss-Legendre"],
         xlab="N", ylab="Relative error", scale=:log10, grid=false)
    return backend_save_and_return(backend(), "../images/rel_error_transformed_quadrature_1d")
end

"Plot comparing error with and without transformed scheme in 2d with singularity in corner"
function plot_error_against_n_points_2d_corner()
    N = 1:1000
    exact = int_gauss_trans_2d.(f_naughty,2000)
    int_gauss_err = (abs.(int_gauss_2d.(f_naughty,N) .- exact) .+ 1e-16)./abs(exact)
    int_gauss_trans_err = (abs.(int_gauss_trans_2d.(f_naughty,N) .- exact) .+ 1e-16)./abs(exact)
    plot(N, [int_gauss_err, int_gauss_trans_err], label=["Gauss-Legendre" "Transformed Gauss-Legendre"],
         xlab="N", ylab="Relative error", scale=:log10, grid=false)
    return backend_save_and_return(backend(), "../images/rel_error_transformed_quadrature_corner")
end

"Plot comparing error with and without transformed scheme in 2d with singularity in interior"
function plot_error_against_n_points_2d_mid()
    x_crit = (0.7,0.4)
    a = 0.5
    f(x,y) = f_rip_im(x, y, x_crit=x_crit)
    N = 1:250
    exact = int_gauss_trans_2d_mid.(f, 500, x_crit=x_crit)
    int_gauss_err = (abs.(int_gauss_2d.(f,N) .- exact) .+ 1e-16)./abs(exact)

    plot(xlab="Number of knot points", ylab="Relative error", scale=:log10,
         grid=false, legend=:bottomleft, xticks=[1,10,100],ylims=(0.6e-15,4))

    int_gauss_trans_err = (abs.(int_gauss_trans_2d_mid.(f,N,x_crit=x_crit) .- exact) .+ 1e-16)./abs(exact)
    plot!(N, int_gauss_err, label="Gauss-Legendre", linestyle=:solid, color=:purple)
    plot!(1:250, (1:250).^(-1), label=L"O(N^{-1})", linestyle=:dash, color=:purple)

    plot!(N, int_gauss_trans_err, label="Transformed Gauss-Legendre", linestyle=:solid, color=RGB(0.2,0.8,0.2))
    plot!(1:250, (1:250).^(-6), label=L"O(N^{-6})", linestyle=:dash, color=RGB(0.2,0.8,0.2))

    return backend_save_and_return(backend(), "../images/rel_error_transformed_quadrature_im")
end

function plot_f_rip()
    ν = 0:0.01:1
    f(ν,ζ) = f_rip_im(ν,ζ;x_crit=(0.5,0.0))
    ε = 0.0001
    plot(xlab=L"\nu", ylab=L"\zeta", zlab=L"Im($\frac{\partial \tilde \nu}{\partial \nu}$)", zlim=(-5,5), clims=(-5,5), colorbar=false)
    surface!(ν, (-1:0.01:0).-ε, f)
    surface!(ν, ( 0:0.01:1).+ε, f)
    return backend_save_and_return(backend(), "../images/dtnu_dnu_rip_surface")
end
