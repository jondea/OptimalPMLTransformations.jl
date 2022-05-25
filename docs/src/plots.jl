import OptimalPMLTransformations: HankelSeries, two_mode_pole_coef, PolarCoordinates, PMLCoordinates, AnnularPML, optimal_pml_transformation_solve, find_rips
using Plots

include("complex_plot_utils.jl")

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

function plot_trajectories(;
    R = 2.0,
    δ = 1.0,
    θ = 0.0,
    k = 1.0,
    a = two_mode_pole_coef(0:1, 2.0+1.0im),
    xlims=(-50,50),
    ylims=(-50,50),
    n_samples=200,)

    plot()

    function u(tr)
        return HankelSeries(k, a)(PolarCoordinates(tr,θ))
    end

    complex_abs_contour!(u; xlims, ylims, n_samples)
    complex_angle_heatmap!(u; xlims, ylims, n_samples)

end
