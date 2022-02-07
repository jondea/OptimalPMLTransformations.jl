
import FastGaussQuadrature: gausslegendre

"N Gauss-Legendre nodes and weights, but rescaled for integration from 0 to 1"
function gausslegendreunit(N)
    nodes, weights = gausslegendre(N)
    nodes .= (nodes .+ 1)/2
    weights .= weights./2
    return nodes, weights
end

"Integrate f from 0 to 1 with N Gauss-Legendre nodes"
function int_gauss(f::Function,N::Int)
    nodes, weights = gausslegendreunit(N)
    sum( weights .* f.(nodes) )
end

"Integrate f from x, y = 0 to 1 with N Gauss-Legendre nodes in each direction"
function int_gauss_2d(f::Function,N::Int)
    nodes, weights = gausslegendreunit(N)
    sum( weights .* weights' .* f.(nodes, nodes') )
end

"N Gauss-Legendre nodes and weights transformed to integrate x^-a singularity at 0"
function gausslegendreunittrans(N::Int,a::Real)
    nodes, weights = gausslegendreunit(N)

    weights .= weights.*(nodes.^(1/a-1))./a
    nodes .= nodes.^(1/a)
    return nodes, weights
end

"""
Integrate f from 0 to 1 with N Gauss-Legendre nodes, transformed for x^-a
singularity at 0
"""
function int_gauss_trans(f::Function,N::Int;a=0.5)
    nodes, weights = gausslegendreunittrans(N,a)
    sum( weights .* f.(nodes) )
end

"""
Integrate f from x,y = 0 to 1 with N Gauss-Legendre nodes in each direction,
transformed for x^-a singularity at 0
"""
function int_gauss_trans_2d(f::Function,N::Int;a=0.5)
    nodes, weights = gausslegendreunittrans(N,a)
    sum( weights .* weights' .* f.(nodes, nodes') )
end

"""
n-th node and weight for a Gauss-Legendre scheme transformed for a singularity
|(x-x_crit)^-a| using 2N nodes.
"""
function gausslegendretrans_mid(n::Int, nodes::Vector, weights::Vector, x_crit::Number)
    N = length(nodes)

    if 0 < n <= N
        # Left half
        node = (1-nodes[N-n+1])*x_crit
        weight = weights[N-n+1]*x_crit
    elseif N < n <= 2N
        # Right half
        node = x_crit + nodes[n-N]*(1-x_crit)
        weight = weights[n-N]*(1-x_crit)
    else
        error("There are only $(2N) nodes, you asked for $n")
    end
    return node, weight
end

"""
Integrate f for x = 0 to 1 using a Gauss-Legendre scheme transformed for a
singularity |(x-x_crit)^-a| using 2N nodes.
"""
function int_gauss_trans_mid(f::Function, N::Int;a=0.5, x_crit=0.7)
    nodes, weights = gausslegendreunittrans(N,a)
    int_f = 0.0
    for n in 1:2N
        node, weight = gausslegendretrans_mid(n, nodes, weights, x_crit)
        int_f += weight * f(node)
    end
    return int_f
end

"""
n-th node and weight for a Gauss-Legendre scheme transformed for a singularity
|(x-x_crit)^-a|+|(y-y_crit)^-a| using 4N^2 nodes.
"""
function gausslegendretrans_mid(n::Int, nodes::Vector, weights::Vector, x_crit::Tuple{Number,Number})
    N = length(nodes)

    if 0 < n <= 2(N^2)
        # Left half
        if iseven(div(n-1,N))
            # Bottom left quadrant
            nx = N-div(n-1,2N)
            ny = N-rem(n-1,2N)
            node = ((1-nodes[nx])*x_crit[1], (1-nodes[ny])*x_crit[2])
            weight = weights[nx]*weights[ny]*x_crit[1]*x_crit[2]
        else
            # Top left quadrant
            nx = N-div(n-1,2N)
            ny = rem(n-N,2N)
            node = ((1-nodes[nx])*x_crit[1], x_crit[2]+nodes[ny]*(1-x_crit[2]))
            weight = weights[nx]*weights[ny]*x_crit[1]*(1-x_crit[2])
        end
    elseif 2(N^2) < n <= 4(N^2)
        # Right half
        if iseven(div(n-1,N))
            # Bottom right quadrant
            nx = div(n-1,2N)-N+1
            ny = N-rem(n-1,2N)
            node = (x_crit[1]+nodes[nx]*(1-x_crit[1]), (1-nodes[ny])*x_crit[2])
            weight = weights[nx]*weights[ny]*(1-x_crit[1])*x_crit[2]
        else
            # Top right quadrant
            nx = div(n-1,2N)-N+1
            ny = rem(n-N,2N)
            node = (x_crit[1]+nodes[nx]*(1-x_crit[1]), x_crit[2]+nodes[ny]*(1-x_crit[2]))
            weight = weights[nx]*weights[ny]*(1-x_crit[1])*(1-x_crit[2])
        end
    else
        error("There are only $(2N) nodes, you asked for $n")
    end

    return node, weight
end

"""
Integrate f over x,y = 0 to 1 using a Gauss-Legendre scheme transformed for a
singularity |(x-x_crit)^-a|+|(y-y_crit)^-a| using 4N^2 nodes.
"""
function int_gauss_trans_2d_mid(f::Function, N::Int; a=0.5, x_crit=(0.7,0.2))
    nodes, weights = gausslegendreunittrans(N,a)
    int_f = 0.0
    for n in 1:4(N^2)
        node, weight = gausslegendretrans_mid(n, nodes, weights, x_crit)
        int_f += weight * f(node[1],node[2])
    end
    return int_f
end

function integrate_quad(u::AbstractFieldFunction, pml::PMLGeometry, integrand::Function, ζ_range, integration_order)
    field_fnc_νζ(tν, ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)},
        PMLCoordinates(tν,ζ), pml)

    rips = find_rips(field_fnc_νζ, ζ_range[1], ζ_range[2], Nζ=21, ν=0.999, ε=1e-3)

    ν_crit = only(rips).ν
    ζ_crit = only(rips).ζ

    ζ_width = ζ_range[2]-ζ_range[1]

    # Find singularity
    s_crit = ((ζ_crit - ζ_range[1])/ζ_width, ν_crit)

    # Get Gauss-Legendre knot points and weights, and transform to [0,1]
    nodes, weights = gausslegendre(integration_order)
    nodes .= (nodes .+ 1)/2
    weights .= weights./2

    integral = 0.0 + 0.0im

    n_knot = 1
    for _ in 1:2integration_order # Loop around ζ
        # Initialise stepping through and integrating
        trans_node, _ = gausslegendretrans_mid(n_knot, nodes, weights, s_crit)
        ζ = ζ_range[1] + trans_node[1] * ζ_width
        ν = 0.0
        ν_prev = ν
        tν = 0.0 + 0.0im
        tν_prev = tν
        field_fnc_ν(tν) = field_fnc_νζ(tν, ζ)
        U_field = field_fnc_ν(0.0+0.0im)
        field = U_field
        for _ in 1:2integration_order # Loop around ν
            trans_node, trans_weight = gausslegendretrans_mid(n_knot, nodes, weights, s_crit)
            ν = trans_node[2]
            tν, ∂tν_∂ν, ∂tν_∂ζ, ν_prev, field = optimal_pml_transformation_solve(field_fnc_ν, ν;
             ν0=ν_prev, tν0=tν_prev, field0=field, U_field=U_field, householder_order=3)
            integral += trans_weight*integrand(ν, ∂tν_∂ν)*ζ_width

            n_knot += 1
            ν_prev = ν
            tν_prev = tν
        end
    end
    return integral
end
