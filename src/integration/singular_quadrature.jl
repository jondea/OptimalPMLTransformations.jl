
using FastGaussQuadrature
using LinearAlgebra

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
