
struct Rip2D
    ζ::Float64
    ν::Float64
    tν::Complex{Float64}
end

"Use Newton's method to find a single rip from an initial guess"
function find_rip(u_pml::PMLFieldFunction, ν::Real, ζ::Real, tν::Number; ε=1e-12)

    x = SA[ν, ζ, real(tν), imag(tν)]

	U, ∂U_∂tζ = u_pml(NamedTuple{(:u, :∂u_∂tζ)}, 0.0 + 0.0im, ζ)
	(;u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ) = u_pml(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ)}, tν, ζ)

    # Create vector of residuals
	o_rip = ∂u_∂tν # ∂u_∂tν == 0 implies rip
	o_opt = u - U*(1-ν) # Ensures that tν stays on the optimal transformation
    r = SA[real(o_rip), imag(o_rip), real(o_opt), imag(o_opt)]

    n_iter = 1
    while maximum(abs.(r)) > ε*abs(U) && n_iter < 100
        # Create Jacobian of objectives and unknowns
        ∂o_opt_∂ζ = ∂u_∂tζ - ∂U_∂tζ*(1-ν)
        J = SA[
            0        real(∂2u_∂tν∂tζ)  real(∂2u_∂tν2)  -imag(∂2u_∂tν2);
            0        imag(∂2u_∂tν∂tζ)  imag(∂2u_∂tν2)   real(∂2u_∂tν2);
            real(U)  real(∂o_opt_∂ζ)   real(∂u_∂tν)    -imag(∂u_∂tν)  ;
            imag(U)  imag(∂o_opt_∂ζ)   imag(∂u_∂tν)     real(∂u_∂tν)
        ]

        # Perform Newton step
        x = x - J\r

        # Get values of unknowns from vector
        ν = x[1]
        ζ = x[2]
        tν = x[3] + im*x[4]

		# Recompute field and residual at new point
		U, ∂U_∂tζ = u_pml(NamedTuple{(:u, :∂u_∂tζ)}, 0.0 + 0.0im, ζ)
		u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ= u_pml(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ)}, tν, ζ)

		o_rip = ∂u_∂tν
		o_opt = u - U*(1-ν)
	    r = SA[real(o_rip), imag(o_rip), real(o_opt), imag(o_opt)]
        n_iter += 1
    end

    return Rip2D(ζ, ν, tν)
end

"Use Newton's method to find a single rip somewhere between two lines"
function find_rip(u_pml::PMLFieldFunction, tν₋::InterpLine, tν₊::InterpLine)
	# Find mostly likely point for rip along the two lines, which we will then use as an initial guess

    function rip_objective(p::InterpPoint)
        obj = abs((1-p.ν)*p.∂tν_∂ν)
        return isnan(obj) ? typeof(obj)(-Inf) : obj
    end

	p₋ = argmax(rip_objective, tν₋.points)
	p₊ = argmax(rip_objective, tν₊.points)
	ζ₋ = tν₋.ζ
	ζ₊ = tν₊.ζ
	νᵣ = (p₋.ν + p₊.ν)/2
	tνᵣ = (p₋.tν + p₊.tν)/2
	ζᵣ = (ζ₊ + ζ₋)/2
	rip = find_rip(u_pml, νᵣ, ζᵣ, tνᵣ)
	if !(ζ₋ <= rip.ζ <= ζ₊)
		@warn "ζ from find_rip was outside of range $ζ₋ $rip.ζ $ζ₊"
	end
    if isnan(rip.tν)
        @warn "tν from rip isnan: rip.tν"
    end
    if !(0 <= rip.ν <= 1)
        @warn "rip ν  is outside PML domain: rip.ν"
    end
	rip
end

function relative_l2_difference_knots(line1::InterpLine, line2::InterpLine)
	νs1 = map(p->p.ν, line1.points)
	νs2 = map(p->p.ν, line2.points)
	max_common_ν = min(maximum(νs1), maximum(νs2))
	knots = filter(ν->ν<=max_common_ν, sort(vcat(νs1, νs2)))
	line1_points = line1.(knots)
	line2_points = line2.(knots)
	rel_diff = (2*norm(line1_points - line2_points)
		/(norm(line1_points) + norm(line2_points)))
	return rel_diff
end

function relative_l2_difference(tν₋::InterpLine, tν₊::InterpLine)
    function sq_diff(s0::InterpSegment, s1::InterpSegment, ν::Number)
        tν0 = s0(ν).tν
        tν1 = s1(ν).tν
        return abs2(tν0 - tν1)
    end

    l2_sq_diff = sqrt(line_integrate_hcubature(tν₋, tν₊, sq_diff; atol=1e-12, rtol=1e-10, maxevals=10_000))

    l2_sq_tν₋ = sqrt(integrate(tν₋, p->abs2(p.tν)))
    l2_sq_tν₊ = sqrt(integrate(tν₊, p->abs2(p.tν)))

    return 2*l2_sq_diff / (l2_sq_tν₋ + l2_sq_tν₊)
end

"""
Find all rips of size >ε between ζs, by recursively subdividing until difference between ζs is <δ.

Size of rip is defined by the function tν_metric, which finds the distance between two interpolations.
If a rip persists and the difference between ζs is <δ we use Newtons method to find the rip accurately.
We then push this onto our vector of rips.
"""
function find_rips!(rips::Vector{Rip2D}, u_pml::PMLFieldFunction, ζs::AbstractVector;
	ε=1e-2, δ=1e-5, tν_metric=relative_l2_difference,
	tν₋ = interpolation(u_pml, first(ζs)), tν₊ = interpolation(u_pml, last(ζs)),)::Vector{Rip2D}

	tν₁ = tν₋
	for (ζ₁, ζ₂) in consecutive_pairs(ζs)
		tν₂ = ζ₂ == last(ζs) ? tν₊ : interpolation(u_pml, ζ₂)
        m = tν_metric(tν₁,tν₂)
        if isnan(m)
            error("Metric was nan")
        end
		if m > ε
			if abs(ζ₂ - ζ₁) < δ
				push!(rips, find_rip(u_pml, tν₁, tν₂))
			else
				find_rips!(rips, u_pml, range(ζ₁, ζ₂, length=3); ε, δ, tν₋=tν₁, tν₊=tν₂)
			end
		end
		tν₁ = tν₂
	end

    return rips
end

function find_rips(args...; kwargs...)::Vector{Rip2D}
	rips = Vector{Rip2D}(undef,0)
	find_rips!(rips, args...; kwargs...)
end
