
struct Rip2D
    ζ::Float64
    ν::Float64
    tν::Complex{Float64}
end

rpad_round(x,pad;kwargs...) = rpad(string(round(x;kwargs...)),pad)

"Debug function used when finding rips"
function rip_message(jump, ζ₋, ζ₊, descending)
    printstyled(
        "Jump of " * rpad_round(jump,9,sigdigits=3)
        * " between " * rpad_round(ζ₋,7,digits=3) * " and " * rpad_round(ζ₊,7,digits=3)
        * " " * (descending ? "descending" : "terminating"),
        color=(descending ? :light_red : :green)
    )
    println()
end

function classify_outer_boundary(u::AbstractFieldFunction, pml::PMLGeometry, ζ₋, ζ₊; kwargs...)
    u_pml_coords(tν, ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,complex(ζ)), pml)
    return classify_outer_boundary(u_pml_coords, ζ₋, ζ₊; kwargs...)
end

function classify_outer_boundary(field_fnc::Function, ζ₋, ζ₊; Nζ=101, ε=1e-5, δ=1e-1, verbose=true)

    rips = find_rips(field_fnc, ζ₋, ζ₊; Nζ=Nζ, ν=1.0-1e-9, ε=ε, δ=δ, verbose=false)

    println_rip(rip::Rip2D) = println("Rip2D at ζ = $(rip.ζ)  ν = $(rip.ν)  tν = $(rip.tν)")

    function check_bounded(ζ₋, ζ₊)
        ζ = (ζ₋ + ζ₊)/2
        mapping = optimal_pml_transformation_solve(tν->field_fnc(tν,ζ), 1.0, silent_failure=true)
        if mapping[4] == 1 && abs(mapping[2]) < 1e8
            print("Bounded")
        else
            print("Unbounded")
        end
        println(" between ζ = $ζ₋ and $ζ₊")
    end

    if length(rips) == 0
        println("No rips detected")
        check_bounded(ζ₋, ζ₊)
        return rips
    end

    check_bounded(ζ₋, rips[1].ζ)
    println_rip(rips[1])

    for i in 2:length(rips)
        check_bounded(rips[i-1].ζ, rips[i].ζ)
        println_rip(rips[i])
    end

    check_bounded(rips[end].ζ, ζ₊)

    return rips
end

function find_rips(u::AbstractFieldFunction, pml::PMLGeometry, ζ₋, ζ₊; kwargs...)::Vector{Rip2D}
    u_pml_coords(tν) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,ζ), pml)
    return find_rips(u_pml_coords, ζ₋, ζ₊; kwargs...)
end

function find_rips(field_fnc::Function, ζ₋, ζ₊; Nζ=101, ν=1.0-1e-9, ε=1e-5, δ=1e-1, verbose=true)::Vector{Rip2D}

    rips = Vector{Rip2D}(undef,0)

    ζs = range(ζ₋, stop=ζ₊, length=Nζ)

    if verbose println("Calculating $Nζ samples of tnu at nu = 1 - $(1-ν)") end
    if verbose
        tνs = @showprogress [optimal_pml_transformation_solve(tν->field_fnc(tν,ζ), ν)[1] for ζ in ζs]
    else
        tνs =               [optimal_pml_transformation_solve(tν->field_fnc(tν,ζ), ν)[1] for ζ in ζs]
    end

    scale = sum(abs,tνs)/Nζ

    if verbose
        println("Average abs value $(round(scale,sigdigits=3)), δ=$δ so looking for jumps > $(round(δ*scale,sigdigits=3)),")
        println("======================")
        println("")
    end

    for i in 1:(Nζ-1)
        if abs(tνs[i+1] - tνs[i]) > δ*scale
            if verbose rip_message(abs(tνs[i+1] - tνs[i]), ζs[i], ζs[i+1], true) end
            find_and_add_rips!(rips, field_fnc, ζs[i], ζs[i+1], tνs[i], tνs[i+1], scale; ν=ν, ε=ε, δ=δ, verbose=verbose)
        else
            if verbose rip_message(abs(tνs[i+1] - tνs[i]), ζs[i], ζs[i+1], false) end
        end
    end

    return rips

end

function find_and_add_rips!(rips::Vector{Rip2D}, field_fnc::Function, ζ₋, ζ₊, tν₋, tν₊, scale; ν=1.0-1e-9, ε=1.0e-5, δ=1e-1, verbose=true)

    ζ_mid = (ζ₋ + ζ₊)/2

    if abs(ζ₊ - ζ₋) < ε

        ν_vec₋ = [0.0]; tν_vec₋ = [0.0+0.0im]; ∂tν_∂ν_vec₋ = [0.0+0.0im]
        optimal_pml_transformation_solve(tν->field_fnc(tν,ζ₋), ν, ν_vec₋, tν_vec₋, ∂tν_∂ν_vec₋)
        rip_ind₋ = argmax(abs.(∂tν_∂ν_vec₋).*(1 .- ν_vec₋))
        ν_rip₋ = ν_vec₋[rip_ind₋]
        tν_rip₋ = tν_vec₋[rip_ind₋]

        ν_vec₊ = [0.0]; tν_vec₊ = [0.0+0.0im]; ∂tν_∂ν_vec₊ = [0.0+0.0im]
        optimal_pml_transformation_solve(tν->field_fnc(tν,ζ₊), ν, ν_vec₊, tν_vec₊, ∂tν_∂ν_vec₊)
        rip_ind₊ = argmax(abs.(∂tν_∂ν_vec₊).*(1 .- ν_vec₊))
        ν_rip₊ = ν_vec₊[rip_ind₊]
        tν_rip₊ = tν_vec₊[rip_ind₊]

        ν_rip = (ν_rip₋ + ν_rip₊)/2
        ν_rip_err = abs(ν_rip₋ - ν_rip₊)/2
        tν_rip = (tν_rip₋ + tν_rip₊)/2
        tν_rip_err = (tν_rip₋ - tν_rip₊)/2

        ε_newton = 1e-12
        ν_rip, ζ_rip, tν_rip = pole_newton_solve(field_fnc, ν_rip, ζ_mid, tν_rip; ε=ε_newton)

        if verbose && !( ζ₋< ζ_rip < ζ₊)
            @warn "The ζ we found using the newton method is not between what bisection found" ζ₋ ζ_rip ζ₊
        end

        if verbose
            printstyled("Terminating recursion, rip at ζ = $(ζ_rip) at ν = $(ν_rip), tν = $(tν_rip)"; bold=true, color=:red)
            println()
        end

        push!(rips, Rip2D(ζ_rip, ν_rip, tν_rip))
        return
    end

    tν_mid = optimal_pml_transformation_solve(tν->field_fnc(tν,ζ_mid), ν)[1]

    if abs(tν₋ - tν_mid) > δ*scale
        if verbose rip_message(abs(tν₋ - tν_mid), ζ₋, ζ_mid, true) end
        find_and_add_rips!(rips, field_fnc, ζ₋, ζ_mid, tν₋, tν_mid, scale; ν=ν, ε=ε, δ=δ, verbose=verbose)
    else
        if verbose rip_message(abs(tν₋ - tν_mid), ζ₋, ζ_mid, false) end
    end

    if abs(tν_mid - tν₊) > δ*scale
        if verbose rip_message(abs(tν_mid - tν₊), ζ_mid, ζ₊, true) end
        find_and_add_rips!(rips, field_fnc, ζ_mid, ζ₊, tν_mid, tν₊, scale; ν=ν, ε=ε, δ=δ, verbose=verbose)
    else
        if verbose rip_message(abs(tν_mid - tν₊), ζ_mid, ζ₊, false) end
    end

end

# function find_rips(::SingleAngularFourierMode, arg...; verbose=true)
#     if verbose
#         println("The optimal transformation for a single angular Fourier mode has no rips")
#     end
#     return Vector{Rip2D}[]
# end

# function find_rips(::PlanarWave, arg...; verbose=true)
#     if verbose
#         println("The optimal transformation for a planar wave has no rips")
#     end
#     return Vector{Rip2D}[]
# end
