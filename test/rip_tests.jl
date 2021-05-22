import Test: @test, @testset
using OptimalPMLTransformations
using Tau

@testset "Find rips/classify outer boundary" begin

    R = 2.0
    δ = 1.0
    pml = AnnularPML(R, δ)
    θ₋ = 0
    θ₊ = τ

    @testset "Two mode pole" begin
        k = 1.0
        a = two_mode_pole_coef(0:1, 2.0+1.0im)
        u = HankelSeries(k, a)
        u_pml_coords(tν, tζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,tζ), pml)
        rips = classify_outer_boundary(u_pml_coords, θ₋, θ₊; Nζ=101, ε=1e-5, δ=1e-1, verbose=true)
        @test length(rips) == 2
    end

    @testset "Scattered k=0.1" begin
        k = 0.1
        a = scattered_coef(-10:10, k)
        u = HankelSeries(k, a)
        u_pml_coords(tν, tζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,tζ), pml)
        rips = classify_outer_boundary(u_pml_coords, θ₋, θ₊; Nζ=101, ε=1e-5, δ=1e-1, verbose=true)
        @test length(rips) == 4
        @test all([rips[1].ν, rips[4].ν] .≈ 0.99998500527112)
        @test all([rips[2].ν, rips[3].ν] .≈ 0.7028136109896637)
        @test all([rips[1].tν, rips[4].tν] .≈ -9.248066154125096 + 49.94321309993313im)
        @test all([rips[2].tν, rips[3].tν] .≈ -2.0743817352843426 + 3.175340472255964im)
        # 1,4 and 2,3 are reflections of each other in x axis
        @test all([rips[1].ζ, τ-rips[4].ζ] .≈ 1.1131640045166318)
        @test all([rips[2].ζ, τ-rips[3].ζ] .≈ 1.4203088430842012)
    end

    @testset "Scattered k=1.0" begin
        k = 1.0
        a = scattered_coef(-10:10, k)
        u = HankelSeries(k, a)
        u_pml_coords(tν, tζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,tζ), pml)
        rips = classify_outer_boundary(u_pml_coords, θ₋, θ₊; Nζ=101, ε=1e-5, δ=1e-1, verbose=true)
        @test length(rips) == 0
    end

end
