import Test: @test, @testset
using StaticArrays
using OptimalPMLTransformations
using Tau

@testset "Cartesian constructors and operators" begin

    x1 = CartesianCoordinates(3.4, 5.6)
    x2 = CartesianCoordinates((3.4, 5.6))
    x3 = CartesianCoordinates(SVector(3.4, 5.6))
    @test x1 == x2
    @test x2 == x3
    @test x1 ≈ x2
    @test x2 ≈ x3

    @test (x2 - x3) ≈ SVector(0.0, 0.0)

    xc1 = CartesianCoordinates(3.4+0.0im, 5.6+2.3im)

    xf1 = CartesianCoordinates(3.4f0, 5.6f0)
end

@testset "Polar" begin
    p1 = CartesianCoordinates(3.4, 5.6)
    p2 = CartesianCoordinates(3.4, 5.6)
    @test p1 == p2
    @test p1 ≈ p2
end

@testset "Cartesian and Polar" begin

    x1 = CartesianCoordinates(1.5, 0.0)
    x2 = CartesianCoordinates(1.5*√2/2, 1.5*√2/2)
    x3 = CartesianCoordinates(0.0, 1.5)
    x4 = CartesianCoordinates(0.0, -1.5)

    p1  = PolarCoordinates(1.5, 0.0)
    p2  = PolarCoordinates(1.5, τ/8)
    p3  = PolarCoordinates(1.5, τ/4)
    p4a = PolarCoordinates(1.5, 3τ/4)
    p4b = PolarCoordinates(1.5, -τ/4)
    p4c = PolarCoordinates(1.5, 7τ/4)

    @test x1 ≈ p1
    @test x2 ≈ p2
    @test x3 ≈ p3
    @test x4 ≈ p4a
    @test x4 ≈ p4b
    @test x4 ≈ p4c

end

@testset "Annular PML" begin

    x1 = CartesianCoordinates(1.5, 0.0)
    x2 = CartesianCoordinates(1.5*√2/2, 1.5*√2/2)
    x3 = CartesianCoordinates(0.0, 1.5)
    x4 = CartesianCoordinates(0.0, -1.5)

    p1 = PolarCoordinates(1.5, 0.0)
    p2 = PolarCoordinates(1.5, τ/8)
    p3 = PolarCoordinates(1.5, τ/4)
    p4 = PolarCoordinates(1.5, 3τ/4)

    ν1 = PMLCoordinates(0.5, 0.0)
    ν2 = PMLCoordinates(0.5, τ/8)
    ν3 = PMLCoordinates(0.5, τ/4)
    ν4 = PMLCoordinates(0.5, 3τ/4)

    R = 1.0
    δ = 1.0
    pml = AnnularPML(R,δ)

    @test convert(CartesianCoordinates, ν1, pml) ≈ x1
    @test convert(PolarCoordinates, ν1, pml) ≈ p1

    @test convert(CartesianCoordinates, ν2, pml) ≈ x2
    @test convert(PolarCoordinates, ν2, pml) ≈ p2

    @test convert(CartesianCoordinates, ν3, pml) ≈ x3
    @test convert(PolarCoordinates, ν3, pml) ≈ p3

    @test convert(CartesianCoordinates, ν4, pml) ≈ x4
    @test convert(PolarCoordinates, ν4, pml) ≈ p4

end
