import Test: @test, @testset
using OptimalPMLTransformations

@testset "Single Hankel basics" begin
    k = 3.2
    m = 4
    a = 3.4

    u1 = single_hankel_mode(k,m,a)
    u2 = single_angular_fourier_mode(k,m,a)

    R = 1.0
    δ = 1.0
    pml = AnnularPML(R,δ)

    x1 = CartesianCoordinates(1.5, 0.0)
    x2 = CartesianCoordinates(1.5*√2/2, 1.5*√2/2)
    x3 = CartesianCoordinates(0.0, 1.5)
    x4 = CartesianCoordinates(0.0, -1.5)

    @test u1(x1) ≈ u2(x1)
    @test u1(x2) ≈ u2(x2)
    @test u1(x3) ≈ u2(x3)
    @test u1(x4) ≈ u2(x4)

    p1 = PolarCoordinates(1.5, 0.0)
    p2 = PolarCoordinates(1.5, τ/8)
    p3 = PolarCoordinates(1.5, τ/4)
    p4 = PolarCoordinates(1.5, 3τ/4)

    @test u1(x1) ≈ u1(p1)
    @test u1(x2) ≈ u1(p2)
    @test u1(x3) ≈ u1(p3)
    @test u1(x4) ≈ u1(p4)

    ν1 = PMLCoordinates(0.5, 0.0)
    ν2 = PMLCoordinates(0.5, τ/8)
    ν3 = PMLCoordinates(0.5, τ/4)
    ν4 = PMLCoordinates(0.5, 3τ/4)

    @test u1(ν1, pml) ≈ u1(p1)
    @test u1(ν2, pml) ≈ u1(p2)
    @test u1(ν3, pml) ≈ u1(p3)
    @test u1(ν4, pml) ≈ u1(p4)

end
