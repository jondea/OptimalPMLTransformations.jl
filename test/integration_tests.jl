using OptimalPMLTransformations
import OptimalPMLTransformations: integrate_between, InterpLine, InterpPoint

import Test: @test, @testset

@testset "Integration" begin

    @testset "Constant function" begin
        value = 4.3

        line1 = InterpLine(1.0, [InterpPoint(0.0, value, 0.0, 0.0), InterpPoint(0.2, value, 0.0, 0.0), InterpPoint(1.0, value, 0.0, 0.0)])
        line2 = InterpLine(2.0, [InterpPoint(0.0, value, 0.0, 0.0), InterpPoint(0.5, value, 0.0, 0.0), InterpPoint(1.0, value, 0.0, 0.0)])

        integrand(p::InterpPoint) = p.tν

        integral = integrate_between(line1, line2, integrand; order=2)

        @test integral ≈ value
    end

    @testset "Linear function through PML" begin
        line1 = InterpLine(0.1, [InterpPoint(0.0, 1.0, 1.0, 0.0), InterpPoint(0.2, 1.2, 1.0, 0.0), InterpPoint(1.0, 2.0, 1.0, 0.0)])
        line2 = InterpLine(0.2, [InterpPoint(0.0, 1.0, 1.0, 0.0), InterpPoint(0.5, 1.5, 1.0, 0.0), InterpPoint(1.0, 2.0, 1.0, 0.0)])

        integrand(p::InterpPoint) = p.tν

        integral = integrate_between(line1, line2, integrand; order=2)

        @test integral ≈ 0.15
    end

    @testset "Linear function across PML" begin
        line1 = InterpLine(0.0, [InterpPoint(0.0, 0.0, 0.0, 0.5), InterpPoint(0.7, 0.0, 0.0, 0.5), InterpPoint(1.0, 0.0, 0.0, 0.5)])
        line2 = InterpLine(2.0, [InterpPoint(0.0, 1.0, 0.0, 0.5), InterpPoint(0.3, 1.0, 0.0, 0.5), InterpPoint(1.0, 1.0, 0.0, 0.5)])

        integrand(p::InterpPoint) = p.tν

        integral = integrate_between(line1, line2, integrand; order=2)

        @test integral ≈ 1
    end

    @testset "Fundamental theorem of calculus" begin
        # Integrate the derivative of tν and confirm it equals tν

        integrand(tν) = abs(tν)

        R = 2.0
        δ = 1.0
        pml = AnnularPML(R, δ)

        k = 1.0
        a = two_mode_pole_coef(0:1, 2.0+1.0im)
        u = HankelSeries(k, a)

        ν_max = 0.99

        function create_line(ζ)
            ν_vec = Float64[]
            tν_vec = ComplexF64[]
            ∂tν_∂ν_vec = ComplexF64[]
            ∂tν_∂ζ_vec = ComplexF64[]
            optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec)
            return InterpLine(ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec)
        end

        tν_interp1 = create_line(0.0)
        tν_interp2 = InterpLine(1.0, tν_interp1.points)

        integrand(p::InterpPoint) = p.∂tν_∂ν

        integral = integrate_between(tν_interp1, tν_interp2, integrand; order=2)

        @test integral ≈ last(tν_interp1.points).tν
    end

end
