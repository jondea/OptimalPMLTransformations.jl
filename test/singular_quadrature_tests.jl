
import Test: @test, @testset
using OptimalPMLTransformations

import Base: one
one(x,y) = one(x)

@testset "Singular integration" begin

    @testset "Unit function integration" begin

        for N in 2:7:23

            begin
                nodes, weights = gausslegendreunit(N)
                @test all(0 .<= nodes .<= 1)
                @test sum(weights) ≈ 1
            end

            @test int_gauss(one, N) ≈ 1

            @test int_gauss_2d(one, N) ≈ 1

            begin
                nodes, weights = gausslegendreunittrans(N, 0.5)
                @test all(map(n -> all(0 .<= n .<= 1), nodes))
                @test sum(weights) ≈ 1
            end

            @test int_gauss_trans(one, N; a=0.5) ≈ 1

            @test int_gauss_trans_2d(one, N; a=0.5) ≈ 1

            begin
                a = 0.5
                x_crit = 0.3
                nodes, weights = gausslegendreunittrans(N,a)
                tnodes, tweights = gausslegendretrans_mid(N,nodes,weights,x_crit)
                @test all(0 .<= nodes .<= 1)
                @test sum(weights) ≈ 1
            end

            @test int_gauss_trans_mid(one, N) ≈ 1

            begin
                a = 0.5
                x_crit = (0.3, 0.6)
                nodes, weights = gausslegendreunittrans(N,a)
                tnodes, tweights = gausslegendretrans_mid(N,nodes,weights,x_crit)
                @test all(map(n -> all(0 .<= n .<= 1), nodes))
                @test sum(weights) ≈ 1
            end

            @test int_gauss_trans_2d_mid(one, N) ≈ 1

        end

    end

    @testset "Integrable singularity integration" begin

        a = 0.5
        f_1d(x_crit) = x -> complex(x-x_crit)^-a * (1 + 10(x-x_crit) - 8(x-x_crit)^2)
        antiderivative_of_f_1d(x_crit) = x -> complex(x-x_crit)^-a * ((x-x_crit)/(1-a) + 10(x-x_crit)^2/(2-a) - 8(x-x_crit)^3/(3-a))

        N = 10

        x_crit = 0.0
        @test int_gauss_trans(f_1d(x_crit), N; a) ≈ antiderivative_of_f_1d(x_crit)(1)

        x_crit = 0.7
        @test int_gauss_trans_mid(f_1d(x_crit), N; a, x_crit) ≈ antiderivative_of_f_1d(x_crit)(1) - antiderivative_of_f_1d(x_crit)(0)

        f_2d(xy_crit) = (x,y) -> sqrt((x-xy_crit[1])^2+(y-xy_crit[2])^2)^-a * (1 + 10x - 8x^2) * (2 - 5x + 2x^2)
        f_rip(xy_crit) = (x,y) -> begin
            z = -(x-xy_crit[1]) + (y-xy_crit[2])*im
            -1/sqrt(z)
        end

        # Test for no-throw until we work out closed form
        xy_crit=(0.0,0.0)
        int_gauss_trans_2d(f_2d(xy_crit), N; a)
        int_gauss_trans_2d(f_rip(xy_crit), N; a)
        @test true

        # Test for no-throw until we work out closed form
        xy_crit=(0.1,0.9)
        int_gauss_trans_2d_mid(f_2d(xy_crit), N; a, x_crit=xy_crit)
        int_gauss_trans_2d_mid(f_rip(xy_crit), N; a, x_crit=xy_crit)
        @test true

    end

end
