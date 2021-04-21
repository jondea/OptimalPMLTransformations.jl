

f1(ν_bar) = -im*log(1-ν_bar)
f2(ν_bar) = ν_bar
f3(ν_bar) = 1/ν_bar
f4(ν_bar) = ν_bar^2
f5(ν_bar) = ν_bar^3

trans_basis = [f1, f2, f3, f4, f5]

function nearly_optimal_trans(u::AbstractFieldFunction, )

    # Initial guess gives us SFB
    a0 = [1.0+0.0im, 0, 0, 0, 0]

    trans(a_vec, ν_bar) = sum([a*f(ν_bar) for (a,f) in zip(a_vec, trans_basis)])

    function cost(a)

        integral = 0
        for θ in θs
            # Use gauss knot points? Tensor product in θ and ν?
            # Cache transformation?
            for ν_bar in ν_bar_vec
                _u = u(trans(a, ν_bar), θ)
                first_deriv_in_ν = du_dr*dr_dν
                second_deriv_in_ν = d2u_dr2*(dr_dν)^2 + du_dr*d2r_dν2
                integral += sum(magnitude(second_deriv_in_ν)) # Squared magnitude?
            end
        end
    end

    # Apply newton method to minimise cost
    # How to differentiate the cost function? Autodiff?

    tν =
    x = X + tν/k
end
