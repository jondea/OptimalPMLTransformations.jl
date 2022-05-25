log10abs(z) = log10(abs(z))
logsigmoid(z) = z/(1+abs(z))

function complex_angle_heatmap!(f; xlims=(-50,50), ylims=(-50,50), n_samples=200)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    heatmap(x, y, (x,y)->angle(f(x+im*y)); color=:colorwheel,
        xlims, ylims)
end

function complex_angle_contour!(f; xlims=(-50,50), ylims=(-50,50), n_samples=200)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    contour!(x, y, (x,y)->abs(angle(f(x+im*y))), color=:colorwheel, clims=(-1π,1π))
end

function complex_angle_and_abs_contours!(f; xlims=(-50,50), ylims=(-50,50), n_samples=200)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    contour!(x, y, (x,y)->angle(f(x+im*y)), color=:white)
    contour!(x, y, (x,y)->log(abs(f(x+im*y))), color=:black)
end

function complex_abs_contour!(f; xlims=(-50,50), ylims=(-50,50), n_samples=200)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    contour!(x, y, (x,y)->log(abs(f(x+im*y))), color=:black)
end

function complex_heatmap!(f, complex_to_real_fnc;
        xlims=(-50,50), ylims=(-50,50), n_samples=100,
        kwargs...)
    x = range(xlims..., length=n_samples)
    y = range(ylims..., length=n_samples)
    heatmap!(x, y, (x,y)->complex_to_real_fnc(f(x+im*y)), kwargs...)
end

function complex_quiver!(f; xlims=(-50,50), ylims=(-50,50),
        arrow_scale=0.1, arrow_fnc=z->arrow_scale*(xlims[2]-xlims[1])*logsigmoid(z),
        n_arrows=20, kwargs...
    )

    xh = (xlims[2] - xlims[1])/n_arrows
    yh = (ylims[2] - ylims[1])/n_arrows

    x = (xlims[1]+xh/2):xh:(xlims[2]-xh/2)
    y = (ylims[1]+yh/2):yh:(ylims[2]-yh/2)

    grid = x .+ im.*reshape(y,1,:)

    complex_to_2tuple(z) = (real(z),imag(z))
    arrows = (complex_to_2tuple∘arrow_fnc∘f).(grid)
    quiver!(real.(grid), imag.(grid), quiver=arrows, arrow=arrow(:closed, :head, 0.2, 0.1), color=:black)
end
