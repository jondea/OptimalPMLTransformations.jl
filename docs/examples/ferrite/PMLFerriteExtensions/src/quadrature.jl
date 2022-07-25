
function anisotropic_quadrature(G::Type{Ferrite.RefCube}, order_x, order_y)
	T = Float64
	qx = QuadratureRule{1,G}(order_x)
	qy = QuadratureRule{1,G}(order_y)
	weights = flatten(getweights(qx) .* getweights(qy)')
	points = flatten([Tensors.Vec(x[1],y[1]) for x in getpoints(qx), y in getpoints(qy)])
	return QuadratureRule{2,G,T}(weights, points)
end
