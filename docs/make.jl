using Documenter

include("makeplots.jl")

makedocs(
    modules=[OptimalPMLTransformations],
    format=Documenter.HTML(),
    sitename="OptimalPMLTransformations.jl",
    pages=[
        "Home" => "index.md",
    ]
)

if get(ENV, "TRAVIS", "") == ""
    makeplots()
end

# Only build plots in travis if we are deploying
# And dont install the dependencies unless we are deploying
function myDeps()
    if get(ENV, "TRAVIS", "") != ""
        println("Installing deploy dependencies")
        makeplots()
    end
end

deploydocs(
    repo = "github.com/jondea/OptimalPMLTranformations.jl.git",
    deps = myDeps
)
