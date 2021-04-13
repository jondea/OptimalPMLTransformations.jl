
using Plots

"Function to do backend specific save procedure and return plot (or not)"
function backend_save_and_return(b::Plots.PGFPlotsBackend, filename)

    savefig("$filename.tex")
    run(`sed -i 's|{152.4mm}|{0.8\\textwidth}|g' $filename.tex`)
    run(`sed -i 's|{101.6mm}|{0.6\\textwidth}|g' $filename.tex`)

    savefig("$filename.pdf")

    # Return nothing because current pgfplots errors in repl
    return nothing
end

function backend_save_and_return(b::Plots.PyPlotBackend, filename)
    savefig("$filename.png")
    return plot!()
end
