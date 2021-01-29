using Aurora
using Documenter

makedocs(;
    modules=[Aurora],
    authors="Nikos Ignatiadis <nikos.ignatiadis01@gmail.com> and contributors",
    repo="https://github.com/nignatiadis/Aurora.jl/blob/{commit}{path}#L{line}",
    sitename="Aurora.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nignatiadis.github.io/Aurora.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nignatiadis/Aurora.jl",
)
