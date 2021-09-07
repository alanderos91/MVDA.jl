using Code
using Documenter

DocMeta.setdocmeta!(Code, :DocTestSetup, :(using Code); recursive=true)

makedocs(;
    modules=[Code],
    authors="Alfonso Landeros <alanderos@ucla.edu> and contributors",
    repo="https://github.com/alanderos91/Code.jl/blob/{commit}{path}#{line}",
    sitename="Code.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
