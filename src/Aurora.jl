module Aurora

using Reexport

using LaTeXStrings
using LinearAlgebra
using MultivariateStats
using NearestNeighbors
using RecipesBase

using Statistics
@reexport using StatsBase


abstract type AbstractAurora end
abstract type AbstractFittedAurora end
StatsBase.predict(fitted_aurora::AbstractFittedAurora) = fitted_aurora.μs

include("replicated_sample.jl")
include("auroral.jl")
include("auroraknn.jl")

export ReplicatedSample,
       Auroral,
       CoeyCunningham,
       AuroraKNN

end
