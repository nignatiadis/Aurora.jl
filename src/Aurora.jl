module Aurora

using Reexport

using LinearAlgebra
using NearestNeighbors

using Statistics
@reexport using StatsBase


abstract type AbstractAurora end
abstract type AbstractFittedAurora end

include("replicated_sample.jl")
include("auroral.jl")
include("auroraknn.jl")

export ReplicatedSample,
       Auroral,
       CoeyCunningham,
       AuroraKNN

end
