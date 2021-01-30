module Aurora

using Reexport

using InvertedIndices
using LinearAlgebra
using Statistics
@reexport using StatsBase

include("replicated_sample.jl")
include("auroral.jl")

export ReplicatedSample,
       Auroral

end
