using Aurora
using Test

@testset "Aurora.jl" begin
    include("test_replicated_sample.jl")
    include("test_aurora_nn.jl")
end
