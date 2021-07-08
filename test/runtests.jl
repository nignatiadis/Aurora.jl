using Aurora
using Test

@testset "Replicated Samples" begin
    include("test_replicated_sample.jl")
end

@testset "Auroral" begin
    include("test_auroral.jl")
end

@testset "Aurora-KNN" begin
    include("test_aurora_nn.jl")
end
