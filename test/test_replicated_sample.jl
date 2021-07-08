using Aurora
using StatsBase
Z = ReplicatedSample(randn(10))
@test collect(Z.Z) ==  Z.sorted_Z[Z.rank_idx]

μs = randn(1000)  # generate true
zs = sqrt(10) .* randn(1000, 10) .+ μs



Zs1 = ReplicatedSample.(copy.(eachrow(zs)))
Zs2 = ReplicatedSample.(zs)

@test mean.(Zs1) == mean.(Zs2)


Aurora.design_matrix(Zs1, 1)

empty_mat = zeros(9, 1000)
empty_vec = zeros(9)
empty_vec2 = zeros(11)

for j in Base.OneTo(10)
    @test reduce(hcat, sort.(deleteat!.(copy.(getfield.(Zs1, :Z)), j))) == Aurora.design_matrix(Zs1, j)
    @test reduce(hcat, sort.(deleteat!.(copy.(eachrow(zs)), j))) == Aurora.design_matrix(Zs1, j)
    Aurora.design_matrix!(empty_mat, Zs1, j)
    @test Aurora.design_matrix(Zs1, j) == empty_mat
    i = sample(1:1000)
    Aurora.design_row!(empty_vec, Zs1[i], j)
    @test sort(deleteat!(copy(Zs1[i].Z), j)) == empty_vec

    Aurora.design_response_row!(empty_vec2, Zs1[i], j)
    @test [1; sort(deleteat!(copy(Zs1[i].Z), j)); Zs1[i].Z[j]] == empty_vec2
    @test Zs1[i].Z[j] == StatsBase.response(Zs1[i], j)
end
