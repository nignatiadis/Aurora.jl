Z = ReplicatedSample(randn(10))
@test collect(Z.Z) ==  Z.sorted_Z[Z.rank_idx]

μs = randn(1000)  # generate true
zs = sqrt(10) .* randn(1000, 10) .+ μs



Zs1 = ReplicatedSample.(copy.(eachrow(zs)))
Zs2 = ReplicatedSample.(zs)

@test mean.(Zs1) == mean.(Zs2)
