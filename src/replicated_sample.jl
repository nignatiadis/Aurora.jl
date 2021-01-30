struct ReplicatedSample{TS<:AbstractVector,SS,RS} #<: EBayesSample{TS}
    Z::TS
    sorted_Z::SS
    rank_idx::RS
end

function ReplicatedSample(Z)
    sorted_Z = sort(Z)
    rank_idx = StatsBase.ordinalrank(Z)
    ReplicatedSample(Z, sorted_Z, rank_idx)
end

function Base.broadcasted(::typeof(ReplicatedSample), Zs::AbstractMatrix)
    ReplicatedSample.(copy.(eachrow(Zs)))
end


Statistics.mean(Z::ReplicatedSample) = mean(Z.Z)
Statistics.std(Z::ReplicatedSample) = std(Z.Z)

StatsBase.nobs(Z::ReplicatedSample) = length(Z.Z)



function design_response_row!(vec, Z::ReplicatedSample, j)
    vec[1] = 1.0
    K = nobs(Z)
    vec[2:K] = Z.sorted_Z[Not(Z.rank_idx[j])]
    vec[K+1] = Z.Z[j]
    vec
end
