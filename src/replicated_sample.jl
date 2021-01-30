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

function StatsBase.response(Z::ReplicatedSample, j::Integer)
    Z.Z[j]
end

function design_row!(vec, Z::ReplicatedSample, j)
    K = nobs(Z)
    rank_j = Z.rank_idx[j]
    if rank_j > 1
        vec[1:(rank_j-1)] = Z.sorted_Z[1:(rank_j-1)]
    end
    if rank_j < K
        vec[(rank_j):(K-1)] = Z.sorted_Z[(rank_j + 1):K]
    end
    vec
end

function design_response_row!(vec, Z::ReplicatedSample, j)
    vec[1] = 1.0
    K = nobs(Z)
    design_row!(view(vec, 2:K), Z, j)
    vec[K+1] = StatsBase.response(Z, j)
    vec
end

function design_matrix!(matrix, Zs::AbstractVector{<:ReplicatedSample}, j)
    for (i,Z) in enumerate(Zs)
        design_row!(view(matrix, :, i), Z, j)
    end
    matrix
end

function design_matrix(Zs::AbstractVector{<:ReplicatedSample}, j)
    n = length(Zs)
    K = nobs(Zs[1])
    design_matrix!(zeros(K-1,n), Zs, j)
end
