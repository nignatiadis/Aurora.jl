struct Auroral <: AbstractAurora end

Base.@kwdef struct FittedAuroral{Ts, Ms, Ss} <: AbstractFittedAurora
    βs::Ts
    βs_list::Vector{Ts}
    μs_mat::Ms
    μs::Ss
    method
end

function StatsBase.fit(auroral::Auroral, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1])
    n = length(Zs)
    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")
    μs_mat = zeros(n,K)

    # See https://github.com/dmbates/CopenhagenEcon/blob/master/jmd/03-LinearAlgebra.jmd
    cache_vec = zeros(K+1)
    cache_mat = zeros(K+1,n)
    cache_cov_mat = zeros(K+1, K+1)

    βs_list = Vector{typeof(cache_vec)}(undef, K)

    for j in Base.OneTo(K)
        fill!(cache_mat, 0)
        fill!(cache_cov_mat, 0)
        fill!(cache_vec, 0)

        for (i,Z) in enumerate(Zs)
            design_response_row!(view(cache_mat, :, i), Z, j)
        end

        mul!(cache_cov_mat, cache_mat, cache_mat')
        chr = cholesky!(cache_cov_mat)

        RXX = UpperTriangular(view(chr.U, 1:K, 1:K))
        βs_list[j] = ldiv!(RXX, copy(chr.U[1:K, end]))

        for (i,Z) in enumerate(Zs)
            design_response_row!(cache_vec, Z, j)
            μs_mat[i,j] = dot(βs_list[j], view(cache_vec,1:K))
        end
    end
    μs = vec(mean(μs_mat; dims=2))
    βs = vec(mean(hcat(βs_list...), dims=2))
    FittedAuroral(βs = βs, βs_list = βs_list,
                μs = μs, μs_mat = μs_mat,
                method = auroral)
end

struct CoeyCunningham <: AbstractAurora end

function StatsBase.fit(auroral::CoeyCunningham, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1])
    n = length(Zs)
    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")
    μs_mat = zeros(n,K)

    cache_means = zeros(n)
    cache_responses = zeros(n)
    cache_vec = zeros(K-1)

    βs_list = Vector{typeof(cache_vec)}(undef, K)

    for j in Base.OneTo(K)
        cache_responses .= StatsBase.response.(Zs, j)
        fill!(cache_means, 0)

        for (i,Z) in enumerate(Zs)
            cache_means[i] = mean(design_row!(cache_vec, Z, j))
        end

        β_slope = cov(cache_means, cache_responses)/var(cache_means)
        β_intercept = mean(cache_responses) - β_slope*mean(cache_means)

        βs_list[j] = [β_intercept; β_slope]

        μs_mat[:,j] = β_intercept .+ β_slope .* cache_means
    end
    μs = vec(mean(μs_mat; dims=2))
    βs = vec(mean(hcat(βs_list...), dims=2))
    FittedAuroral(βs = βs, βs_list = βs_list,
                μs = μs, μs_mat = μs_mat,
                method = auroral)
end
