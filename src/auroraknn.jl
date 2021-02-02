Base.@kwdef struct AuroraKNN{S}
    kKNN::S = 100
end

struct FittedAuroraKNN{Ms, Ss}
    μs_mat::Ms
    μs::Ss
    aurora
end

function StatsBase.fit(aurora::AuroraKNN, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1]) # todo check homoskedastic
    n = length(Zs)
    kKNN  = aurora.kKNN

    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")

    μs_mat = zeros(n,K)
    cache_mat = zeros(K-1,n)
    cache_Y = zeros(n)


    for j in Base.OneTo(K)
        design_matrix!(cache_mat, Zs, j)
        cache_Y[:] = StatsBase.response.(Zs, j)

        kdd = KDTree(cache_mat)
        _idxs,_dists = knn(kdd, cache_mat, kKNN, false);

        for i in Base.OneTo(n)
            μs_mat[i,j] = mean(cache_Y[_idxs[i]])
        end
    end
    μs = mean(μs_mat; dims=2)
    FittedAuroraKNN(μs, μs_mat, aurora)
end

StatsBase.predict(fitted_aurora::FittedAuroraKNN) = fitted_aurora.μs
