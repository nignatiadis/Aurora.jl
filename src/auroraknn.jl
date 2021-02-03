Base.@kwdef struct AuroraKNN <: AbstractAurora
    kKNN::Int = 1000
    loocv::Bool = true
end

Base.@kwdef struct FittedAuroraKNN{Ms, Ss} <: AbstractFittedAurora
    μs_mat::Ms
    μs::Ss
    kskNN::Vector{Int}
    aurora
end

function StatsBase.fit(aurora::AuroraKNN, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1]) # todo check homoskedastic
    n = length(Zs)
    kKNN  = aurora.kKNN
    loocv = aurora.loocv

    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")

    μs_mat = zeros(n,K)
    cache_mat = zeros(K-1,n)
    cache_Y = zeros(n)

    if loocv
        cache_knn = zeros(kKNN, n)
        cache_lnn_loo = zeros(kKNN)
        kskNN = zeros(Int, K)
    else
        kskNN = fill(kKNN, K)
    end

    for j in Base.OneTo(K)
        design_matrix!(cache_mat, Zs, j)
        cache_Y[:] = StatsBase.response.(Zs, j)

        kdd = KDTree(cache_mat)
        _idxs,_dists = knn(kdd, cache_mat, kKNN, loocv);

        #return (kdd = kdd, idxs= _idxs, dists=_dists, cache_Y = cache_Y)
        if loocv
            for i in Base.OneTo(n)
                cumsum!(view(cache_knn, :, i), view(cache_Y, _idxs[i]))
            end
            cache_knn .-= cache_knn[1,:]'
            cache_knn ./= [1; 1:(kKNN-1)]
            cache_knn .-= cache_Y'
            cache_lnn_loo .= vec(mean(abs2, cache_knn; dims=2))
            k_opt =  argmin(cache_lnn_loo)
            kskNN[j] = k_opt
            μs_mat[:,j] .= ((cache_knn[k_opt, :] .+ cache_Y).*(k_opt-1) .+ cache_Y)./ k_opt
        else
            for i in Base.OneTo(n)
                μs_mat[i,j] = mean(view(cache_Y, _idxs[i]))
            end
        end
    end
    μs = mean(μs_mat; dims=2)
    FittedAuroraKNN(μs = μs, μs_mat = μs_mat, aurora=aurora, kskNN = kskNN)
end
