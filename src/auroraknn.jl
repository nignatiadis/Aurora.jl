"""
    AuroraKNN(;kKNN=1000, loocv=true, tree=KDTree,
               pca=false, pca_dimension=false)

Aurora with `k`-Nearest neighbors.  If `looc=false`, then `k` is chosen equal to `kKNN`,
while if `loocv=true`, then `k` is selected for each held-out replicate by
Leave-One-Out Cross-validation among the choices 1,...`,kKNN`.

`tree` describes the nearest neighbor computation strategy. The following options are
available: `:auto`, as well as ,
`:kdtree`, `:balltree` and `:brutetree` from the `NearestNeighbors.jl` package.

If `pca=true`, a dimension reduction strategy is employed to find nearest neighbors using
PCA (principal component analysis). For each held-out replicate, the order statistics
are projected into the principal component subspace of dimension `pca_dimension`. Note that
in this case, the resulting nearest neighbors may only be interpreted as approximate
nearest neighbors.
"""
Base.@kwdef struct AuroraKNN <: AbstractAurora
    kKNN::Int = 1000
    loocv::Bool = true
    tree::Symbol = :auto
    pca::Bool = false
    pca_dimension::Int = 10
end

Base.@kwdef struct FittedAuroraKNN{Ms, Ss} <: AbstractFittedAurora
    μs_mat::Ms
    μs::Ss
    kskNN::Vector{Int}
    aurora
end

function StatsBase.fit(aurora::AuroraKNN, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1])
    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")

    n = length(Zs)
    kKNN  = min(aurora.kKNN, n-2)
    loocv = aurora.loocv
    tree_symbol = aurora.tree

    if tree_symbol == :auto
        tree_symbol = K <= 12 ? :kdtree : :brutetree
    end

    if tree_symbol == :kdtree
        tree = KDTree
    elseif tree_symbol == :balltree
        tree = BallTree
    elseif tree_symbol == :brutetree
        tree = BruteTree
    else
        throw("Nearest neighbor strategy chosen is not available.")
    end

    pca = aurora.pca && aurora.pca_dimension < K-1

    μs_mat = zeros(n,K)
    cache_mat = zeros(K-1, n)
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

        if !pca
            kdd = tree(cache_mat)
            _idxs,_dists = knn(kdd, cache_mat, kKNN, loocv);
        else
            pca_fit = fit(PCA, cache_mat; maxoutdim=aurora.pca_dimension, pratio=1)
            pca_projected_X = transform(pca_fit, cache_mat)
            kdd = tree(pca_projected_X)
            _idxs,_dists = knn(kdd, pca_projected_X, kKNN, loocv);
        end

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
