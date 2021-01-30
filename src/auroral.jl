struct Auroral end

struct FittedAuroral{Ts, Ms, Ss}
    βs::Ts
    μs_mat::Ms
    μs::Ss
end

function StatsBase.fit(::Auroral, Zs::AbstractVector{<:ReplicatedSample})
    K = nobs(Zs[1]) # todo check homoskedastic
    n = length(Zs)
    any(nobs.(Zs) != K) || throw("All Zs should have the same number of replicates.")
    μs_mat = zeros(n,K)

    chr = cholesky(zeros(K+1, K+1) + I)
    cache_vec = zeros(K+1)
    βs = Vector{typeof(cache_vec)}(undef, K)

    for j in Base.OneTo(K)
        fill!(chr.factors, 0)
        fill!(cache_vec, 0)
        for Z in Zs
            design_response_row!(cache_vec, Z, j)
            lowrankupdate!(chr, cache_vec)
        end
        RXX = UpperTriangular(view(chr.U, 1:K, 1:K))
        βs[j] = ldiv!(RXX, copy(chr.U[1:K, end]))

        for (i,Z) in enumerate(Zs)
            design_response_row!(cache_vec, Z, j)
            μs_mat[i,j] = dot(βs[j], view(cache_vec,1:K))
        end
    end
    μs = mean(μs_mat; dims=2)
    FittedAuroral(βs, μs, μs_mat)
end

StatsBase.predict(fitted_auroral::FittedAuroral) = fitted_auroral.μs
