using LinearAlgebra
using NearestNeighbors
n=200
K=10
μs = randn(n);
# generate true means
zs = sqrt(K) .* randn(n, K) .+ μs
# Observe 10 noisy observations for each mean
Zs = ReplicatedSample.(zs)

knn_cv_fit = StatsBase.fit(AuroraKNN(kKNN=100, loocv=true), Zs)

for j in Base.OneTo(K)
    knn_tmp_fit = StatsBase.fit(AuroraKNN(kKNN=knn_cv_fit.kskNN[j], loocv=false), Zs)
    @test knn_tmp_fit.μs_mat[:,j] ≈ knn_cv_fit.μs_mat[:,j]
end

# test Tree options

knn_fit_auto_loocv = StatsBase.fit(AuroraKNN(kKNN=22, loocv=true), Zs)
knn_fit_kd_loocv = StatsBase.fit(AuroraKNN(kKNN=22, loocv=true, tree=:kdtree), Zs)
knn_fit_ball_loocv = StatsBase.fit(AuroraKNN(kKNN=22, loocv=true, tree=:balltree), Zs)
knn_fit_brute_loocv = StatsBase.fit(AuroraKNN(kKNN=22, loocv=true, tree=:brutetree), Zs)

@test predict(knn_fit_auto_loocv) == predict(knn_fit_kd_loocv)
@test predict(knn_fit_kd_loocv) == predict(knn_fit_ball_loocv)
@test predict(knn_fit_kd_loocv) == predict(knn_fit_brute_loocv)



knn_fit_kd = StatsBase.fit(AuroraKNN(kKNN=20, loocv=false, tree=:kdtree), Zs)
knn_fit_ball = StatsBase.fit(AuroraKNN(kKNN=20, loocv=false, tree=:balltree), Zs)
knn_fit_brute = StatsBase.fit(AuroraKNN(kKNN=20, loocv=false, tree=:brutetree), Zs)

@test knn_fit_kd.kskNN == fill(20, 10)
@test predict(knn_fit_kd) ≈ predict(knn_fit_ball)
@test predict(knn_fit_kd) ≈ predict(knn_fit_brute)



for _ in Base.OneTo(10)
    i = sample(1:n)
    μs_tmp = zeros(K)
    for j in Base.OneTo(K)
        Zs_mat = Aurora.design_matrix(Zs, j)
        Y_vec = StatsBase.response.(Zs, j)
        dists_to_i = [norm(Zs_mat[:,i] .- zrow) for zrow in eachcol(Zs_mat)]
        sorted_idxs = sortperm(dists_to_i)
        @test sorted_idxs[1] == i
        μs_tmp[j] = mean(Y_vec[sorted_idxs[1:20]])
        @test  μs_tmp[j] ≈ knn_fit_kd.μs_mat[i, j]
    end
    @test mean(μs_tmp) ≈ knn_fit_kd.μs[i]
end



# test whether k=1 agrees with just sample mean

n=501
K=13
μs = sample(-1:1, n, replace=true)
# generate true means
zs =   sqrt(K) .* randn(n, K) .+ μs
# Observe 10 noisy observations for each mean
Zs = ReplicatedSample.(zs)

knn_cv_fit = StatsBase.fit(AuroraKNN(kKNN=1, loocv=true), Zs)
@test predict(knn_cv_fit) ≈ mean.(Zs)

knn_fit = StatsBase.fit(AuroraKNN(kKNN=1, loocv=false), Zs)
@test predict(knn_fit) ≈ mean.(Zs)


# Check the LOO NN code


kmax = 20
knn_cv_fit = StatsBase.fit(AuroraKNN(kKNN=kmax, loocv=true), Zs)
ks_opt = zeros(Int64, K)
mus_mat_opt = zeros(n, K)

for j in Base.OneTo(K)
    Zs_mat = Aurora.design_matrix(Zs, j)
    Y_vec = StatsBase.response.(Zs, j)

    kdd = KDTree(Zs_mat)

    loos = zeros(kmax)
    loos[1] = mean(abs2, Y_vec)

    for k in 1:(kmax-1)
        _idxs,_dists = knn(kdd, Zs_mat, k+1, true);
        mus_guess = [mean(Y_vec[_idx[2:end]]) for _idx in _idxs]
        loos[k+1] = mean(abs2, mus_guess .- Y_vec)
    end

    ks_opt[j] = argmin(loos)

    _idxs,_dists = knn(kdd, Zs_mat, ks_opt[j], true);
    mus_mat_opt[:,j] = [mean(Y_vec[_idx]) for _idx in _idxs]
end


@test ks_opt == knn_cv_fit.kskNN
@test mus_mat_opt ≈ knn_cv_fit.μs_mat
