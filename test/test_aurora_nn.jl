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

knn_fit_kd = StatsBase.fit(AuroraKNN(kKNN=100, loocv=false, tree=:kdtree), Zs)
knn_fit_ball = StatsBase.fit(AuroraKNN(kKNN=100, loocv=false, tree=:balltree), Zs)
knn_fit_brute = StatsBase.fit(AuroraKNN(kKNN=100, loocv=false, tree=:brutetree), Zs)

@test knn_fit_kd.kskNN == fill(100, 10)
@test predict(knn_fit_kd) ≈ predict(knn_fit_ball)
@test predict(knn_fit_kd) ≈ predict(knn_fit_brute)
