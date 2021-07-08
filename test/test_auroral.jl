n=500
K=21
μs = randn(n);
# generate true means
zs = sqrt(K) .* randn(n, K) .+ μs
# Observe 10 noisy observations for each mean
Zs = ReplicatedSample.(zs)

# Test Auroral

auroral_fit = StatsBase.fit(Auroral(), Zs)

@test typeof(auroral_fit.method) == Auroral
manual_pred_mat = zeros(500, 21)


for j in Base.OneTo(K)
    XjT = vcat(fill(1.0, 1, 500), Aurora.design_matrix(Zs, j))
    Yj = StatsBase.response.(Zs, j)
    βsj = transpose(XjT) \ Yj
    @test βsj ≈ auroral_fit.βs_list[j]
    manual_pred_mat[:,j] = transpose(XjT) * βsj
end

@test manual_pred_mat ≈ auroral_fit.μs_mat
rowmeans = mean(manual_pred_mat, dims=2)
@test rowmeans ≈ auroral_fit.μs
@test rowmeans ≈ StatsBase.predict(auroral_fit)

for j in Base.OneTo(K)
    @test mean(getindex.(auroral_fit.βs_list, j)) ≈ auroral_fit.βs[j]
end

# Test CC implementation

cc_fit = StatsBase.fit(CoeyCunningham(), Zs)
@test typeof(cc_fit.method) == CoeyCunningham

manual_pred_mat = zeros(500, 21)

for j in Base.OneTo(K)
    XjT = vcat(fill(1.0, 1, 500), mean(Aurora.design_matrix(Zs, j); dims=1))
    Yj = StatsBase.response.(Zs, j)
    βsj = transpose(XjT) \ Yj
    @test βsj ≈ cc_fit.βs_list[j]
    manual_pred_mat[:,j] = transpose(XjT) * βsj
end

@test manual_pred_mat ≈ cc_fit.μs_mat
rowmeans = mean(manual_pred_mat, dims=2)
@test rowmeans ≈ cc_fit.μs
@test rowmeans ≈ StatsBase.predict(cc_fit)

for j in Base.OneTo(2)
    @test mean(getindex.(cc_fit.βs_list, j)) ≈ cc_fit.βs[j]
end


 # Aurora and CC-L should agree when K=2


n=511
K=2
μs = randn(n);
# generate true means
zs = sqrt(K) .* randn(n, K) .+ μs
# Observe 10 noisy observations for each mean
Zs = ReplicatedSample.(zs)

auroral_fit = StatsBase.fit(Auroral(), Zs)
cc_fit = StatsBase.fit(CoeyCunningham(), Zs)

@test predict(cc_fit) ≈ predict(auroral_fit)
@test cc_fit.βs_list ≈ auroral_fit.βs_list
@test cc_fit.βs ≈ auroral_fit.βs
@test cc_fit.μs_mat ≈ auroral_fit.μs_mat
@test cc_fit.μs ≈ auroral_fit.μs
