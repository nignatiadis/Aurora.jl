# Aurora: Averages of Units by Regressing on Ordered Replicates Adaptively.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nignatiadis.github.io/Aurora.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nignatiadis.github.io/Aurora.jl/dev)
[![Build Status](https://github.com/nignatiadis/Aurora.jl/workflows/CI/badge.svg)](https://github.com/nignatiadis/Aurora.jl/actions)
[![Coverage](https://codecov.io/gh/nignatiadis/Aurora.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nignatiadis/Aurora.jl)

Julia implementation of 

> Ignatiadis, N., Saha, S., Sun D. L., & Muralidharan, O. (2019).  **Empirical Bayes mean estimation with nonparametric errors via order statistic regression.** [[arXiv]](https://arxiv.org/abs/1911.05970)


Example code for Auroral (Aurora with linear regression)
```julia
julia> using Aurora

julia> μs = randn(10000); # generate true means

julia> zs = sqrt(10) .* randn(10000, 10) .+ μs; # Observe 10 noisy observations for each mean

julia> Zs = ReplicatedSample.(zs);

julia> auroral_fit = fit(Auroral(), Zs); # fit Auroral

julia> mean(abs2, μs .- predict(auroral_fit)) # MSE of Auroral
0.5289866834207907

julia> mean(abs2, μs .- mean.(Zs)) # Compare to MSE of row-wise mean
0.9830253207576279
```