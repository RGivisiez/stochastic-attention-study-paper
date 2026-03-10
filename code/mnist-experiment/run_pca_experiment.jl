#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# PCA Experiment: Does projecting MNIST into its PCA subspace improve SA
# sample quality by removing null-space dimensions?
#
# Hypothesis: With K=100 patterns in d=784, the memory matrix has rank ≤100.
# Noise injected into the 684 null-space dimensions is wasted (no gradient
# signal to shape it). PCA concentrates noise into signal-carrying directions,
# improving SNR and (hopefully) sample quality.
#
# Sweep: PCA rank r ∈ {20, 50, 80, 100} plus the d=784 baseline (no PCA).
# For each r, β* is found via entropy inflection, then SA runs at:
#   - β_gen  = 2β*  (generation regime)
#   - β_ret  = 5β*  (retrieval regime)
# All metrics are computed in 784-d pixel space for cross-rank comparability.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── Global parameters ────────────────────────────────────────────────────────
const number_of_examples  = 100
const number_of_rows      = 28
const number_of_cols      = 28
const number_of_pixels    = number_of_rows * number_of_cols
const α_step              = 0.01
const n_chains            = 30
const T_per_chain         = 5000
const T_burnin            = 2000
const thin_interval       = 100
const samples_per_chain   = 5
const σ_init              = 0.01
const DIGIT               = 3
const R_VALUES            = [20, 50, 80, 100]   # PCA ranks to test

# ── Helpers ──────────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; nr::Int=28, nc::Int=28)
    X = reshape(s, nr, nc) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W    = 28, 28
    canvas  = zeros(Float64, nrows*H + (nrows-1)*gap, ncols*W + (ncols-1)*gap)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx in 1:(nrows*ncols)
        r  = div(idx-1, ncols);  c  = rem(idx-1, ncols)
        y0 = r*(H+gap)+1;        x0 = c*(W+gap)+1
        img = decode(samples[indices[idx]])
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

function chain_metric_se(samps, metric_fn; nc=n_chains, spc=samples_per_chain)
    vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:nc]
    return std(vals) / sqrt(nc)
end

"""
    find_entropy_inflection(X̂; α, n_betas, β_range) -> NamedTuple

Compute the entropy inflection point β* for the memory matrix X̂.
"""
function find_entropy_inflection(X̂::Matrix{Float64};
                                  α::Float64=0.01,
                                  n_betas::Int=50,
                                  β_range::Tuple{Float64,Float64}=(0.1, 1000.0))

    d, K = size(X̂)
    βs = 10 .^ range(log10(β_range[1]), log10(β_range[2]), length=n_betas)

    n_probes = min(K, 20)
    Hs = zeros(n_betas)
    for (bi, β) in enumerate(βs)
        H_sum = 0.0
        for k in 1:n_probes
            H_sum += attention_entropy(X̂[:, k], X̂, β)
        end
        Hs[bi] = H_sum / n_probes
    end

    log_βs = log.(βs)
    dH = diff(Hs) ./ diff(log_βs)
    d2H = diff(dH) ./ diff(log_βs[1:end-1])

    inflection_idx = 1
    min_d2H = Inf
    for i in 1:length(d2H)
        if d2H[i] < min_d2H
            min_d2H = d2H[i]
            inflection_idx = i + 1
        end
    end

    β_star = βs[inflection_idx]
    snr_star = sqrt(α * β_star / (2 * d))
    β_star_theory = sqrt(d)
    snr_star_theory = sqrt(α / (2 * sqrt(d)))

    @info "  Entropy inflection (d=$d, K=$K): β*=$(round(β_star, digits=2)), SNR*=$(round(snr_star, digits=4)), theory β*=$(round(β_star_theory, digits=2))"

    return (β_star=β_star, snr_star=snr_star,
            β_star_theory=β_star_theory, snr_star_theory=snr_star_theory,
            βs=βs, Hs=Hs)
end

# ── Load MNIST ───────────────────────────────────────────────────────────────
@info "Loading MNIST …"
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# ── Build raw pixel matrix (un-normalized) ───────────────────────────────────
ϵ = 1e-12
X_raw = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    xᵢ = reshape(transpose(digits_dict[DIGIT][:, :, i]) |> Matrix, number_of_pixels) |> vec
    X_raw[:, i] = xᵢ
end
K = size(X_raw, 2)
@info "Raw pixel matrix: $(size(X_raw)) (K=$K)"

# ── Also build the full-space unit-normalized matrix (for pixel-space metrics) ─
X̂_full = copy(X_raw)
for i in 1:K
    X̂_full[:, i] ./= (norm(X̂_full[:, i]) + ϵ)
end

# ── Variance explained analysis ──────────────────────────────────────────────
@info "Computing PCA variance explained …"
pca_full = MultivariateStats.fit(PCA, X_raw; maxoutdim=K)
cum_var = cumsum(principalvars(pca_full)) ./ tvar(pca_full)
for r in R_VALUES
    pct = round(100 * cum_var[min(r, length(cum_var))], digits=1)
    @info "  r=$r: $(pct)% variance explained"
end

# ── Run SA at each PCA rank ──────────────────────────────────────────────────

"""
    run_sa_at_rank(r, X_raw, X̂_full) -> NamedTuple

Run the full SA pipeline at PCA rank r. Returns metrics in both PCA and pixel space.
"""
function run_sa_at_rank(r::Int, X_raw::Matrix{Float64}, X̂_full::Matrix{Float64})
    d_full, K = size(X_raw)

    # ── PCA projection ───────────────────────────────────────────────────
    pca_model = MultivariateStats.fit(PCA, X_raw; maxoutdim=r)
    d_pca = outdim(pca_model)
    Z = MultivariateStats.transform(pca_model, X_raw)  # d_pca × K
    var_explained = sum(principalvars(pca_model)) / tvar(pca_model)
    @info "  PCA r=$r: $d_full → $d_pca dims ($(round(100*var_explained, digits=1))% variance)"

    # ── Unit-normalize in PCA space ──────────────────────────────────────
    X̂ = copy(Z)
    for k in 1:K
        X̂[:, k] ./= (norm(X̂[:, k]) + ϵ)
    end

    # ── Entropy inflection → β* ──────────────────────────────────────────
    pt = find_entropy_inflection(X̂; α=α_step)
    β_gen = max(round(Int, 2 * pt.β_star), 1)
    β_ret = max(round(Int, 5 * pt.β_star), 1)
    @info "  β_gen=$β_gen (2β*), β_ret=$β_ret (5β*)"

    # ── SA multi-chain: generation regime ────────────────────────────────
    @info "  Running SA generation (β=$β_gen) …"
    gen_samples_pca = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d_pca)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=Float64(β_gen), α=α_step, seed=12345+c)
        chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs; push!(gen_samples_pca, chain_pool[idx]); end
    end

    # ── SA multi-chain: retrieval regime ─────────────────────────────────
    @info "  Running SA retrieval (β=$β_ret) …"
    ret_samples_pca = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d_pca)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=Float64(β_ret), α=α_step, seed=12345+c)
        chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs; push!(ret_samples_pca, chain_pool[idx]); end
    end

    # ── Project samples back to pixel space ──────────────────────────────
    gen_samples_pixel = [vec(MultivariateStats.reconstruct(pca_model, ξ)) for ξ in gen_samples_pca]
    ret_samples_pixel = [vec(MultivariateStats.reconstruct(pca_model, ξ)) for ξ in ret_samples_pca]

    # ── Metrics in PCA space ─────────────────────────────────────────────
    gen_nov_pca   = mean(sample_novelty(ξ, X̂) for ξ in gen_samples_pca)
    gen_div_pca   = sample_diversity(gen_samples_pca)
    gen_en_pca    = mean(hopfield_energy(ξ, X̂, Float64(β_gen)) for ξ in gen_samples_pca)
    ret_nov_pca   = mean(sample_novelty(ξ, X̂) for ξ in ret_samples_pca)
    ret_div_pca   = sample_diversity(ret_samples_pca)
    ret_en_pca    = mean(hopfield_energy(ξ, X̂, Float64(β_ret)) for ξ in ret_samples_pca)

    # ── Metrics in pixel space (cross-rank comparable) ───────────────────
    gen_maxcos_pixel = mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in gen_samples_pixel)
    gen_nov_pixel    = 1.0 - gen_maxcos_pixel
    gen_div_pixel    = sample_diversity(gen_samples_pixel)
    ret_maxcos_pixel = mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in ret_samples_pixel)
    ret_nov_pixel    = 1.0 - ret_maxcos_pixel
    ret_div_pixel    = sample_diversity(ret_samples_pixel)

    # ── Standard errors (pixel space, generation) ────────────────────────
    gen_nov_se = chain_metric_se(gen_samples_pixel,
        g -> mean(1.0 - nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    gen_div_se = chain_metric_se(gen_samples_pixel, g -> sample_diversity(g))
    gen_maxcos_se = chain_metric_se(gen_samples_pixel,
        g -> mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    ret_nov_se = chain_metric_se(ret_samples_pixel,
        g -> mean(1.0 - nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    ret_div_se = chain_metric_se(ret_samples_pixel, g -> sample_diversity(g))
    ret_maxcos_se = chain_metric_se(ret_samples_pixel,
        g -> mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in g))

    snr_gen = sqrt(α_step * β_gen / (2 * d_pca))
    snr_ret = sqrt(α_step * β_ret / (2 * d_pca))

    return (r=r, d_pca=d_pca, var_explained=var_explained,
            β_star=pt.β_star, β_gen=β_gen, β_ret=β_ret,
            snr_gen=snr_gen, snr_ret=snr_ret,
            # generation metrics (pixel space)
            gen_nov=gen_nov_pixel, gen_nov_se=gen_nov_se,
            gen_div=gen_div_pixel, gen_div_se=gen_div_se,
            gen_maxcos=gen_maxcos_pixel, gen_maxcos_se=gen_maxcos_se,
            gen_en_pca=gen_en_pca,
            # retrieval metrics (pixel space)
            ret_nov=ret_nov_pixel, ret_nov_se=ret_nov_se,
            ret_div=ret_div_pixel, ret_div_se=ret_div_se,
            ret_maxcos=ret_maxcos_pixel, ret_maxcos_se=ret_maxcos_se,
            ret_en_pca=ret_en_pca,
            # PCA-space metrics (for reference)
            gen_nov_pca=gen_nov_pca, gen_div_pca=gen_div_pca,
            ret_nov_pca=ret_nov_pca, ret_div_pca=ret_div_pca,
            # entropy curve
            entropy_curve=(pt.βs, pt.Hs),
            # samples for visualization
            gen_samples_pixel=gen_samples_pixel,
            ret_samples_pixel=ret_samples_pixel)
end

# ── Run the d=784 baseline (no PCA) ─────────────────────────────────────────
function run_baseline_no_pca(X_raw::Matrix{Float64}, X̂_full::Matrix{Float64})
    d, K = size(X_raw)

    # use the full-space unit-normalized matrix
    X̂ = X̂_full

    # entropy inflection in full space
    pt = find_entropy_inflection(X̂; α=α_step)
    β_gen = max(round(Int, 2 * pt.β_star), 1)
    β_ret = max(round(Int, 5 * pt.β_star), 1)
    @info "  Baseline d=$d: β_gen=$β_gen (2β*), β_ret=$β_ret (5β*)"

    # SA generation
    @info "  Running SA generation (β=$β_gen) …"
    gen_samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=Float64(β_gen), α=α_step, seed=12345+c)
        chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs; push!(gen_samples, chain_pool[idx]); end
    end

    # SA retrieval
    @info "  Running SA retrieval (β=$β_ret) …"
    ret_samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=Float64(β_ret), α=α_step, seed=12345+c)
        chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs; push!(ret_samples, chain_pool[idx]); end
    end

    # metrics (already in pixel space)
    gen_maxcos = mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in gen_samples)
    gen_nov    = 1.0 - gen_maxcos
    gen_div    = sample_diversity(gen_samples)
    gen_en     = mean(hopfield_energy(ξ, X̂, Float64(β_gen)) for ξ in gen_samples)
    ret_maxcos = mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in ret_samples)
    ret_nov    = 1.0 - ret_maxcos
    ret_div    = sample_diversity(ret_samples)
    ret_en     = mean(hopfield_energy(ξ, X̂, Float64(β_ret)) for ξ in ret_samples)

    gen_nov_se = chain_metric_se(gen_samples,
        g -> mean(1.0 - nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    gen_div_se = chain_metric_se(gen_samples, g -> sample_diversity(g))
    gen_maxcos_se = chain_metric_se(gen_samples,
        g -> mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    ret_nov_se = chain_metric_se(ret_samples,
        g -> mean(1.0 - nearest_cosine_similarity(ξ, X̂_full) for ξ in g))
    ret_div_se = chain_metric_se(ret_samples, g -> sample_diversity(g))
    ret_maxcos_se = chain_metric_se(ret_samples,
        g -> mean(nearest_cosine_similarity(ξ, X̂_full) for ξ in g))

    snr_gen = sqrt(α_step * β_gen / (2 * d))
    snr_ret = sqrt(α_step * β_ret / (2 * d))

    # entropy curve at full resolution for plotting
    pt_curve = (pt.βs, pt.Hs)

    return (r=d, d_pca=d, var_explained=1.0,
            β_star=pt.β_star, β_gen=β_gen, β_ret=β_ret,
            snr_gen=snr_gen, snr_ret=snr_ret,
            gen_nov=gen_nov, gen_nov_se=gen_nov_se,
            gen_div=gen_div, gen_div_se=gen_div_se,
            gen_maxcos=gen_maxcos, gen_maxcos_se=gen_maxcos_se,
            gen_en_pca=gen_en,
            ret_nov=ret_nov, ret_nov_se=ret_nov_se,
            ret_div=ret_div, ret_div_se=ret_div_se,
            ret_maxcos=ret_maxcos, ret_maxcos_se=ret_maxcos_se,
            ret_en_pca=ret_en,
            gen_nov_pca=gen_nov, gen_div_pca=gen_div,
            ret_nov_pca=ret_nov, ret_div_pca=ret_div,
            entropy_curve=pt_curve,
            gen_samples_pixel=gen_samples,
            ret_samples_pixel=ret_samples)
end

# ── Main sweep ───────────────────────────────────────────────────────────────
results = Dict{Int, Any}()

# PCA ranks
for r in R_VALUES
    @info "═══ Running PCA rank r=$r ═══"
    results[r] = run_sa_at_rank(r, X_raw, X̂_full)
end

# Baseline (no PCA)
@info "═══ Running baseline d=784 (no PCA) ═══"
results[784] = run_baseline_no_pca(X_raw, X̂_full)

# ── Print comparison table ───────────────────────────────────────────────────
println("\n" * "="^100)
println("PCA EXPERIMENT RESULTS — MNIST digit 3, K=$K")
println("="^100)

fmt(x, d) = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
fmtpm(v, se, dv, ds) = "$(fmt(v,dv)) ± $(fmt(se,ds))"

# Header
println("\n── Generation regime (β = 2β*) ──")
println(rpad("r", 6) * rpad("d_eff", 7) * rpad("Var%", 7) * rpad("β*", 8) *
        rpad("β_gen", 8) * rpad("SNR", 8) *
        rpad("MaxCos(px)", 16) * rpad("Novelty(px)", 16) *
        rpad("Diversity(px)", 16) * "Energy(PCA)")

all_ranks = sort(collect(keys(results)))
for r in all_ranks
    res = results[r]
    var_pct = round(100 * res.var_explained, digits=1)
    println(rpad(r, 6) * rpad(res.d_pca, 7) * rpad(var_pct, 7) *
            rpad(fmt(res.β_star, 1), 8) * rpad(res.β_gen, 8) *
            rpad(fmt(res.snr_gen, 4), 8) *
            rpad(fmtpm(res.gen_maxcos, res.gen_maxcos_se, 3, 3), 16) *
            rpad(fmtpm(res.gen_nov, res.gen_nov_se, 3, 3), 16) *
            rpad(fmtpm(res.gen_div, res.gen_div_se, 3, 3), 16) *
            string(fmt(res.gen_en_pca, 3)))
end

println("\n── Retrieval regime (β = 5β*) ──")
println(rpad("r", 6) * rpad("d_eff", 7) * rpad("β_ret", 8) * rpad("SNR", 8) *
        rpad("MaxCos(px)", 16) * rpad("Novelty(px)", 16) *
        rpad("Diversity(px)", 16) * "Energy(PCA)")

for r in all_ranks
    res = results[r]
    println(rpad(r, 6) * rpad(res.d_pca, 7) * rpad(res.β_ret, 8) *
            rpad(fmt(res.snr_ret, 4), 8) *
            rpad(fmtpm(res.ret_maxcos, res.ret_maxcos_se, 3, 3), 16) *
            rpad(fmtpm(res.ret_nov, res.ret_nov_se, 3, 3), 16) *
            rpad(fmtpm(res.ret_div, res.ret_div_se, 3, 3), 16) *
            string(fmt(res.ret_en_pca, 3)))
end

# ── LaTeX table rows ─────────────────────────────────────────────────────────
println("\n── LaTeX table rows (generation regime) ──")
fmtlatex(v, se, dv, ds) = "\$$(fmt(v,dv)) \\pm $(fmt(se,ds))\$"
for r in all_ranks
    res = results[r]
    label = r == 784 ? "SA (d=784, no PCA)" : "SA (PCA r=$r)"
    println("$label & $(fmtlatex(res.gen_maxcos, res.gen_maxcos_se, 3, 3)) & " *
            "$(fmtlatex(res.gen_nov, res.gen_nov_se, 3, 3)) & " *
            "$(fmtlatex(res.gen_div, res.gen_div_se, 3, 3)) \\\\")
end

# ── Save visualization grids ─────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

for r in all_ranks
    res = results[r]
    label = r == 784 ? "nopca" : "r$(r)"

    # generation grid
    fname_gen = "Fig_mnist_pca_grid_$(label)_gen.png"
    save(joinpath(figpath, fname_gen), Gray.(build_grid(res.gen_samples_pixel)))
    @info "Saved $fname_gen"

    # retrieval grid
    fname_ret = "Fig_mnist_pca_grid_$(label)_ret.png"
    save(joinpath(figpath, fname_ret), Gray.(build_grid(res.ret_samples_pixel)))
    @info "Saved $fname_ret"
end

# ── Entropy inflection curves (overlay plot) ─────────────────────────────────
p_entropy = plot(xlabel="β (inverse temperature)", ylabel="H(β) / log K",
                 title="Entropy inflection — MNIST digit 3 (K=$K)",
                 xscale=:log10, legend=:topright, size=(700, 450))

colors = [:blue, :green, :orange, :red, :gray]
for (i, r) in enumerate(all_ranks)
    res = results[r]
    βs, Hs = res.entropy_curve
    label = r == 784 ? "d=784 (no PCA)" : "PCA r=$r"
    plot!(p_entropy, βs, Hs ./ log(K), label=label, lw=2, color=colors[i])
    vline!([res.β_star], label="", ls=:dash, color=colors[i], alpha=0.5)
end
savefig(p_entropy, joinpath(figpath, "Fig_mnist_pca_entropy_curves.pdf"))
@info "Saved entropy curves figure"

# ── SNR vs pixel-space max-cos plot ──────────────────────────────────────────
snrs_gen = [results[r].snr_gen for r in all_ranks]
maxcos_gen = [results[r].gen_maxcos for r in all_ranks]
maxcos_se = [results[r].gen_maxcos_se for r in all_ranks]

p_snr = scatter(snrs_gen, maxcos_gen, yerror=maxcos_se,
    xlabel="SNR at generation β", ylabel="Max-cos to stored patterns (pixel space)",
    title="PCA rank effect on generation quality",
    label="", ms=8, color=:coral, legend=:topleft, size=(600, 400))
for (i, r) in enumerate(all_ranks)
    lbl = r == 784 ? "d=784" : "r=$r"
    annotate!(snrs_gen[i], maxcos_gen[i] + 0.01, text(lbl, 8))
end
savefig(p_snr, joinpath(figpath, "Fig_mnist_pca_snr_vs_maxcos.pdf"))
@info "Saved SNR vs max-cos figure"

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n" * "="^100)
println("SUMMARY")
println("="^100)
baseline = results[784]
for r in R_VALUES
    res = results[r]
    maxcos_ratio = res.gen_maxcos / baseline.gen_maxcos
    println("  r=$r: gen max-cos=$(fmt(res.gen_maxcos, 3)) ($(fmt(maxcos_ratio, 2))× baseline), " *
            "nov=$(fmt(res.gen_nov, 3)), div=$(fmt(res.gen_div, 3)), " *
            "β*=$(fmt(res.β_star, 1)), β_gen=$(res.β_gen)")
end
println("  d=784: gen max-cos=$(fmt(baseline.gen_maxcos, 3)) (baseline), " *
        "nov=$(fmt(baseline.gen_nov, 3)), div=$(fmt(baseline.gen_div, 3)), " *
        "β*=$(fmt(baseline.β_star, 1)), β_gen=$(baseline.β_gen)")

println("\nDone.")
