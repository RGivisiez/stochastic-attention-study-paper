#!/usr/bin/env julia
# run_finance_beta_sweep.jl
#
# Fine β sweep for the finance case study (MALA warm-start sequential generation).
#
# Three diagnostic groups per β value:
#   (1) Marginal fidelity     — mean KS D-statistic (gen vs hist) over all 424 tickers
#   (2) Chain mobility        — mean/std acceptance rate and its lag-1 ACF
#                               (the key lever for regime-switching dynamics)
#   (3) Volatility clustering — mean ACF(|g|, lags 1–5) and ACF(g², lags 1–10)
#                               over all tickers (ARCH signature)
#
# Usage:
#   cd code/financial-experiment
#   julia --project=. run_finance_beta_sweep.jl

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "Include.jl"))

# ── 1. Data loading (exact notebook pipeline) ────────────────────────────────────
println("Loading market data …")
original_dataset = MyTrainingMarketDataSet() |> x -> x["dataset"];
maximum_number_trading_days = original_dataset["AAPL"] |> nrow;

dataset = Dict{String,DataFrame}();
for (ticker, data) in original_dataset
    if nrow(data) == maximum_number_trading_days
        dataset[ticker] = data
    end
end

list_of_tickers = keys(dataset) |> collect |> sort;

G = let
    r̄  = 0.0
    Δt = 1/252
    log_growth_matrix(dataset, list_of_tickers, Δt=Δt, risk_free_rate=r̄)
end;

memories, scaled_memories = let
    M   = transpose(G) |> Matrix   # d × K  (firms × days)
    d, K = size(M)
    ϵ   = 1e-12
    S   = zeros(d, K)
    for i in 1:K
        col   = M[:, i]
        col_c = col .- mean(col)
        S[:, i] = col_c ./ (norm(col_c) + ϵ)
    end
    (M, S)
end;

d = size(scaled_memories, 1)   # 424 firms
K = size(scaled_memories, 2)   # 2766 trading days
α = 0.01                       # MALA step size
println("d = $d, K = $K")

# ── 2. KS D-statistic (two-sample, no external dependency) ──────────────────────
function ks_D(x::Vector{Float64}, y::Vector{Float64})
    combined  = sort(vcat(x, y))
    n, m      = length(x), length(y)
    ecdf_x    = [count(≤(v), x) / n for v in combined]
    ecdf_y    = [count(≤(v), y) / m for v in combined]
    maximum(abs.(ecdf_x .- ecdf_y))
end

# ── 3. Sweep parameters ──────────────────────────────────────────────────────────
β_values = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 75.0, 100.0]
T_days   = 1500   # ~6 years of daily returns
T_inner  = 200    # MALA inner steps per day

X = scaled_memories
M = memories

println("\nβ sweep: $(β_values)")
println("T_days=$T_days, T_inner=$T_inner\n")

# result storage
header = ("β", "SNR", "accept̄", "std(accept)", "ACF(accept,1)",
          "KS D̄", "ACF(|g|,1-5)", "ACF(g²,1-10)")
rows = NamedTuple[]

for β_val in β_values
    rng = Random.MersenneTwister(42)
    idx₀ = rand(rng, 1:K)
    ξ = X[:, idx₀] .+ 0.01 .* randn(rng, d)

    G_seq        = Matrix{Float64}(undef, T_days, length(list_of_tickers))
    accept_rates = Vector{Float64}(undef, T_days)

    for day in 1:T_days
        result = mala_sample(X, ξ, T_inner;
                             β=Float64(β_val), α=α,
                             seed=rand(rng, 1:100_000))
        ξ = result.Ξ[end, :]          # warm-start carry-forward
        accept_rates[day] = result.accept_rate

        # project chain state → raw return vector
        logits      = Float64(β_val) .* (X' * ξ)
        p           = softmax(logits)
        G_seq[day, :] = M * p
    end

    # acceptance rate diagnostics
    μ_acc  = mean(accept_rates)
    σ_acc  = std(accept_rates)
    acf1_acc = length(accept_rates) > 2 ?
                   cor(accept_rates[1:end-1], accept_rates[2:end]) : 0.0

    # per-ticker diagnostics
    ks_D_vals    = Float64[]
    abs_acf_vals = Float64[]
    sq_acf_vals  = Float64[]

    for j in eachindex(list_of_tickers)
        g_gen  = G_seq[:, j]
        g_hist = G[:, j]

        push!(ks_D_vals,    ks_D(g_gen, g_hist))
        push!(abs_acf_vals, mean(abs.(autocor(abs.(g_gen),  1:5))))
        push!(sq_acf_vals,  mean(      autocor(g_gen .^ 2, 1:10)))
    end

    SNR = sqrt(α * β_val / (2d))

    row = (
        β              = β_val,
        SNR            = SNR,
        mean_accept    = μ_acc,
        std_accept     = σ_acc,
        acf1_accept    = acf1_acc,
        ks_D_mean      = mean(ks_D_vals),
        acf_abs_1to5   = mean(abs_acf_vals),
        acf_sq_1to10   = mean(sq_acf_vals),
    )
    push!(rows, row)

    @printf("β=%5.0f  SNR=%.4f  accept=%.3f±%.3f  ACF_acc(1)=%.3f  KS D=%.4f  ACF|g|(1-5)=%.4f  ACF g²(1-10)=%.4f\n",
        β_val, SNR, μ_acc, σ_acc, acf1_acc,
        mean(ks_D_vals), mean(abs_acf_vals), mean(sq_acf_vals))
end

# ── 4. Console summary (already printed row-by-row above) ────────────────────────
df = DataFrame(rows)
println("\n── Summary table ───────────────────────────────────────────────────────────")
println(join(rpad.(string.(collect(header)), 16), ""))
for r in eachrow(df)
    @printf("%-16.0f%-16.4f%-16.3f%-16.3f%-16.3f%-16.4f%-16.4f%-16.4f\n",
        r.β, r.SNR, r.mean_accept, r.std_accept, r.acf1_accept,
        r.ks_D_mean, r.acf_abs_1to5, r.acf_sq_1to10)
end

# ── 5. LaTeX table ───────────────────────────────────────────────────────────────
println("""
\\begin{table}[t]
\\centering
\\caption{Finance β sweep (MALA warm-start, T=1500 days, T\\textsubscript{inner}=200).
  \\textit{accept} and \\textit{std(accept)} describe chain mobility;
  ACF(accept,1) measures regime persistence.
  KS \\(\\bar{D}\\) is the mean two-sample KS statistic over 424 tickers (lower = better marginal fit).
  ACF(\\(|g|\\),1--5) and ACF(\\(g^2\\),1--10) are mean autocorrelations of absolute
  and squared returns (higher = stronger ARCH / volatility-clustering signature).}
\\label{tab:finance-beta-sweep}
\\small
\\begin{tabular}{rcccccc}
\\toprule
\$\\beta\$ & SNR & accept & std(accept) & ACF(accept,1) & KS \\(\\bar{D}\\) & ACF(\\(|g|\\),1--5) & ACF(\\(g^2\\),1--10) \\\\
\\midrule""")

for r in rows
    @printf("%.0f & %.4f & %.3f & %.3f & %.3f & %.4f & %.4f & %.4f \\\\\n",
        r.β, r.SNR, r.mean_accept, r.std_accept, r.acf1_accept,
        r.ks_D_mean, r.acf_abs_1to5, r.acf_sq_1to10)
end

println("""\\bottomrule
\\end{tabular}
\\end{table}""")

# ── 7. MCMC artifact diagnostic: β=25, T_inner=2000 vs 200 ──────────────────────
# If ACF(|g|) drops substantially with more mixing steps per day, the vol-clustering
# signal at T_inner=200 is MCMC autocorrelation carry-over, not a real phenomenon.
println("\n── MCMC artifact diagnostic: β=25, T_inner ∈ {200, 2000} ──────────────────")
for T_diag in [200, 2000]
    rng = Random.MersenneTwister(42)
    idx₀ = rand(rng, 1:K)
    ξ = X[:, idx₀] .+ 0.01 .* randn(rng, d)

    G_diag = Matrix{Float64}(undef, T_days, length(list_of_tickers))
    for day in 1:T_days
        result = mala_sample(X, ξ, T_diag; β=25.0, α=α, seed=rand(rng, 1:100_000))
        ξ = result.Ξ[end, :]
        logits = 25.0 .* (X' * ξ)
        p = softmax(logits)
        G_diag[day, :] = M * p
    end

    abs_acf_vals = [mean(abs.(autocor(abs.(G_diag[:, j]), 1:5))) for j in eachindex(list_of_tickers)]
    sq_acf_vals  = [mean(      autocor(G_diag[:, j] .^ 2, 1:10)) for j in eachindex(list_of_tickers)]
    @printf("  T_inner=%4d  ACF(|g|,1-5)=%.4f  ACF(g²,1-10)=%.4f\n",
        T_diag, mean(abs_acf_vals), mean(sq_acf_vals))
end
println("  (If ACF drops at T_inner=2000, the signal is MCMC carry-over, not real vol clustering.)")

println("\nDone.")
