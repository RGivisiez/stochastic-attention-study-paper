#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Before/After Figure for Digit 8
# Shows stored patterns (input) alongside generated samples (output) from the
# same chains, making the generation process visually concrete.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── decode helper ─────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    X̂ = replace(X, -1 => 0)
    return X̂
end

# ── Parameters (same as main experiment) ──────────────────────────────────────
const digit = 8
const number_of_examples = 100
const number_of_pixels = 28 * 28
const α_step = 0.01
const β_inv_temp = 2000.0
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const σ_init = 0.01

# ── Load MNIST ────────────────────────────────────────────────────────────────
@info "Loading MNIST data …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)

# ── Build memory matrix for digit 8 ──────────────────────────────────────────
ϵ = 1e-12
X  = zeros(Float64, number_of_pixels, number_of_examples)
X̂  = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    image_array = digits_image_dictionary[digit][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, number_of_pixels) |> vec
    X[:, i] = xᵢ
end
for i in 1:number_of_examples
    xᵢ = X[:, i]
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
K = size(X̂, 2)
@info "Memory matrix: $(size(X̂))"

# ── Select 8 chains and run SA ───────────────────────────────────────────────
n_show = 8  # number of before/after pairs to display
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K, 30, replace=false)  # same as experiment
show_indices = pattern_indices[1:n_show]  # take first 8

stored_patterns = Vector{Vector{Float64}}()   # the "before" images
generated_samples = Vector{Vector{Float64}}() # the "after" images

for (c, k) in enumerate(show_indices)
    # Store the seed pattern
    push!(stored_patterns, X̂[:, k])
    
    # Run SA chain from near this pattern
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
    (_, Ξ) = sample(X̂, sₒ, T_per_chain; β = β_inv_temp, α = α_step, seed = 12345 + c)
    
    # Take one representative post-burn-in sample from this chain
    # Use the last thinned sample for maximum separation from initialization
    last_idx = T_per_chain  # last row of trajectory
    push!(generated_samples, Ξ[last_idx, :])
end

@info "Generated $(length(generated_samples)) before/after pairs"

# ── Build the 2-row figure ──────────────────────────────────────────────────
# Top row: stored patterns (what each chain started near)
# Bottom row: generated samples (what came out)
H, W = 28, 28
gap = 2          # gap between images
row_gap = 6      # larger gap between "before" and "after" rows

canvas_w = n_show * W + (n_show - 1) * gap
canvas_h = 2 * H + row_gap

canvas = zeros(Float64, canvas_h, canvas_w)

for i in 1:n_show
    x0 = (i - 1) * (W + gap) + 1
    
    # Top row: stored pattern
    img_top = decode(stored_patterns[i])
    lo, hi = minimum(img_top), maximum(img_top)
    if hi > lo; img_top = (img_top .- lo) ./ (hi - lo); end
    canvas[1:H, x0:x0+W-1] .= img_top
    
    # Bottom row: generated sample
    img_bot = decode(generated_samples[i])
    lo, hi = minimum(img_bot), maximum(img_bot)
    if hi > lo; img_bot = (img_bot .- lo) ./ (hi - lo); end
    canvas[H+row_gap+1:2*H+row_gap, x0:x0+W-1] .= img_bot
end

figpath = _PATH_TO_FIG
fname = "Fig_mnist_before_after_digit8.png"
save(joinpath(figpath, fname), Gray.(canvas))
@info "Saved $fname to $figpath"

# ── Also compute cosine similarity between each pair ─────────────────────────
println("\nPer-chain cosine similarity (stored → generated):")
for i in 1:n_show
    cs = dot(stored_patterns[i], generated_samples[i]) / 
         (norm(stored_patterns[i]) * norm(generated_samples[i]))
    println("  Chain $i (pattern $(show_indices[i])): cos = $(round(cs, digits=4))")
end
mean_cs = mean(
    dot(stored_patterns[i], generated_samples[i]) / 
    (norm(stored_patterns[i]) * norm(generated_samples[i]))
    for i in 1:n_show
)
println("  Mean: $(round(mean_cs, digits=4))")
