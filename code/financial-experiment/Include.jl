# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "..", "src");
const _PATH_TO_DATA = joinpath(_ROOT, "..", "data");
const _PATH_TO_FIG = joinpath(_ROOT, "..", "figs");


# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using VLDataScienceMachineLearningPackage
using Images
using ImageInTerminal
using FileIO
using ImageIO
using OneHotArrays
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Colors
using Distances
using DataStructures
using Test
using Random
using LinearAlgebra
using Printf
using NNlib
using DataFrames
using StatsPlots
using StatsBase
using Distributions
using PrettyTables
using HypothesisTests

# set the random seed for reproducibility
Random.seed!(1234);

# include the source code for the project -
include(joinpath(_PATH_TO_SRC, "Data.jl"))
include(joinpath(_PATH_TO_SRC, "Compute.jl"))
include(joinpath(_PATH_TO_SRC, "Utilities.jl"))