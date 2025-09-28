using CSV
using DataFrames
using MLJ
using LIBSVM
using FilePathsBase
using Plots
using StatsPlots
using Statistics
using PrettyTables
using StatsBase: mode
include("/content/models.jl") 



SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0 


"""
    init_results_df()

Initialize an empty DataFrame for storing model evaluation results.

# Returns
- `DataFrame`: An empty results table with predefined schema:
  - `model::String`
  - `params::String`
  - `acc_mean::Float64`
  - `acc_std::Float64`
  - `f1_mean::Float64`
  - `f1_std::Float64`
  - `confusion_matrix::Any`
  - `accuracy_vector::Any`
"""
function init_results_df()
    return DataFrame(
        model = String[],
        params = String[],  # stringified Dict for readability
        acc_mean = Float64[],
        acc_std = Float64[],
        f1_mean = Float64[],
        f1_std = Float64[],
        confusion_matrix = Any[],
        accuracy_vector = Any[]
    )
end


"""
    add_result!(df, modelType, params, stats)

Add one experiment result to the results DataFrame.

# Arguments
- `df::DataFrame`: DataFrame of results.
- `modelType::Symbol`: Model identifier.
- `params::Dict`: Hyperparameter dictionary.
- `stats::Tuple`: Output from `modelCrossValidation`.

# Returns
- `DataFrame`: Updated DataFrame with one new row appended.
"""
function add_result!(df::DataFrame, modelType::Symbol, params::Dict, stats)
    push!(df, (
        string(modelType),
        string(params),   # store Dict as string for readability
        stats[1][1],      # acc mean
        stats[1][2],      # acc std
        stats[7][1],      # f1 mean
        stats[7][2],      # f1 std
        stats[8],         # confusion matrix
        stats[9]          # accuracy vector
    ))
    return df
end


"""
    run_experiments!(df, modelType, param_grid, inputs, targets, indices; normalize=true)

Run cross-validation for a set of hyperparameter configurations and log results into a DataFrame.

# Arguments
- `df::DataFrame`: DataFrame to store results.
- `modelType::Symbol`: Model identifier (e.g., `:SVC`, `:KNeighborsClassifier`).
- `param_grid::Vector{Dict}`: List of hyperparameter dictionaries to test.
- `inputs::Matrix`: Feature matrix.
- `targets::Vector`: Target labels.
- `indices`: Cross-validation indices.
- `normalize::Bool`: Whether to normalize data (default: `true`).

# Returns
- `DataFrame`: Updated DataFrame with appended experiment results.
"""
function run_experiments!(df::DataFrame, modelType::Symbol, param_grid::Vector,
                          inputs, targets, indices; normalize=true)

    for config in param_grid
        stats = modelCrossValidation(modelType, config, (inputs, targets), indices; normalize=normalize)
        println("[$(modelType)] config: $config → acc=$(round(stats[1][1], digits=3)) f1=$(round(stats[7][1], digits=3))")
        add_result!(df, modelType, config, stats)
    end

    return df
end

"""
    train_svm(results, inputs, targets, indices)

Train and evaluate SVM classifiers across predefined configurations.
"""
function train_svm(results, inputs, targets, indices)
    run_experiments!(results, :SVC, MODEL_CONFIGS[:SVC], inputs, targets, indices)
end

"""
    train_dt(results, inputs, targets, indices, normalize)

Train and evaluate Decision Tree classifiers with varying depths.

# Arguments
- `normalize::Bool`: Whether to normalize input features.
"""
function train_dt(results, inputs, targets, indices, normalize::Bool)
    run_experiments!(results, :DecisionTreeClassifier, MODEL_CONFIGS[:DecisionTreeClassifier_norm], inputs, targets, indices; normalize=normalize)
end

"""
    train_knn(results, inputs, targets, indices)

Train and evaluate k-NN classifiers across different neighbor counts.
"""
function train_knn(results, inputs, targets, indices)
    run_experiments!(results, :KNeighborsClassifier, MODEL_CONFIGS[:KNeighborsClassifier], inputs, targets, indices)
end

"""
    train_ann(results, inputs, targets, indices)

Train and evaluate Artificial Neural Networks across predefined topologies.
"""
function train_ann(results, inputs, targets, indices)
    run_experiments!(results, :ANN, MODEL_CONFIGS[:ANN], inputs, targets, indices)
end

"""
    train_dome(results, inputs, targets, indices)

Train and evaluate DoME models across varying node/learning rate configurations.
"""
function train_dome(results, inputs, targets, indices)
    run_experiments!(results, :DoME, MODEL_CONFIGS[:DoME], inputs, targets, indices)
end