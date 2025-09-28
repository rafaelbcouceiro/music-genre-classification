# =========================
#   MUSIC GENRE CLASSIFICATION PIPELINE
# =========================
# This script performs an end-to-end ML experiment on a music dataset. 
# It includes EDA, model training, evaluation, statistical tests, and visualization.
# The results are stored and visualized for comparison across multiple classifiers.

# -------------------------
#   Load necessary modules
# -------------------------
include("eda.jl")         # Exploratory Data Analysis utilities
include("evaluation.jl")  # Evaluation and plotting functions
include("statsTests.jl")  # Statistical tests for model comparison
include("trainer.jl")     # Training routines for different models


# -------------------------
#   MODEL CONFIGURATIONS
# -------------------------
"""
    MODEL_CONFIGS

Dictionary mapping model types (`Symbol`) to vectors of hyperparameter configurations (`Dict`).

Supported models:
- `:SVC` : Support Vector Classifier with various kernels and regularizations.
- `:DecisionTreeClassifier_norm` / `:DecisionTreeClassifier_not_norm` : Decision Trees with different max depths.
- `:KNeighborsClassifier` : k-Nearest Neighbors with varying neighbor counts.
- `:ANN` : Artificial Neural Networks with different topologies, learning rates, and transfer functions.
- `:DoME` : DoME algorithm with different maximum nodes and learning rates.
"""
const MODEL_CONFIGS = Dict(
    :SVC => [
        Dict(:kernel => "rbf", :gamma => 0.01, :C => 1),
        Dict(:kernel => "linear", :C => 0.1),
        Dict(:kernel => "linear", :C => 1),
        Dict(:kernel => "poly", :C => 1, :gamma => 0.1, :coef0 => 1, :degree => 2),
        Dict(:kernel => "poly", :C => 10, :gamma => 0.1, :coef0 => 1, :degree => 2),
        Dict(:kernel => "poly", :C => 1, :gamma => 0.01, :coef0 => 1, :degree => 3),
        Dict(:kernel => "sigmoid", :C => 1, :gamma => 0.1, :coef0 => 1)
    ],
    :DecisionTreeClassifier_norm => [Dict(:max_depth => d) for d in (5,6,8,9,10,12,13,14,15,20,11)],
    :DecisionTreeClassifier_not_norm => [Dict(:max_depth => d) for d in (5,6,8,9,10,12,13,14,15,20,11)],
    :KNeighborsClassifier => [Dict(:n_neighbors => k) for k in (3,5,7,10,15,18,20,22)],
    :ANN => [
        Dict(:topology => [48, 24], :learningRate => 0.01),
        Dict(:topology => [36, 18], :learningRate => 0.01),
        Dict(:topology => [32, 16], :transferFunctions => [relu, relu], :learningRate => 0.1),
        Dict(:topology => [32, 16], :transferFunctions => [tanh, tanh], :learningRate => 0.01),
        Dict(:topology => [64, 32], :transferFunctions => [tanh, relu], :learningRate => 0.01)
    ],
    :DoME => [
        Dict(:maximumNodes => n, :learningRate => lr) for (n, lr) in [(5,0.005),(10,0.01),(15,0.015),
                                                                     (20,0.02),(20,0.03),(30,0.02)]
    ] ∪ [Dict(:maximumNodes => n) for n in (35,40,43,45,50)]
)


# =========================
#   LOAD DATASET
# =========================
data_path = "data/songs_dataset.csv"
df = CSV.read(data_path, DataFrame)

# Remove non-numeric columns not used as features
select!(df, Not([1, 2]))  # Drop song title and artist columns

# Separate target variable and input features
targets = String.(df.genre)
inputs = Matrix{Float32}(select(df, Not("genre")))

# Load cross-validation indices
indices = load_indices_vector("data/crossValidationIndices.csv")


# =========================
#   EXPLORATORY DATA ANALYSIS
# =========================
categorical_features = ["key", "mode", "time_signature"]
numerical_features = setdiff(names(df), ["key", "mode", "genre", "time_signature"])

# Display basic statistics
println(basic_statistics(df))
println(summary_table(df, numerical_features))

# Visualizations
plot_boxplot_by_target(df, "tempo", "genre")
plot_correlation_heatmap(df, numerical_features)
plot_categorical_distributions_grouped(df, categorical_features, "genre")
plot_numerical_distributions_grouped(df, numerical_features, "genre")


# =========================
#   INITIALIZE RESULTS DATAFRAME
# =========================
results = init_results_df()


# =========================
#   MODEL TRAINING
# =========================
train_svm(results, inputs, targets, indices)
train_dt(results, inputs, targets, indices, true)   # Normalized features
train_dt(results, inputs, targets, indices, false)  # Non-normalized
train_knn(results, inputs, targets, indices)
train_ann(results, inputs, targets, indices)
train_dome(results, inputs, targets, indices)


# =========================
#   RESULTS VISUALIZATION
# =========================
# Print summary tables per model type
print_results_table(results; filter_model="SVC")
print_results_table(results; filter_model="DecisionTreeClassifier")
print_results_table(results; filter_model="KNeighborsClassifier")
print_results_table(results; filter_model="ANN")
print_results_table(results; filter_model="DoME")

# Generate confusion matrices
plot_confusion_matrix(results, "SVC", unique(targets); 
    savepath="results/figures/evaluation/best_svm_confusion.png")
plot_confusion_matrix(results, "DecisionTreeClassifier", unique(targets); 
    savepath="results/figures/evaluation/best_dt_confusion.png")
plot_confusion_matrix(results, "KNeighborsClassifier", unique(targets); 
    savepath="results/figures/evaluation/best_knn_confusion.png")
plot_confusion_matrix(results, "ANN", unique(targets); 
    savepath="results/figures/evaluation/best_ANN.png")
plot_confusion_matrix(results, "DoME", unique(targets); 
    savepath="results/figures/evaluation/best_dome.png")


# Comparative boxplot of top models
plot_top_models_boxplot(results, 5; 
    savepath="results/figures/evaluation/models_boxplot.png")


# =========================
#   STATISTICAL TESTS
# =========================
# Perform Friedman test on top N models
friedman_results = friedman_test_analysis(results; top_n=5)

# Perform post-hoc Wilcoxon test for pairwise comparisons
wilcoxonPostHoc(results)
