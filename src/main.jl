include("eda.jl")
include("evaluation.jl")
include("statsTests.jl")
include("trainer.jl")

using CSV
using DataFrames
using JSON3

function load_results_df(path::String)
    raw_df = CSV.read(path, DataFrame)

    # parsea el string de la matriz como expresión Julia
    raw_df.confusion_matrix = [eval(Meta.parse(cm)) for cm in raw_df.confusion_matrix]
    raw_df.accuracy_vector  = [eval(Meta.parse(vec)) for vec in raw_df.accuracy_vector]

    return raw_df
end


result_df = load_results_df("/Users/rafa/Documents/proyects/foundationalModelsTranslation/music-genre-classification/experiment_results.csv")

"""
    MODEL_CONFIGS

Dictionary of model types and their corresponding hyperparameter grids.  
Each entry maps a model identifier (`Symbol`) to a vector of `Dict` hyperparameter configurations.

Models:
- `:SVC`: Support Vector Classifier configurations with different kernels.
- `:DecisionTreeClassifier_norm` / `:DecisionTreeClassifier_not_norm`: Decision trees with varying depths.
- `:KNeighborsClassifier`: k-NN with different neighbor counts.
- `:ANN`: Artificial Neural Network configurations (topology, learning rate, etc.).
- `:DoME`: DoME algorithm with varying maximum nodes and learning rates.
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
        Dict(:topology => [48, 24], :numExecutions => 20, :maxEpochs => 150, :learningRate => 0.01),
        Dict(:topology => [36, 18], :numExecutions => 50, :maxEpochs => 150, :learningRate => 0.01),
        Dict(:topology => [32, 16], :transferFunctions => [relu, relu], :numExecutions => 20, :learningRate => 0.1),
        Dict(:topology => [32, 16], :transferFunctions => [tanh, tanh], :numExecutions => 30, :learningRate => 0.01),
        Dict(:topology => [64, 32], :transferFunctions => [tanh, relu], :numExecutions => 30, :learningRate => 0.01)
    ],

    :DoME => [
        Dict(:maximumNodes => n, :learningRate => lr) for (n, lr) in [(5,0.005),(10,0.01),(15,0.015),
                                                                     (20,0.02),(20,0.03),(30,0.02)]
    ] ∪ [Dict(:maximumNodes => n) for n in (35,40,43,45,50)]
)

"""
#######################
#    LEER DATASET     #
#######################

data_path = "data/songs_dataset.csv"
df = CSV.read(data_path, DataFrame)
select!(df, Not([1, 2]))  # Eliminar columnas de canción y artista
targets = String.(df.genre)
inputs = Matrix{Float32}(select(df, Not("genre")));



# Cargar archivo de índices
indices = load_indices_vector("data/crossValidationIndices.csv");



########################
# OBTENER ESTADÍSTICAS #
########################

cat = ["key", "mode", "time_signature"]
target = "genre"
numerical = setdiff(names(df), ["key", "mode", "genre", "time_signature"]);

# Mostrar tablas y gráficos usados en la memoria
println(basic_statistics(df)); 
println(summary_table(df, numerical)); 
plot_boxplot_by_target(df, "tempo", "genre"); 
plot_correlation_heatmap(df, numerical); 
plot_categorical_distributions_grouped(df, cat, target); 
plot_numerical_distributions_grouped(df, numerical, target);


results = init_results_df()

# Entrenamientos
train_svm(results, inputs, targets, indices);
train_dt(results, inputs, targets, indices, true);
train_dt(results, inputs, targets, indices, false);
train_knn(results, inputs, targets, indices);
#train_ann(results, inputs, targets, indices);
#train_dome(results, inputs, targets, indices);"""

# Mostrar tablas por tipo de modelo
print_results_table(result_df; filter_model="SVC");
print_results_table(result_df; filter_model="DecisionTreeClassifier");
print_results_table(result_df; filter_model="KNeighborsClassifier");
print_results_table(result_df; filter_model="ANN");
#print_results_table(results; filter_model="DoME");

# Generar matrices de confusión (usando índices de fila del DataFrame)
plot_confusion_matrix(result_df, "SVC", unique(targets);
    savepath="results/figures/evaluation/best_svm_confusion.png");   # Ejemplo para SVM
plot_confusion_matrix(result_df, "DecisionTreeClassifier", unique(targets);
    savepath="results/figures/evaluation/best_dt_confusion.png");  # Ejemplo para Decision Tree
plot_confusion_matrix(result_df, "KNeighborsClassifier", unique(targets);
    savepath="results/figures/evaluation/best_knn_confusion.png");  # Ejemplo para KNN
plot_confusion_matrix(result_df, "ANN", unique(targets); 
    savepath="results/figures/evaluation/best_ANN.png");  # Ejemplo para ANN
#plot_confusion_matrix(results, "DoME", unique(targets); savepath = "results/figures/evaluation/best_dome.png");  # Ejemplo para DoME


# Boxplot comparativo
plot_models_boxplot(result_df, ["SVC", "DecisionTreeClassifier", "KNeighborsClassifier", "ANN"];
    savepath="results/figures/evaluation/models_boxplot.png");


# Ejemplo de uso simplificado:

# O comparar los 3 mejores modelos
friedman_results = friedman_test_analysis(result_df; top_n=5);
wilcoxonPostHoc(result_df);