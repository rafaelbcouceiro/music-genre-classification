using DataFrames
using PrettyTables
using Plots
using StatsPlots


"""
    print_results_table(df; filter_model=nothing)

Print summary table of model performance including accuracy and F1 statistics.

# Arguments
- `df::DataFrame`: Results DataFrame.
- `filter_model::Union{Nothing,String}`: Optional filter by model name.

# Returns
- Nothing. Prints a formatted table.
"""
function print_results_table(df::DataFrame; filter_model::Union{Nothing,String}=nothing)
    subdf = filter_model === nothing ? df : filter(:model => ==(filter_model), df)

    table = DataFrame(
        Model = subdf.model,
        Params = subdf.params,
        acc_mean = round.(subdf.acc_mean, digits=4),
        acc_std = round.(subdf.acc_std, digits=4),
        f1_mean = round.(subdf.f1_mean, digits=4),
        f1_std = round.(subdf.f1_std, digits=4),
    )

    pretty_table(table, backend=Val(:text), alignment=:l)
end



"""
    plot_confusion_matrix(results_df, model_type, labels; savepath="confusion_matrix.png")

Plot and save a confusion matrix heatmap for the best model of a given type.

# Arguments
- `results_df::DataFrame`: DataFrame containing model results with columns: model, params, acc_mean, confusion_matrix, etc.
- `model_type::String`: Type of model to plot (e.g., "SVC", "DecisionTreeClassifier", "KNeighborsClassifier", "ANN", "DoME").
- `labels::Vector{String}`: Class labels.
- `savepath::String`: File path where the plot will be saved (default: "confusion_matrix.png").

# Returns
- `Nothing`. Saves the confusion matrix plot as an image.
"""
function plot_confusion_matrix(results_df::DataFrame, model_type::String, labels::Vector{String}; savepath::String="confusion_matrix.png")
    # Filter results for the specified model type
    model_results = filter(row -> row.model == model_type, results_df)
    
    if nrow(model_results) == 0
        error("No results found for model type: $model_type")
    end
    
    # Find the model with highest accuracy
    best_idx = argmax(model_results.acc_mean)
    best_model = model_results[best_idx, :]
    
    # Get confusion matrix and parameters
    confusion_matrix = best_model.confusion_matrix
    model_params = best_model.params
    accuracy = best_model.acc_mean
    
    # Create heatmap
    p = heatmap(confusion_matrix,
                xlabel="Prediction", ylabel="True Label",
                xticks=(1:length(labels), labels),
                yticks=(1:length(labels), labels),
                yflip=true,
                xrotation=90,
                color=:blues,
                aspect_ratio=:equal,
                clims=(0, 350),
                annotations=[(j, i, text(round(Int, confusion_matrix[i, j]), 8, :white)) 
                            for i in 1:size(confusion_matrix, 1) 
                            for j in 1:size(confusion_matrix, 2)],
                colorbar=true,
                size=(800, 700))
    
    # Save confusion matrix as an image
    savefig(savepath)
    
    println("Plotted confusion matrix for best $model_type model with parameters: $model_params")
end

"""
    plot_top_models_boxplot(results_df, top_n; savepath="top_models_boxplot.png")

Plot and save a boxplot comparing the top-N models by mean accuracy.

# Arguments
- `results_df::DataFrame`: DataFrame containing model results.
- `top_n::Int`: Number of top models to select based on mean accuracy.
- `savepath::String`: File path where the plot will be saved (default: "top_models_boxplot.png").

# Returns
- `Nothing`. Saves the boxplot comparing the top-N models.
"""
function plot_top_models_boxplot(results_df::DataFrame, top_n::Int; savepath::String="top_models_boxplot.png")
    # 1. Sort models by mean accuracy descending
    sorted_results = sort(results_df, :acc_mean, rev=true)
    n_select = min(top_n, nrow(sorted_results))
    top_models = sorted_results[1:n_select, :]

    accuracies = Float64[]
    model_names = String[]

    println("SELECTED MODELS (Top $n_select by mean accuracy):")
    for (i, row) in enumerate(eachrow(top_models))
        accuracy_vector = row.accuracy_vector
        model_name = "$(row.model)_$i"   # Name with index to differentiate configs

        append!(accuracies, accuracy_vector)
        append!(model_names, fill(model_name, length(accuracy_vector)))

        println("Top $(i): $model_name with mean accuracy: $(round(row.acc_mean, digits=3))")
    end

    if isempty(accuracies)
        error("No valid models found in the dataset")
    end

    # Create boxplot
    boxplot(model_names, accuracies;
            xlabel = "Model",
            ylabel = "Accuracy",
            title = "Top $n_select Models Comparison Boxplot",
            legend = false,
            xrotation = 45,
            size = (800, 600))

    # Save boxplot
    savefig(savepath)

    println("Boxplot saved with $n_select top models")
end
