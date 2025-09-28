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
    plot_models_boxplot(results_df, model_types; savepath="models_boxplot.png")

Plot and save a boxplot comparing the best model of each specified type.

# Arguments
- `results_df::DataFrame`: DataFrame containing model results.
- `model_types::Vector{String}`: List of model types to compare (e.g., ["SVC", "DecisionTreeClassifier", "KNeighborsClassifier"]).
- `savepath::String`: File path where the plot will be saved (default: "models_boxplot.png").

# Returns
- `Nothing`. Saves the boxplot comparing the best models of each type.
"""
function plot_models_boxplot(results_df::DataFrame, model_types::Vector{String}; savepath::String="models_boxplot.png")
    accuracies = Float64[]
    model_names = String[]
    
    for model_type in model_types
        # Filter results for this model type
        model_results = filter(row -> row.model == model_type, results_df)
        
        if nrow(model_results) == 0
            @warn "No results found for model type: $model_type"
            continue
        end
        
        # Find the model with highest accuracy
        best_idx = argmax(model_results.acc_mean)
        best_model = model_results[best_idx, :]
        
        # Get the accuracy vector for the best model
        accuracy_vector = best_model.accuracy_vector
        
        # Add all accuracy values from cross-validation
        append!(accuracies, accuracy_vector)
        append!(model_names, fill(model_type, length(accuracy_vector)))
        
        println("Best $model_type: $(best_model.params) with mean accuracy: $(round(best_model.acc_mean, digits=3))")
    end
    
    if isempty(accuracies)
        error("No valid models found for the specified types")
    end
    
    # Create boxplot
    boxplot(model_names, accuracies;
            xlabel = "Model Type",
            ylabel = "Accuracy",
            title = "Best Models Comparison Boxplot",
            legend = false,
            xrotation = 45,
            size = (800, 600))
    
    # Save boxplot
    savefig(savepath)
    
    println("Boxplot saved with $(length(unique(model_names))) model types")
end