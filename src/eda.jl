using DataFrames, StatsPlots, CSV, Statistics, PrettyTables

# --- Basic statistics function ---
"""
Compute basic descriptive statistics for all columns in a DataFrame.

Args:
    df (DataFrame): Input dataset.

Returns:
    DataFrame with summary statistics.
"""
function basic_statistics(df::DataFrame)
    return describe(df)
end

# --- Dispersion measures function ---
"""
Compute dispersion measures (standard deviation, variance, interquartile range) for numeric columns.

Args:
    df (DataFrame): Input dataset.

Returns:
    DataFrame containing variable name, std, var, and IQR.
"""
function dispersion_summary(df::DataFrame)
    numeric_cols = names(df, eltype.(eachcol(df)) .== Float64)
    return DataFrame(
        variable = numeric_cols,
        std = [std(skipmissing(df[!, col])) for col in numeric_cols],
        var = [var(skipmissing(df[!, col])) for col in numeric_cols],
        iqr = [quantile(skipmissing(df[!, col]), 0.75) - quantile(skipmissing(df[!, col]), 0.25) for col in numeric_cols]
    )
end

"""
Generate a summary table (min, max, mean, std) for selected variables and display it as text.

Args:
    df (DataFrame): Input dataset.
    variables (Vector{String}): List of variables to summarize.

Returns:
    Nothing. Prints the summary table.
"""
function summary_table(df::DataFrame, variables::Vector{String})
    summary = DataFrame(
        Variable = String[],
        Min = Float64[],
        Max = Float64[],
        Mean = Float64[],
        Std = Float64[]
    )

    for var in variables
        push!(summary, (
            string(var),
            round(minimum(skipmissing(df[!, var])), digits=4),
            round(maximum(skipmissing(df[!, var])), digits=4),
            round(mean(skipmissing(df[!, var])), digits=4),
            round(std(skipmissing(df[!, var])), digits=4)
        ))
    end

    # Display in plain text format
    pretty_table(summary, backend = Val(:text), alignment = :l)
end

# --- Correlation matrix function ---
"""
Compute Pearson correlation matrix for selected numeric variables.

Args:
    df (DataFrame): Input dataset.
    numericalNames (Vector{String}): List of numeric column names.

Returns:
    Tuple: (names of numeric variables, correlation matrix).
"""
function pearson_correlation_matrix(df::DataFrame, numericalNames::Vector{String})
    num_df = select(df, numericalNames)
    mat = reduce(hcat, [Float64.(num_df[!, col]) for col in names(num_df)])
    corr_matrix = cor(mat, dims=1)
    return names(num_df), corr_matrix
end

"""
Plot and save a Pearson correlation heatmap for numeric variables.

Args:
    df (DataFrame): Input dataset.
    numericalValues (Vector{String}): List of numeric column names.
    savepath (String, optional): Path to save plot (default: "plots/correlation_heatmap.png").

Returns:
    Nothing. Saves the plot to file.
"""
function plot_correlation_heatmap(df::DataFrame, numericalValues; savepath::String="results/figures/correlation_heatmap.png")
    names, corr = pearson_correlation_matrix(df, numericalValues)
    n = length(names)

    heatmap(corr,
        xflip = false,
        xticks = (1:n, names),
        yticks = (1:n, names),
        c = :coolwarm,
        clim = (-1, 1),
        title = "Pearson Correlation Matrix",
        xlabel = "", ylabel = "",
        xrotation = 90,
        size = (800, 750),
        dpi = 300,
        annotations = [(j, i, text(round(corr[i, j]; digits=2), 8, :black))
                    for i in 1:size(corr, 1) 
                    for j in 1:size(corr, 2)],
    )

    savefig(savepath)
end

# --- Categorical distributions ---
"""
Plot grouped histograms for categorical variables by a target variable, saving multiple plots.

Args:
    df (DataFrame): Input dataset.
    categorical (Vector{String}): List of categorical column names.
    target (String): Target variable for grouping.
    output_dir (String, optional): Directory to save plots (default: "plots/categorical").

Returns:
    Nothing. Saves grouped histograms.
"""
function plot_categorical_distributions_grouped(df::DataFrame, categorical::Vector{String}, target::String; output_dir::String="results/figures/categorical")
    mkpath(output_dir)
    
    for (i, chunk) in enumerate(Iterators.partition(categorical, 4))
        plots = []
        
        for col in chunk
            # Create separate histogram for each categorical variable
            p = StatsPlots.groupedhist(df[!, col], group=df[!, target],
                          xlabel = col, ylabel = "Count",
                          title = "Histogram of $(col) by $(target)",
                          legend = true)
            
            push!(plots, p)
        end
        
        # Combine multiple plots into one layout
        combined_plot = plot(plots..., layout = (2, 2), size=(1000, 800), dpi=300)
        savefig(combined_plot, joinpath(output_dir, "grouped_histograms_$(i).png"))
    end
end

# --- Numerical distributions ---
"""
Plot grouped boxplots for numeric variables by a target variable, saving multiple plots.

Args:
    df (DataFrame): Input dataset.
    numerical (Vector{String}): List of numeric column names.
    target (String): Target variable for grouping.
    output_dir (String, optional): Directory to save plots (default: "plots/numerical").

Returns:
    Nothing. Saves grouped boxplots.
"""
function plot_numerical_distributions_grouped(df::DataFrame, numerical::Vector{String}, target::String; output_dir::String="results/figures/numerical")
    mkpath(output_dir)
    n = length(numerical)
    for (i, chunk) in enumerate(Iterators.partition(numerical, 4))
        plt = plot(layout = (2, 2), size=(1000, 800), dpi=300)
        for (j, col) in enumerate(chunk)
            boxplot!(plt[j], df[!, target], df[!, col],
                xlabel = "", ylabel = col,
                title = "Boxplot of $(col) by $(target)",
                legend = false,
                xticks = :auto,
                xrotation = 90
            )
        end
        savefig(plt, joinpath(output_dir, "grouped_boxplots_$(i).png"))
    end
end

# --- Single variable numerical distribution ---
"""
Plot a boxplot of a single numeric variable by a target variable.

Args:
    df (DataFrame): Input dataset.
    variable (String): Numeric variable to plot.
    target (String): Target variable for grouping.
    output_path (String, optional): Path to save plot (default: "plots/tempo_boxplot.png").

Returns:
    Nothing. Saves the boxplot.
"""
function plot_boxplot_by_target(df::DataFrame, variable::String, target::String; output_path::String="results/figures/tempo_by_genre.png")

    mkpath(dirname(output_path))
    plt = boxplot(df[!, target], df[!, variable],
        xlabel = target,
        ylabel = variable,
        title = "Boxplot of $(variable) by $(target)",
        legend = false,
        xticks = :auto,
        xrotation = 45,
        dpi = 300,
        size = (800, 650)
    )
    savefig(plt, output_path)
end
