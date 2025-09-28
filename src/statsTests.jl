using DataFrames
using HypothesisTests
using Statistics
using PrettyTables
using Plots
using Distributions
using Combinatorics
using StatsPlots   # For heatmap with annotations

"""
    friedman_test_analysis(results_df; top_n=5, α=0.05)

Perform the Friedman test to compare the top-performing ML models.

# Arguments
- `results_df::DataFrame`: DataFrame containing model results. 
  Required columns: `:model`, `:acc_mean`, `:acc_std`, `:accuracy_vector`, `:params`.
- `top_n::Int`: Number of top models to compare (default = 5).
- `α::Float64`: Significance level (default = 0.05).

# Returns
- `Dict`: Dictionary containing test statistics, p-value, rankings, and model data.
"""
function friedman_test_analysis(results_df::DataFrame; top_n::Int=5, α::Float64=0.05)
    println("=== FRIEDMAN TEST - TOP $top_n MODELS ===\n")
    
    # 1. Select top-N models by mean accuracy
    sorted_results = sort(results_df, :acc_mean, rev=true)
    top_models = sorted_results[1:min(top_n, nrow(sorted_results)), :]
    
    model_data = []
    model_names = String[]
    
    println("SELECTED MODELS (Top $top_n by mean accuracy):")
    for (i, row) in enumerate(eachrow(top_models))
        accuracy_vector = row.accuracy_vector
        model_name = "$(row.model)_$i"   # Assign index to differentiate configurations
        push!(model_data, accuracy_vector)
        push!(model_names, model_name)
        
        println("$i. $(row.model): $(round(row.acc_mean, digits=4)) ± $(round(row.acc_std, digits=4))")
        println("   Hyperparameters: $(row.params)")
    end
    
    if length(model_data) < 3
        error("At least 3 models are required for the Friedman test. Found $(length(model_data)).")
    end
    
    println("\n" * "="^60)
    
    # 2. Prepare data matrix (folds × models)
    n_folds = length(model_data[1])
    data_matrix = hcat(model_data...)  
    
    println("DATA SUMMARY:")
    println("- Number of folds: $n_folds")
    println("- Number of models: $(length(model_names))")
    println("- Models compared: $(join(model_names, ", "))")
    
    # 3. Compute rankings per fold
    rankings = zeros(Int, n_folds, length(model_names))
    for fold in 1:n_folds
        fold_accuracies = data_matrix[fold, :]
        rankings[fold, :] = sortperm(sortperm(fold_accuracies, rev=true))
    end
    
    println("\nRANKINGS PER FOLD:")
    ranking_df = DataFrame(rankings, model_names)
    insertcols!(ranking_df, 1, :Fold => 1:n_folds)
    pretty_table(ranking_df, header=names(ranking_df), crop=:none)
    
    # 4. Compute average rankings
    avg_rankings = mean(rankings, dims=1)[1, :]
    
    println("\nAVERAGE RANKING:")
    for (i, model) in enumerate(model_names)
        println("$(model): $(round(avg_rankings[i], digits=2))")
    end
    
    # 5. Friedman test statistic
    n = n_folds
    k = length(model_names)
    R_j = sum(rankings, dims=1)[1, :]
    friedman_stat = (12 / (n * k * (k + 1))) * sum(R_j.^2) - 3 * n * (k + 1)
    dof = k - 1
    p_value = 1 - cdf(Chisq(dof), friedman_stat)
    
    is_significant = p_value < α
    
    println("\n" * "="^60)
    println("FRIEDMAN TEST RESULTS:")
    println("- Friedman statistic: $(round(friedman_stat, digits=4))")
    println("- Degrees of freedom: $dof")
    println("- p-value: $(round(p_value, digits=6))")
    println("- Significance level: $α")
    println("- Significant differences detected? $(is_significant ? "YES" : "NO")")
    
    if is_significant
        println("\nCONCLUSION: There are statistically significant differences among the models.")
        println("At least one model performs significantly different from the others.")
        
        println("\nFINAL RANKING (best to worst):")
        sorted_indices = sortperm(avg_rankings)
        for (pos, idx) in enumerate(sorted_indices)
            println("$pos. $(model_names[idx]) (average ranking: $(round(avg_rankings[idx], digits=2)))")
        end
    else
        println("\nCONCLUSION: No significant differences among the models.")
    end
    
    # 6. Visualization
    create_ranking_plot(model_names, avg_rankings, p_value, α)
    
    return Dict(
        :statistic => friedman_stat,
        :p_value => p_value,
        :dof => dof,
        :significant => is_significant,
        :alpha => α,
        :rankings => avg_rankings,
        :model_names => model_names,
        :raw_data => data_matrix
    )
end

"""
    create_ranking_plot(model_names, avg_rankings, p_value, α)

Create a bar plot showing model rankings.
"""
function create_ranking_plot(model_names, avg_rankings, p_value, α)
    sorted_indices = sortperm(avg_rankings)
    sorted_names = model_names[sorted_indices]
    sorted_rankings = avg_rankings[sorted_indices]
    
    p = bar(sorted_names, sorted_rankings,
            title="Average Rankings - Friedman Test\n(p = $(round(p_value, digits=4)))",
            xlabel="Models",
            ylabel="Average Ranking (lower = better)",
            color=:lightblue,
            legend=false,
            xrotation=45,
            size=(800, 600))
    
    theoretical_avg = (length(model_names) + 1) / 2
    hline!([theoretical_avg], color=:red, linestyle=:dash, 
           label="Expected ranking under null hypothesis")
    
    significance_text = p_value < α ? "Significant Differences" : "No Significant Differences"
    color_text = p_value < α ? :green : :red
    annotate!([(length(sorted_names)/2, maximum(sorted_rankings) * 0.9, 
               text(significance_text, 12, color_text, :bold))])
    
    savefig("results/figures/evaluation/friedman_rankings.png")
    println("\nPlot saved as 'friedman_rankings.png'")
    return p
end

"""
    wilcoxonPostHoc(results_df; top_n=5, α=0.05)

Perform post-hoc pairwise Wilcoxon tests after a significant Friedman test.
p-values are corrected using Holm-Bonferroni to control for multiple comparisons.

# Arguments
- `results_df::DataFrame`: DataFrame containing model results. 
- `top_n::Int`: Number of top models to compare (default = 5).
- `α::Float64`: Significance level (default = 0.05).

# Returns
- `DataFrame`: Pairwise comparison results with adjusted p-values.
"""
function wilcoxonPostHoc(results_df::DataFrame; top_n::Int=5, α::Float64=0.05)
    println("\n=== WILCOXON POST-HOC TEST (TOP $top_n MODELS) ===\n")
    
    # 1. Select top models
    sorted_results = sort(results_df, :acc_mean, rev=true)
    top_models = sorted_results[1:min(top_n, nrow(sorted_results)), :]
    
    model_data = Dict{String, Vector{Float64}}()
    model_names = String[]
    
    println("SELECTED MODELS:")
    for (i, row) in enumerate(eachrow(top_models))
        model_name = "$(row.model)_$i"
        model_data[model_name] = row.accuracy_vector
        push!(model_names, model_name)
        println("$i. $(row.model): $(round(row.acc_mean, digits=4)) ± $(round(row.acc_std, digits=4))")
        println("   Hyperparameters: $(row.params)")
    end
    
    # 2. Pairwise comparisons
    pairs = collect(combinations(model_names, 2))
    raw_pvals, stats = Float64[], Float64[]
    m1_list, m2_list = String[], String[]
    
    println("\nWILCOXON RESULTS PER PAIR:")
    for (m1, m2) in pairs
        acc1, acc2 = model_data[m1], model_data[m2]
        if length(acc1) != length(acc2)
            @warn "Models $m1 and $m2 have different number of folds, comparison skipped."
            continue
        end
        test = SignedRankTest(acc1, acc2)
        push!(m1_list, m1)
        push!(m2_list, m2)
        push!(stats, test.W)
        push!(raw_pvals, pvalue(test))
    end
    
    # 3. Holm-Bonferroni correction
    m = length(raw_pvals)
    sorted_idx = sortperm(raw_pvals)
    adj_pvals = fill(0.0, m)
    for (rank, idx) in enumerate(sorted_idx)
        adj_pvals[idx] = min((m - rank + 1) * raw_pvals[idx], 1.0)
    end
    
    # 4. Build results DataFrame
    results = DataFrame(
        Model1 = m1_list,
        Model2 = m2_list,
        Statistic = stats,
        RawPValue = raw_pvals,
        AdjPValue = adj_pvals,
        Significant = adj_pvals .< α
    )
    
    println("\n" * "="^60)
    println("PAIRWISE COMPARISON TABLE (Holm-Bonferroni correction, α = $α):")
    pretty_table(results, crop=:none)

    # 5. Build significance matrix for heatmap
    n = length(model_names)
    pval_matrix = fill(NaN, n, n)
    
    for row in eachrow(results)
        i = findfirst(==(row.Model1), model_names)
        j = findfirst(==(row.Model2), model_names)
        if !isnothing(i) && !isnothing(j)
            pval_matrix[i, j] = row.AdjPValue
            pval_matrix[j, i] = row.AdjPValue  # simétrico
        end
    end
    
    # Diagonal = 0 (no self-comparison)
    for i in 1:n
        pval_matrix[i, i] = 0.0
    end
    
    # 6. Plot heatmap
    heatmap(
        x_flip=false,
        1:top_n, 1:top_n, pval_matrix;
        xticks=(1:top_n, model_names),
        yticks=(1:top_n, model_names),
        xlabel="Models", 
        ylabel="Models",
        c=:blues, 
        title="Adjusted p-values",
        clims=(0,1), aspect_ratio=1, size=(800,750),
        colorbar=true,
        annotations=[(j, i, text(round(pval_matrix[i, j]; digits=4), 8)) 
                            for i in 1:size(pval_matrix, 1) 
                            for j in 1:size(pval_matrix, 2)],
    )

    savefig("results/figures/evaluation/wilcoxonHeatmap.png")
    
    return results
end
