module CourtAnalyzer

using Statistics
using CSV
using DataFrames
using Plots
using JSON
using Images
using FileIO
using ONNX
using ONNXRunTime
using Dates
using StatsBase
using HypothesisTests
using LinearAlgebra
using Distributions
using HTTP
using Colors
using ColorSchemes

export analyze_models, load_predictions, compare_models, generate_report, visualize_results, analyze_metrics

"""
Configuration for the analysis tool
"""
mutable struct AnalyzerConfig
    predictions_dir::String
    ground_truth_path::String
    output_dir::String
    models::Vector{String}
    metrics::Vector{Symbol}
    confidence_level::Float64
    
    function AnalyzerConfig(;
        predictions_dir::String = "predictions",
        ground_truth_path::String = "ground_truth.json",
        output_dir::String = "analysis_results",
        models::Vector{String} = String[],
        metrics::Vector{Symbol} = [:kla_05, :kla_10, :cd_iou, :pcr],
        confidence_level::Float64 = 0.95
    )
        return new(predictions_dir, ground_truth_path, output_dir, models, metrics, confidence_level)
    end
end

"""
Model prediction data structure
"""
struct ModelPredictions
    model_name::String
    keypoints::Vector{Matrix{Float64}}  # Each element is a 4×2 keypoint matrix
    image_ids::Vector{String}
    confidence_scores::Vector{Float64}
    inference_times::Vector{Float64}
end

"""
Ground truth annotation structure
"""
struct GroundTruth
    keypoints::Vector{Matrix{Float64}}  # Each element is a 4×3 matrix (x, y, visibility)
    image_ids::Vector{String}
    court_types::Vector{Int}
end

"""
Analysis result for a single model
"""
struct ModelAnalysis
    model_name::String
    metrics::Dict{Symbol, Float64}  # Mean metrics
    metrics_std::Dict{Symbol, Float64}  # Standard deviation
    metrics_ci::Dict{Symbol, Tuple{Float64, Float64}}  # Confidence intervals
    per_image_metrics::Dict{String, Dict{Symbol, Float64}}  # Metrics per image
    success_rate::Dict{Symbol, Float64}  # Success rate per metric threshold
    error_analysis::Dict{Symbol, Any}  # Additional error analysis
end

"""
Load ground truth annotations from a JSON file
"""
function load_ground_truth(path::String)
    if !isfile(path)
        error("Ground truth file not found: $(path)")
    end
    
    # Load JSON
    data = JSON.parsefile(path)
    
    # Extract data
    keypoints = Matrix{Float64}[]
    image_ids = String[]
    court_types = Int[]
    
    for (image_id, annotation) in data
        if haskey(annotation, "keypoints")
            # Convert to 4×3 matrix
            kpts = zeros(Float64, 4, 3)
            for (i, point) in enumerate(annotation["keypoints"])
                if i <= 4 && length(point) >= 3
                    kpts[i, 1] = point[1]  # x
                    kpts[i, 2] = point[2]  # y
                    kpts[i, 3] = point[3]  # visibility
                end
            end
            
            push!(keypoints, kpts)
            push!(image_ids, image_id)
            
            # Extract court type if available
            court_type = haskey(annotation, "court_type") ? annotation["court_type"] : 0
            push!(court_types, court_type)
        end
    end
    
    return GroundTruth(keypoints, image_ids, court_types)
end

"""
Load model predictions from files
"""
function load_predictions(model_name::String, predictions_dir::String)
    model_dir = joinpath(predictions_dir, model_name)
    
    if !isdir(model_dir)
        error("Predictions directory not found for model $(model_name): $(model_dir)")
    end
    
    # Load predictions.json if available
    pred_path = joinpath(model_dir, "predictions.json")
    if isfile(pred_path)
        return load_predictions_from_json(model_name, pred_path)
    end
    
    # Otherwise, load from individual files
    keypoints = Matrix{Float64}[]
    image_ids = String[]
    confidence_scores = Float64[]
    inference_times = Float64[]
    
    # List prediction files
    for file in readdir(model_dir)
        if endswith(file, ".json")
            try
                # Load prediction file
                pred_data = JSON.parsefile(joinpath(model_dir, file))
                
                # Extract image ID from filename
                image_id = replace(file, r"\.json$" => "")
                
                # Extract keypoints
                if haskey(pred_data, "keypoints") && length(pred_data["keypoints"]) >= 4
                    kpts = zeros(Float64, 4, 2)
                    for (i, point) in enumerate(pred_data["keypoints"])
                        if i <= 4 && length(point) >= 2
                            kpts[i, 1] = point[1]  # x
                            kpts[i, 2] = point[2]  # y
                        end
                    end
                    
                    push!(keypoints, kpts)
                    push!(image_ids, image_id)
                    
                    # Extract confidence score if available
                    conf = haskey(pred_data, "confidence") ? pred_data["confidence"] : 1.0
                    push!(confidence_scores, conf)
                    
                    # Extract inference time if available
                    inf_time = haskey(pred_data, "inference_time") ? pred_data["inference_time"] : 0.0
                    push!(inference_times, inf_time)
                end
            catch e
                @warn "Error loading prediction file $file: $e"
            end
        end
    end
    
    if isempty(keypoints)
        @warn "No valid predictions found for model $model_name"
    end
    
    return ModelPredictions(model_name, keypoints, image_ids, confidence_scores, inference_times)
end

"""
Load predictions from a single JSON file
"""
function load_predictions_from_json(model_name::String, file_path::String)
    # Load JSON
    data = JSON.parsefile(file_path)
    
    keypoints = Matrix{Float64}[]
    image_ids = String[]
    confidence_scores = Float64[]
    inference_times = Float64[]
    
    for (image_id, pred) in data
        if haskey(pred, "keypoints")
            # Convert to 4×2 matrix
            kpts = zeros(Float64, 4, 2)
            for (i, point) in enumerate(pred["keypoints"])
                if i <= 4 && length(point) >= 2
                    kpts[i, 1] = point[1]  # x
                    kpts[i, 2] = point[2]  # y
                end
            end
            
            push!(keypoints, kpts)
            push!(image_ids, image_id)
            
            # Extract confidence score if available
            conf = haskey(pred, "confidence") ? pred["confidence"] : 1.0
            push!(confidence_scores, conf)
            
            # Extract inference time if available
            inf_time = haskey(pred, "inference_time") ? pred["inference_time"] : 0.0
            push!(inference_times, inf_time)
        end
    end
    
    return ModelPredictions(model_name, keypoints, image_ids, confidence_scores, inference_times)
end

"""
Compute keypoint localization accuracy
"""
function compute_kla(pred_keypoints::Matrix{Float64}, gt_keypoints::Matrix{Float64}, threshold::Float64)
    correct_keypoints = 0
    total_keypoints = 0
    
    # Get image diagonal (assuming normalized coordinates)
    diagonal = sqrt(1.0^2 + 1.0^2)
    
    # Check each keypoint
    for i in 1:4
        # Check if ground truth keypoint is visible
        if gt_keypoints[i, 3] > 0
            total_keypoints += 1
            
            # Calculate distance
            distance = norm(pred_keypoints[i, 1:2] - gt_keypoints[i, 1:2])
            
            # Check if distance is below threshold
            if distance < threshold * diagonal
                correct_keypoints += 1
            end
        end
    end
    
    # Return accuracy (or 0 if no visible keypoints)
    return total_keypoints > 0 ? correct_keypoints / total_keypoints : 0.0
end

"""
Compute court detection IoU
"""
function compute_court_iou(pred_keypoints::Matrix{Float64}, gt_keypoints::Matrix{Float64})
    # Extract visible keypoints from ground truth
    visible_indices = findall(gt_keypoints[:, 3] .> 0)
    
    if length(visible_indices) < 3
        # Not enough visible points to form a polygon
        return 0.0
    end
    
    # Simple implementation using a grid-based approach
    # In a real-world scenario, we would use a more efficient algorithm
    
    # Get bounding box
    min_x = min(minimum(pred_keypoints[:, 1]), minimum(gt_keypoints[visible_indices, 1]))
    max_x = max(maximum(pred_keypoints[:, 1]), maximum(gt_keypoints[visible_indices, 1]))
    min_y = min(minimum(pred_keypoints[:, 2]), minimum(gt_keypoints[visible_indices, 2]))
    max_y = max(maximum(pred_keypoints[:, 2]), maximum(gt_keypoints[visible_indices, 2]))
    
    # Create grid
    grid_size = 100
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    
    intersection = 0
    union = 0
    
    for i in 0:grid_size
        for j in 0:grid_size
            x = min_x + i * step_x
            y = min_y + j * step_y
            
            # Check if point is inside predicted polygon
            in_pred = point_in_polygon([x, y], pred_keypoints)
            
            # Check if point is inside ground truth polygon (visible points only)
            in_gt = point_in_polygon([x, y], gt_keypoints[visible_indices, 1:2])
            
            if in_pred && in_gt
                intersection += 1
            end
            
            if in_pred || in_gt
                union += 1
            end
        end
    end
    
    # Return IoU (or 0 if union is empty)
    return union > 0 ? intersection / union : 0.0
end

"""
Check if a point is inside a polygon
"""
function point_in_polygon(point::Vector{Float64}, polygon::Matrix{Float64})
    n = size(polygon, 1)
    inside = false
    
    j = n
    for i in 1:n
        if ((polygon[i, 2] > point[2]) != (polygon[j, 2] > point[2])) &&
           (point[1] < polygon[i, 1] + (polygon[j, 1] - polygon[i, 1]) * 
            (point[2] - polygon[i, 2]) / (polygon[j, 2] - polygon[i, 2]))
            inside = !inside
        end
        j = i
    end
    
    return inside
end

"""
Compute perfect court rate
"""
function compute_pcr(pred_keypoints::Matrix{Float64}, gt_keypoints::Matrix{Float64}, threshold::Float64)
    # Check if all keypoints are correctly localized
    diagonal = sqrt(1.0^2 + 1.0^2)
    all_correct = true
    
    for i in 1:4
        # Check if ground truth keypoint is visible
        if gt_keypoints[i, 3] > 0
            # Calculate distance
            distance = norm(pred_keypoints[i, 1:2] - gt_keypoints[i, 1:2])
            
            # Check if distance is below threshold
            if distance >= threshold * diagonal
                all_correct = false
                break
            end
        end
    end
    
    return all_correct ? 1.0 : 0.0
end

"""
Compute all metrics for one image
"""
function compute_metrics(pred_keypoints::Matrix{Float64}, gt_keypoints::Matrix{Float64})
    # Compute metrics
    kla_05 = compute_kla(pred_keypoints, gt_keypoints, 0.05)
    kla_10 = compute_kla(pred_keypoints, gt_keypoints, 0.10)
    cd_iou = compute_court_iou(pred_keypoints, gt_keypoints)
    pcr = compute_pcr(pred_keypoints, gt_keypoints, 0.05)
    
    return Dict(
        :kla_05 => kla_05,
        :kla_10 => kla_10,
        :cd_iou => cd_iou,
        :pcr => pcr
    )
end

"""
Calculate confidence interval
"""
function confidence_interval(values::Vector{Float64}, confidence_level::Float64)
    n = length(values)
    if n <= 1
        return (0.0, 0.0)
    end
    
    mean_val = mean(values)
    std_val = std(values)
    
    # Critical value for the specified confidence level
    α = 1 - confidence_level
    crit_val = quantile(Normal(), 1 - α/2)
    
    # Compute margin of error
    margin = crit_val * std_val / sqrt(n)
    
    return (mean_val - margin, mean_val + margin)
end

"""
Analyze error patterns
"""
function analyze_errors(pred_keypoints::Vector{Matrix{Float64}}, gt_keypoints::Vector{Matrix{Float64}}, 
                       image_ids::Vector{String}, metrics::Dict{String, Dict{Symbol, Float64}})
    # Find images with errors
    error_images = String[]
    for (img_id, img_metrics) in metrics
        if img_metrics[:pcr] < 1.0
            push!(error_images, img_id)
        end
    end
    
    # Analyze error patterns
    error_analysis = Dict{Symbol, Any}(
        :total_errors => length(error_images),
        :error_rate => length(error_images) / length(image_ids),
        :error_types => Dict{Symbol, Int}(),
        :error_by_point => zeros(Int, 4)
    )
    
    # Analyze each error
    for img_id in error_images
        idx = findfirst(id -> id == img_id, image_ids)
        if idx === nothing
            continue
        end
        
        pred = pred_keypoints[idx]
        gt = gt_keypoints[idx]
        
        # Find which points are erroneously detected
        for i in 1:4
            if gt[i, 3] > 0  # If point is visible
                distance = norm(pred[i, 1:2] - gt[i, 1:2])
                diagonal = sqrt(1.0^2 + 1.0^2)
                
                if distance >= 0.05 * diagonal
                    error_analysis[:error_by_point][i] += 1
                end
            end
        end
    end
    
    return error_analysis
end

"""
Compare predictions with ground truth for a single model
"""
function analyze_model(predictions::ModelPredictions, ground_truth::GroundTruth, config::AnalyzerConfig)
    # Find common image IDs
    common_ids = intersect(predictions.image_ids, ground_truth.image_ids)
    
    if isempty(common_ids)
        error("No common images found between predictions and ground truth")
    end
    
    # Calculate metrics for each image
    per_image_metrics = Dict{String, Dict{Symbol, Float64}}()
    
    # Filter predictions and ground truth to common IDs
    filtered_pred_keypoints = Matrix{Float64}[]
    filtered_gt_keypoints = Matrix{Float64}[]
    filtered_ids = String[]
    
    for id in common_ids
        pred_idx = findfirst(x -> x == id, predictions.image_ids)
        gt_idx = findfirst(x -> x == id, ground_truth.image_ids)
        
        if pred_idx !== nothing && gt_idx !== nothing
            push!(filtered_pred_keypoints, predictions.keypoints[pred_idx])
            push!(filtered_gt_keypoints, ground_truth.keypoints[gt_idx])
            push!(filtered_ids, id)
            
            # Compute metrics for this image
            metrics = compute_metrics(predictions.keypoints[pred_idx], ground_truth.keypoints[gt_idx])
            per_image_metrics[id] = metrics
        end
    end
    
    # Calculate aggregate metrics
    metrics_values = Dict{Symbol, Vector{Float64}}()
    for metric in config.metrics
        metrics_values[metric] = [metrics[metric] for (_, metrics) in per_image_metrics]
    end
    
    # Compute mean metrics
    mean_metrics = Dict{Symbol, Float64}(
        metric => mean(values) for (metric, values) in metrics_values
    )
    
    # Compute standard deviation
    std_metrics = Dict{Symbol, Float64}(
        metric => std(values) for (metric, values) in metrics_values
    )
    
    # Compute confidence intervals
    ci_metrics = Dict{Symbol, Tuple{Float64, Float64}}(
        metric => confidence_interval(values, config.confidence_level) 
        for (metric, values) in metrics_values
    )
    
    # Compute success rates at different thresholds
    success_rate = Dict{Symbol, Float64}()
    
    # KLA success rate at 0.05
    success_rate[:kla_05_success] = mean([m[:kla_05] == 1.0 ? 1.0 : 0.0 for (_, m) in per_image_metrics])
    
    # KLA success rate at 0.10
    success_rate[:kla_10_success] = mean([m[:kla_10] == 1.0 ? 1.0 : 0.0 for (_, m) in per_image_metrics])
    
    # IoU success rate at 0.7
    success_rate[:iou_70_success] = mean([m[:cd_iou] >= 0.7 ? 1.0 : 0.0 for (_, m) in per_image_metrics])
    
    # PCR success rate
    success_rate[:pcr_success] = mean([m[:pcr] for (_, m) in per_image_metrics])
    
    # Analyze error patterns
    error_analysis = analyze_errors(filtered_pred_keypoints, filtered_gt_keypoints, filtered_ids, per_image_metrics)
    
    # Add inference time if available
    if !isempty(predictions.inference_times)
        mean_metrics[:inference_time] = mean(predictions.inference_times)
        std_metrics[:inference_time] = std(predictions.inference_times)
        ci_metrics[:inference_time] = confidence_interval(predictions.inference_times, config.confidence_level)
    end
    
    return ModelAnalysis(
        predictions.model_name,
        mean_metrics,
        std_metrics,
        ci_metrics,
        per_image_metrics,
        success_rate,
        error_analysis
    )
end

"""
Analyze multiple models and generate comparative results
"""
function analyze_models(config::AnalyzerConfig)
    # Load ground truth
    ground_truth = load_ground_truth(config.ground_truth_path)
    
    # If no models specified, find all subdirectories in predictions_dir
    if isempty(config.models)
        if isdir(config.predictions_dir)
            config.models = [d for d in readdir(config.predictions_dir) if isdir(joinpath(config.predictions_dir, d))]
        else
            error("Predictions directory not found: $(config.predictions_dir)")
        end
    end
    
    if isempty(config.models)
        error("No models found in predictions directory: $(config.predictions_dir)")
    end
    
    # Analyze each model
    model_analyses = Dict{String, ModelAnalysis}()
    for model_name in config.models
        println("Analyzing model: $(model_name)")
        
        try
            # Load predictions
            predictions = load_predictions(model_name, config.predictions_dir)
            
            # Analyze model
            analysis = analyze_model(predictions, ground_truth, config)
            
            # Store analysis
            model_analyses[model_name] = analysis
            
        catch e
            @warn "Error analyzing model $(model_name): $(e)"
        end
    end
    
    return model_analyses
end

"""
Compare models using statistical tests
"""
function compare_models(analyses::Dict{String, ModelAnalysis}, config::AnalyzerConfig)
    if length(analyses) < 2
        @warn "Need at least 2 models for comparison"
        return nothing
    end
    
    # Create results structure
    comparison_results = Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}()
    
    # Get all model names
    model_names = collect(keys(analyses))
    
    # Compare each pair of models
    for i in 1:length(model_names)
        for j in (i+1):length(model_names)
            model1 = model_names[i]
            model2 = model_names[j]
            
            # Get analyses
            analysis1 = analyses[model1]
            analysis2 = analyses[model2]
            
            # Find common images
            common_images = intersect(keys(analysis1.per_image_metrics), keys(analysis2.per_image_metrics))
            
            if isempty(common_images)
                @warn "No common images between $model1 and $model2"
                continue
            end
            
            # Compare metrics
            for metric in config.metrics
                # Extract metric values for common images
                values1 = [analysis1.per_image_metrics[img][metric] for img in common_images]
                values2 = [analysis2.per_image_metrics[img][metric] for img in common_images]
                
                # Paired t-test
                t_test = OneSampleTTest(values1 - values2)
                p_value = pvalue(t_test)
                
                # Mean difference
                mean_diff = mean(values1) - mean(values2)
                
                # Effect size (Cohen's d)
                pooled_std = sqrt((std(values1)^2 + std(values2)^2) / 2)
                effect_size = abs(mean_diff) / pooled_std
                
                # Store results
                if !haskey(comparison_results, metric)
                    comparison_results[metric] = Dict{Tuple{String, String}, Dict{Symbol, Any}}()
                end
                
                comparison_results[metric][(model1, model2)] = Dict{Symbol, Any}(
                    :mean_diff => mean_diff,
                    :p_value => p_value,
                    :effect_size => effect_size,
                    :significant => p_value < 0.05,
                    :better_model => mean_diff > 0 ? model1 : model2,
                    :effect_magnitude => effect_size < 0.2 ? "small" : (effect_size < 0.5 ? "medium" : "large")
                )
            end
        end
    end
    
    return comparison_results
end

"""
Generate model ranking
"""
function rank_models(analyses::Dict{String, ModelAnalysis}, metric::Symbol)
    # Extract model names and metric values
    model_names = String[]
    metric_values = Float64[]
    
    for (model_name, analysis) in analyses
        if haskey(analysis.metrics, metric)
            push!(model_names, model_name)
            push!(metric_values, analysis.metrics[metric])
        end
    end
    
    # Sort by metric value (descending)
    sorted_indices = sortperm(metric_values, rev=true)
    
    # Create ranking
    ranking = Dict{String, Int}()
    for (rank, idx) in enumerate(sorted_indices)
        ranking[model_names[idx]] = rank
    end
    
    return ranking
end

"""
Generate visualizations for model comparison
"""
function visualize_results(analyses::Dict{String, ModelAnalysis}, comparison::Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}, config::AnalyzerConfig)
    # Create output directory if it doesn't exist
    mkpath(config.output_dir)
    
    # Extract model names
    model_names = collect(keys(analyses))
    
    # Bar chart for each metric
    for metric in config.metrics
        # Extract metric values and errors
        values = [analyses[model].metrics[metric] for model in model_names]
        errors = [analyses[model].std_metrics[metric] for model in model_names]
        
        # Create bar chart
        p = bar(model_names, values, 
                yerror=errors,
                title=string(metric),
                xlabel="Model", 
                ylabel=string(metric),
                legend=false,
                color=categorical_colors(length(model_names)),
                alpha=0.7)
        
        # Save figure
        metric_file = joinpath(config.output_dir, "$(metric)_comparison.png")
        savefig(p, metric_file)
    end
    
    # Radar chart comparing all models
    if !isempty(config.metrics)
        # Define angles for each metric
        n_metrics = length(config.metrics)
        angles = range(0, 2π, length=n_metrics+1)[1:end-1]
        
        # Create radar chart
        radar_plot = plot(proj=:polar, legend=:topright, size=(800, 600))
        
        for model_name in model_names
            analysis = analyses[model_name]
            
            # Extract normalized metric values
            values = Float64[]
            for metric in config.metrics
                # Normalize to [0, 1]
                max_val = maximum([analyses[m].metrics[metric] for m in model_names])
                if max_val > 0
                    push!(values, analysis.metrics[metric] / max_val)
                else
                    push!(values, 0.0)
                end
            end
            
            # Close the polygon
            angles_closed = [angles..., angles[1]]
            values_closed = [values..., values[1]]
            
            # Add to plot
            plot!(radar_plot, angles_closed, values_closed, label=model_name, linewidth=2, marker=:circle)
        end
        
        # Add metric labels
        for (i, metric) in enumerate(config.metrics)
            annotate!(radar_plot, [(angles[i], 1.1, string(metric))])
        end
        
        # Save figure
        radar_file = joinpath(config.output_dir, "radar_chart.png")
        savefig(radar_plot, radar_file)
    end
    
    # Statistical comparison heatmap
    if !isempty(comparison)
        for metric in keys(comparison)
            # Create labels for heatmap
            labels = model_names
            n = length(labels)
            
            # Create matrices for p-values and effect sizes
            p_values = ones(n, n)
            effect_sizes = zeros(n, n)
            
            # Fill matrices
            for i in 1:n
                for j in 1:n
                    if i != j
                        model1 = labels[i]
                        model2 = labels[j]
                        
                        # Check if comparison exists
                        if haskey(comparison[metric], (model1, model2))
                            p_values[i, j] = comparison[metric][(model1, model2)][:p_value]
                            effect_sizes[i, j] = comparison[metric][(model1, model2)][:effect_size]
                        elseif haskey(comparison[metric], (model2, model1))
                            p_values[i, j] = comparison[metric][(model2, model1)][:p_value]
                            effect_sizes[i, j] = comparison[metric][(model2, model1)][:effect_size]
                        end
                    end
                end
            end
            
            # Create heatmap for p-values
            p_heatmap = heatmap(p_values, 
                               title="P-values for $(metric)",
                               xlabel="Model", ylabel="Model",
                               xticks=(1:n, labels), yticks=(1:n, labels),
                               color=:viridis, clims=(0, 0.1))
            
            # Save figure
            p_file = joinpath(config.output_dir, "$(metric)_pvalues.png")
            savefig(p_heatmap, p_file)
            
            # Create heatmap for effect sizes
            es_heatmap = heatmap(effect_sizes, 
                                title="Effect Sizes for $(metric)",
                                xlabel="Model", ylabel="Model",
                                xticks=(1:n, labels), yticks=(1:n, labels),
                                color=:plasma)
            
            # Save figure
            es_file = joinpath(config.output_dir, "$(metric)_effectsizes.png")
            savefig(es_heatmap, es_file)
        end
    end
    
    # Box plots for metric distributions
    for metric in config.metrics
        # Extract per-image metrics for each model
        data = []
        for model_name in model_names
            analysis = analyses[model_name]
            values = [metrics[metric] for (_, metrics) in analysis.per_image_metrics]
            push!(data, values)
        end
        
        # Create box plot
        bp = boxplot(model_names, data, 
                    title="Distribution of $(metric)",
                    xlabel="Model", ylabel=string(metric),
                    legend=false,
                    color=categorical_colors(length(model_names)),
                    alpha=0.7,
                    outliers=true)
        
        # Save figure
        bp_file = joinpath(config.output_dir, "$(metric)_boxplot.png")
        savefig(bp, bp_file)
    end
    
    # Error analysis bar chart
    if !isempty(analyses)
        # Extract error rates for each model
        error_rates = [analysis.error_analysis[:error_rate] for (_, analysis) in analyses]
        
        # Create bar chart
        error_plot = bar(model_names, error_rates, 
                        title="Error Rate",
                        xlabel="Model", ylabel="Error Rate",
                        legend=false,
                        color=categorical_colors(length(model_names)),
                        alpha=0.7)
        
        # Save figure
        error_file = joinpath(config.output_dir, "error_rates.png")
        savefig(error_plot, error_file)
        
        # Create bar chart for error by point
        if haskey(first(values(analyses)).error_analysis, :error_by_point)
            error_by_point = hcat([analysis.error_analysis[:error_by_point] for (_, analysis) in analyses]...)
            
            # Create grouped bar chart
            point_plot = groupedbar(1:4, error_by_point, 
                                  title="Errors by Keypoint",
                                  xlabel="Keypoint", ylabel="Error Count",
                                  legend=:topleft,
                                  labels=permutedims(model_names),
                                  color=permutedims(categorical_colors(length(model_names))),
                                  alpha=0.7)
            
            # Save figure
            point_file = joinpath(config.output_dir, "errors_by_point.png")
            savefig(point_plot, point_file)
        end
    end
end

"""
Generate HTML report
"""
function generate_html_report(analyses::Dict{String, ModelAnalysis}, comparison::Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}, config::AnalyzerConfig)
    # Create output directory if it doesn't exist
    mkpath(config.output_dir)
    
    # Generate timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    
    # Extract model names
    model_names = collect(keys(analyses))
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Court Detection Model Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { background-color: #e6f7ff; font-weight: bold; }
            .container { display: flex; flex-wrap: wrap; }
            .chart { flex: 1; min-width: 500px; margin: 10px; }
            .footer { margin-top: 50px; color: #666; font-size: 0.8em; }
            .significant { color: #009900; font-weight: bold; }
            .not-significant { color: #990000; }
        </style>
    </head>
    <body>
        <h1>Court Detection Model Analysis</h1>
        <p>Generated on: $(timestamp)</p>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Model</th>
    """
    
    # Add headers for each metric
    for metric in config.metrics
        html *= "<th>$(metric)</th>"
    end
    
    html *= """
            </tr>
    """
    
    # Add rows for each model
    for model_name in model_names
        analysis = analyses[model_name]
        
        html *= """
            <tr>
                <td>$(model_name)</td>
        """
        
        for metric in config.metrics
            if haskey(analysis.metrics, metric)
                value = analysis.metrics[metric]
                ci_low, ci_high = analysis.ci_metrics[metric]
                
                # Format based on metric type
                if metric in [:kla_05, :kla_10, :cd_iou, :pcr]
                    value_str = "$(round(value * 100, digits=1))%"
                    ci_str = "($(round(ci_low * 100, digits=1))% - $(round(ci_high * 100, digits=1))%)"
                else
                    value_str = "$(round(value, digits=4))"
                    ci_str = "($(round(ci_low, digits=4)) - $(round(ci_high, digits=4)))"
                end
                
                html *= """
                    <td>$(value_str)<br><small>$(ci_str)</small></td>
                """
            else
                html *= """
                    <td>N/A</td>
                """
            end
        end
        
        html *= """
            </tr>
        """
    end
    
    html *= """
        </table>
        
        <h2>Rankings</h2>
        <table>
            <tr>
                <th>Model</th>
    """
    
    # Add headers for each metric
    for metric in config.metrics
        html *= "<th>$(metric)</th>"
    end
    
    # Add average rank
    html *= "<th>Average Rank</th>"
    
    html *= """
            </tr>
    """
    
    # Generate rankings for each metric
    rankings = Dict{Symbol, Dict{String, Int}}()
    for metric in config.metrics
        rankings[metric] = rank_models(analyses, metric)
    end
    
    # Add rows for each model
    for model_name in model_names
        html *= """
            <tr>
                <td>$(model_name)</td>
        """
        
        # Calculate average rank
        total_rank = 0
        num_metrics = 0
        
        for metric in config.metrics
            if haskey(rankings, metric) && haskey(rankings[metric], model_name)
                rank = rankings[metric][model_name]
                total_rank += rank
                num_metrics += 1
                
                # Highlight the best model
                if rank == 1
                    html *= """
                        <td class="highlight">$(rank)</td>
                    """
                else
                    html *= """
                        <td>$(rank)</td>
                    """
                end
            else
                html *= """
                    <td>N/A</td>
                """
            end
        end
        
        # Add average rank
        avg_rank = num_metrics > 0 ? total_rank / num_metrics : 0
        html *= """
            <td>$(round(avg_rank, digits=2))</td>
        """
        
        html *= """
            </tr>
        """
    end
    
    html *= """
        </table>
        
        <h2>Statistical Comparisons</h2>
    """
    
    # Add statistical comparison tables for each metric
    for metric in config.metrics
        if haskey(comparison, metric)
            html *= """
                <h3>$(metric) Comparisons</h3>
                <table>
                    <tr>
                        <th>Model A</th>
                        <th>Model B</th>
                        <th>Difference (A-B)</th>
                        <th>P-value</th>
                        <th>Significance</th>
                        <th>Effect Size</th>
                        <th>Better Model</th>
                    </tr>
            """
            
            # Add rows for each comparison
            for ((model1, model2), result) in comparison[metric]
                mean_diff = result[:mean_diff]
                p_value = result[:p_value]
                effect_size = result[:effect_size]
                significant = result[:significant]
                better_model = result[:better_model]
                effect_magnitude = result[:effect_magnitude]
                
                # Format mean difference
                if metric in [:kla_05, :kla_10, :cd_iou, :pcr]
                    mean_diff_str = "$(round(mean_diff * 100, digits=1))%"
                else
                    mean_diff_str = "$(round(mean_diff, digits=4))"
                end
                
                # Format p-value
                if p_value < 0.001
                    p_value_str = "p < 0.001"
                else
                    p_value_str = "p = $(round(p_value, digits=3))"
                end
                
                # Format significance
                sig_class = significant ? "significant" : "not-significant"
                sig_str = significant ? "Significant" : "Not significant"
                
                html *= """
                    <tr>
                        <td>$(model1)</td>
                        <td>$(model2)</td>
                        <td>$(mean_diff_str)</td>
                        <td>$(p_value_str)</td>
                        <td class="$(sig_class)">$(sig_str)</td>
                        <td>$(round(effect_size, digits=2)) ($(effect_magnitude))</td>
                        <td>$(better_model)</td>
                    </tr>
                """
            end
            
            html *= """
                </table>
            """
        end
    end
    
    html *= """
        <h2>Visualizations</h2>
        <div class="container">
    """
    
    # Add visualizations
    for metric in config.metrics
        html *= """
            <div class="chart">
                <img src="$(metric)_comparison.png" alt="$(metric) Comparison" width="100%">
            </div>
        """
    end
    
    html *= """
        </div>
        
        <div class="container">
            <div class="chart">
                <img src="radar_chart.png" alt="Radar Chart" width="100%">
            </div>
        """
    
    # Add box plots
    for metric in config.metrics
        html *= """
            <div class="chart">
                <img src="$(metric)_boxplot.png" alt="$(metric) Distribution" width="100%">
            </div>
        """
    end
    
    html *= """
        </div>
        
        <h2>Error Analysis</h2>
        <div class="container">
            <div class="chart">
                <img src="error_rates.png" alt="Error Rates" width="100%">
            </div>
            <div class="chart">
                <img src="errors_by_point.png" alt="Errors by Keypoint" width="100%">
            </div>
        </div>
        
        <div class="footer">
            <p>Court Detection Model Analysis - Generated by CourtAnalyzer.jl</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    open(joinpath(config.output_dir, "report.html"), "w") do io
        write(io, html)
    end
    
    # Return path to report
    return joinpath(config.output_dir, "report.html")
end

"""
Generate Excel report
"""
function generate_excel_report(analyses::Dict{String, ModelAnalysis}, comparison::Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}, config::AnalyzerConfig)
    # Placeholder - would use ExcelReaders.jl or similar in a real implementation
    @warn "Excel report generation not implemented in this version"
    return
end

"""
Generate JSON report
"""
function generate_json_report(analyses::Dict{String, ModelAnalysis}, comparison::Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}, config::AnalyzerConfig)
    # Create output directory if it doesn't exist
    mkpath(config.output_dir)
    
    # Convert analyses to simpler structure for JSON
    json_analyses = Dict{String, Dict{String, Any}}()
    for (model_name, analysis) in analyses
        json_analyses[model_name] = Dict{String, Any}(
            "metrics" => Dict{String, Float64}(string(k) => v for (k, v) in analysis.metrics),
            "std_metrics" => Dict{String, Float64}(string(k) => v for (k, v) in analysis.std_metrics),
            "ci_metrics" => Dict{String, Any}(string(k) => Dict("low" => v[1], "high" => v[2]) for (k, v) in analysis.ci_metrics),
            "success_rate" => Dict{String, Float64}(string(k) => v for (k, v) in analysis.success_rate),
            "error_analysis" => Dict{String, Any}(string(k) => v for (k, v) in analysis.error_analysis if typeof(v) in [Int, Float64, String, Bool, Nothing])
        )
    end
    
    # Convert comparison to simpler structure for JSON
    json_comparison = Dict{String, Dict{String, Any}}()
    for (metric, comparisons) in comparison
        json_comparison[string(metric)] = Dict{String, Any}()
        for ((model1, model2), result) in comparisons
            key = "$(model1) vs $(model2)"
            json_comparison[string(metric)][key] = Dict{String, Any}(
                string(k) => v for (k, v) in result if typeof(v) in [Int, Float64, String, Bool, Nothing]
            )
        end
    end
    
    # Create final JSON structure
    json_data = Dict{String, Any}(
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "analyses" => json_analyses,
        "comparison" => json_comparison
    )
    
    # Write JSON to file
    open(joinpath(config.output_dir, "report.json"), "w") do io
        JSON.print(io, json_data, 4)
    end
    
    # Return path to report
    return joinpath(config.output_dir, "report.json")
end

"""
Generate comprehensive report
"""
function generate_report(analyses::Dict{String, ModelAnalysis}, comparison::Dict{Symbol, Dict{Tuple{String, String}, Dict{Symbol, Any}}}, config::AnalyzerConfig)
    # Create visualizations
    visualize_results(analyses, comparison, config)
    
    # Generate HTML report
    html_path = generate_html_report(analyses, comparison, config)
    
    # Generate JSON report
    json_path = generate_json_report(analyses, comparison, config)
    
    # Try to open the HTML report in a browser
    try
        if Sys.iswindows()
            run(`cmd /c start $(html_path)`)
        elseif Sys.isapple()
            run(`open $(html_path)`)
        elseif Sys.islinux()
            run(`xdg-open $(html_path)`)
        end
    catch
        @info "Report generated at: $(html_path)"
    end
    
    return html_path
end

"""
Generate colors for categorical data
"""
function categorical_colors(n::Int)
    # Use a good categorical color scheme
    if n <= 8
        # Use ColorSchemes.tableau_10 for up to 8 categories
        return [ColorSchemes.tableau_10[i] for i in range(1, stop=10, length=n)]
    else
        # For more categories, use random colors with enough separation
        colors = distinguishable_colors(n, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
        return colors
    end
end

"""
Analyze specific metrics with detailed explanations
"""
function analyze_metrics(analyses::Dict{String, ModelAnalysis}, config::AnalyzerConfig)
    # Create output directory if it doesn't exist
    mkpath(config.output_dir)
    
    # Open output file
    open(joinpath(config.output_dir, "metrics_analysis.txt"), "w") do io
        write(io, "Court Detection Model Metrics Analysis\n")
        write(io, "======================================\n\n")
        write(io, "Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
        
        # Analyze KLA@0.05
        if :kla_05 in config.metrics
            write(io, "Keypoint Localization Accuracy (KLA@0.05)\n")
            write(io, "----------------------------------------\n")
            write(io, "This metric measures the proportion of keypoints correctly localized within 5% of the image diagonal.\n\n")
            
            # Report best model
            best_model = ""
            best_value = 0.0
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :kla_05) && analysis.metrics[:kla_05] > best_value
                    best_value = analysis.metrics[:kla_05]
                    best_model = model_name
                end
            end
            
            write(io, "Best performing model: $(best_model) ($(round(best_value * 100, digits=1))%)\n\n")
            
            # Report all models
            write(io, "All models:\n")
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :kla_05)
                    value = analysis.metrics[:kla_05]
                    ci_low, ci_high = analysis.ci_metrics[:kla_05]
                    write(io, "- $(model_name): $(round(value * 100, digits=1))% ($(round(ci_low * 100, digits=1))% - $(round(ci_high * 100, digits=1))%)\n")
                end
            end
            
            write(io, "\nInterpretation:\n")
            if best_value >= 0.9
                write(io, "Excellent performance. Most keypoints are accurately localized.\n")
            elseif best_value >= 0.8
                write(io, "Good performance. The majority of keypoints are accurately localized.\n")
            elseif best_value >= 0.7
                write(io, "Acceptable performance. Many keypoints are accurately localized, but there is room for improvement.\n")
            else
                write(io, "Poor performance. Significant improvements are needed.\n")
            end
            write(io, "\n\n")
        end
        
        # Analyze CD-IoU
        if :cd_iou in config.metrics
            write(io, "Court Detection IoU\n")
            write(io, "-----------------\n")
            write(io, "This metric measures the Intersection over Union between predicted and ground truth court boundaries.\n\n")
            
            # Report best model
            best_model = ""
            best_value = 0.0
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :cd_iou) && analysis.metrics[:cd_iou] > best_value
                    best_value = analysis.metrics[:cd_iou]
                    best_model = model_name
                end
            end
            
            write(io, "Best performing model: $(best_model) ($(round(best_value * 100, digits=1))%)\n\n")
            
            # Report all models
            write(io, "All models:\n")
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :cd_iou)
                    value = analysis.metrics[:cd_iou]
                    ci_low, ci_high = analysis.ci_metrics[:cd_iou]
                    write(io, "- $(model_name): $(round(value * 100, digits=1))% ($(round(ci_low * 100, digits=1))% - $(round(ci_high * 100, digits=1))%)\n")
                end
            end
            
            write(io, "\nInterpretation:\n")
            if best_value >= 0.9
                write(io, "Excellent performance. The predicted court boundaries closely match the ground truth.\n")
            elseif best_value >= 0.8
                write(io, "Good performance. The predicted court boundaries match the ground truth well.\n")
            elseif best_value >= 0.7
                write(io, "Acceptable performance. The predicted court boundaries are reasonable, but there is room for improvement.\n")
            else
                write(io, "Poor performance. Significant improvements are needed.\n")
            end
            write(io, "\n\n")
        end
        
        # Analyze PCR
        if :pcr in config.metrics
            write(io, "Perfect Court Rate (PCR)\n")
            write(io, "----------------------\n")
            write(io, "This metric measures the proportion of images where all four court corners are correctly localized.\n\n")
            
            # Report best model
            best_model = ""
            best_value = 0.0
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :pcr) && analysis.metrics[:pcr] > best_value
                    best_value = analysis.metrics[:pcr]
                    best_model = model_name
                end
            end
            
            write(io, "Best performing model: $(best_model) ($(round(best_value * 100, digits=1))%)\n\n")
            
            # Report all models
            write(io, "All models:\n")
            for (model_name, analysis) in analyses
                if haskey(analysis.metrics, :pcr)
                    value = analysis.metrics[:pcr]
                    ci_low, ci_high = analysis.ci_metrics[:pcr]
                    write(io, "- $(model_name): $(round(value * 100, digits=1))% ($(round(ci_low * 100, digits=1))% - $(round(ci_high * 100, digits=1))%)\n")
                end
            end
            
            write(io, "\nInterpretation:\n")
            if best_value >= 0.8
                write(io, "Excellent performance. Most courts are perfectly detected.\n")
            elseif best_value >= 0.6
                write(io, "Good performance. The majority of courts are perfectly detected.\n")
            elseif best_value >= 0.4
                write(io, "Acceptable performance. Many courts are perfectly detected, but there is room for improvement.\n")
            else
                write(io, "Poor performance. Significant improvements are needed.\n")
            end
            write(io, "\n\n")
        end
        
        # Error analysis
        write(io, "Error Analysis\n")
        write(io, "-------------\n")
        write(io, "This section analyzes the errors made by each model.\n\n")
        
        for (model_name, analysis) in analyses
            write(io, "$(model_name):\n")
            
            if haskey(analysis.error_analysis, :error_rate)
                write(io, "- Error rate: $(round(analysis.error_analysis[:error_rate] * 100, digits=1))%\n")
            end
            
            if haskey(analysis.error_analysis, :total_errors)
                write(io, "- Total errors: $(analysis.error_analysis[:total_errors])\n")
            end
            
            if haskey(analysis.error_analysis, :error_by_point)
                write(io, "- Errors by keypoint:\n")
                for i in 1:4
                    write(io, "  - Keypoint $(i): $(analysis.error_analysis[:error_by_point][i]) errors\n")
                end
            end
            
            write(io, "\n")
        end
        
        # Overall recommendations
        write(io, "Overall Recommendations\n")
        write(io, "----------------------\n")
        
        # Determine best model overall
        model_scores = Dict{String, Float64}()
        for (model_name, analysis) in analyses
            score = 0.0
            num_metrics = 0
            
            for metric in [:kla_05, :cd_iou, :pcr]
                if haskey(analysis.metrics, metric)
                    score += analysis.metrics[metric]
                    num_metrics += 1
                end
            end
            
            if num_metrics > 0
                model_scores[model_name] = score / num_metrics
            end
        end
        
        # Sort models by score
        sorted_models = sort(collect(model_scores), by=x -> x[2], rev=true)
        
        if !isempty(sorted_models)
            best_model = sorted_models[1][1]
            best_score = sorted_models[1][2]
            
            write(io, "Based on the analysis, the best overall model is $(best_model) with an average performance of $(round(best_score * 100, digits=1))%.\n\n")
            
            # Provide specific recommendations
            write(io, "Recommendations:\n")
            if haskey(analyses, best_model) && haskey(analyses[best_model].error_analysis, :error_by_point)
                error_by_point = analyses[best_model].error_analysis[:error_by_point]
                
                # Find most problematic keypoint
                max_errors, max_idx = findmax(error_by_point)
                
                if max_errors > 0
                    write(io, "1. The model $(best_model) has the most difficulty with keypoint $(max_idx). Consider improving the detection of this particular corner.\n")
                end
            end
            
            write(io, "2. For production use, $(best_model) is recommended based on its superior performance.\n")
            
            # Compare with second best if available
            if length(sorted_models) > 1
                second_best = sorted_models[2][1]
                second_score = sorted_models[2][2]
                
                improvement = (best_score - second_score) / second_score * 100
                
                write(io, "3. $(best_model) outperforms $(second_best) by $(round(improvement, digits=1))% on average.\n")
            end
        else
            write(io, "Insufficient data to make specific recommendations.\n")
        end
    end
    
    return joinpath(config.output_dir, "metrics_analysis.txt")
end

"""
Main function to run all analyses
"""
function run_analysis(config::AnalyzerConfig)
    # Analyze models
    analyses = analyze_models(config)
    
    if isempty(analyses)
        error("No valid models analyzed")
    end
    
    # Compare models
    comparison = compare_models(analyses, config)
    
    # Generate visualizations and report
    generate_report(analyses, comparison, config)
    
    # Analyze metrics with detailed explanations
    analyze_metrics(analyses, config)
    
    println("Analysis completed. Results saved to: $(config.output_dir)")
    
    return analyses, comparison
end

# Command-line interface
function main()
    # Parse command-line arguments
    if length(ARGS) < 2
        println("Usage: julia court_analyzer.jl <predictions_dir> <ground_truth_path> [output_dir]")
        exit(1)
    end
    
    predictions_dir = ARGS[1]
    ground_truth_path = ARGS[2]
    output_dir = length(ARGS) > 2 ? ARGS[3] : "analysis_results"
    
    # Create config
    config = AnalyzerConfig(
        predictions_dir = predictions_dir,
        ground_truth_path = ground_truth_path,
        output_dir = output_dir
    )
    
    # Run analysis
    run_analysis(config)
end

# Run main function if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end 