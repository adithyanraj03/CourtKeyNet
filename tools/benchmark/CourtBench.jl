module CourtBench

using BSON
using CSV
using DataFrames
using Statistics
using Flux
using Flux: gpu, cpu
using Images
using ImageIO
using FileIO
using ONNX
using ONNXRunTime
using JSON
using Plots
using ProgressMeter
using LinearAlgebra
using Dates
using Random
using Test
using Zygote

export run_benchmark, load_model, compute_metrics, generate_report

"""
Configuration for the benchmark tool
"""
mutable struct BenchmarkConfig
    dataset_path::String
    results_path::String
    models_to_test::Vector{String}
    metrics::Vector{Symbol}
    batch_size::Int64
    num_workers::Int64
    image_size::Tuple{Int64, Int64}
    device::Symbol  # :cpu or :gpu
    
    function BenchmarkConfig(;
        dataset_path::String = "datasets/badminton",
        results_path::String = "results",
        models_to_test::Vector{String} = ["courtkeynet", "yolov8", "hough"],
        metrics::Vector{Symbol} = [:kla_05, :kla_10, :cd_iou, :pcr, :inference_time],
        batch_size::Int64 = 32,
        num_workers::Int64 = 4,
        image_size::Tuple{Int64, Int64} = (640, 640),
        device::Symbol = :cpu
    )
        return new(dataset_path, results_path, models_to_test, metrics, batch_size, num_workers, image_size, device)
    end
end

"""
Result of a single prediction
"""
struct PredictionResult
    keypoints::Matrix{Float64}  # 4×2 matrix of keypoints
    confidence::Float64
    inference_time::Float64
end

"""
Model interface to standardize different model implementations
"""
abstract type AbstractCourtModel end

"""
CourtKeyNet model implementation
"""
mutable struct CourtKeyNetModel <: AbstractCourtModel
    model_path::String
    model::Any
    device::Symbol
    
    function CourtKeyNetModel(model_path::String, device::Symbol)
        # Load the model based on format
        if endswith(model_path, ".onnx")
            model = ONNXRunTime.InferenceSession(model_path)
        elseif endswith(model_path, ".bson")
            model = BSON.load(model_path)[:model]
            if device == :gpu && CUDA.functional()
                model = gpu(model)
            end
        else
            error("Unsupported model format: $(model_path)")
        end
        
        return new(model_path, model, device)
    end
end

"""
YOLOv8 model implementation
"""
mutable struct YOLOv8Model <: AbstractCourtModel
    model_path::String
    model::Any
    device::Symbol
    
    function YOLOv8Model(model_path::String, device::Symbol)
        # Similar implementation for YOLOv8
        # ...
        return new(model_path, nothing, device)
    end
end



"""
Factory function to create model instance based on model name
"""
function load_model(model_name::String, config::BenchmarkConfig)
    # Define paths to model files for each model type
    model_paths = Dict(
        "courtkeynet" => "models/courtkeynet.onnx",
        "yolov8" => "models/yolov8-pose.onnx",
        "frcnn" => "models/faster-rcnn-keypoints.onnx",
        "hough" => "models/hough-transform.jl" 
    )
    
    # Check if model exists
    if !haskey(model_paths, lowercase(model_name))
        error("Model not supported: $(model_name)")
    end
    
    model_path = model_paths[lowercase(model_name)]
    
    # Create appropriate model instance based on model type
    if lowercase(model_name) == "courtkeynet"
        return CourtKeyNetModel(model_path, config.device)
    elseif lowercase(model_name) == "yolov8"
        return YOLOv8Model(model_path, config.device)
    elseif lowercase(model_name) == "frcnn"
        return FRCNNModel(model_path, config.device)
    elseif lowercase(model_name) == "hough"
        # Special case for Hough transform
        return HoughTransformModel(model_path, config.device)
    else
        error("Model implementation not found: $(model_name)")
    end
end

"""
Dataset for court keypoint detection
"""
struct CourtDataset
    image_paths::Vector{String}
    annotations::Vector{Dict{String, Any}}
    transform::Function
    
    function CourtDataset(dataset_path::String, split::String, transform::Function)
        # Load dataset
        images_dir = joinpath(dataset_path, split, "images")
        labels_dir = joinpath(dataset_path, split, "labels")
        
        image_paths = String[]
        annotations = Dict{String, Any}[]
        
        # List image files
        for file in readdir(images_dir)
            if endswith(file, ".jpg") || endswith(file, ".png") || endswith(file, ".jpeg")
                image_path = joinpath(images_dir, file)
                push!(image_paths, image_path)
                
                # Find corresponding label file
                label_file = replace(file, r"\.[^.]+$" => ".txt")
                label_path = joinpath(labels_dir, label_file)
                
                if isfile(label_path)
                    # Read and parse label
                    lines = readlines(label_path)
                    if length(lines) > 0
                        parts = split(lines[1], " ")
                        if length(parts) >= 13  # class_id + bbox(4) + keypoints(4*2)
                            class_id = parse(Int, parts[1])
                            
                            # Bounding box
                            cx = parse(Float64, parts[2])
                            cy = parse(Float64, parts[3])
                            width = parse(Float64, parts[4])
                            height = parse(Float64, parts[5])
                            
                            # Keypoints
                            keypoints = zeros(Float64, 4, 3)  # 4 keypoints with x, y, visibility
                            for i in 1:4
                                idx = 6 + (i-1)*3
                                if idx+2 <= length(parts)
                                    keypoints[i, 1] = parse(Float64, parts[idx])
                                    keypoints[i, 2] = parse(Float64, parts[idx+1])
                                    keypoints[i, 3] = parse(Float64, parts[idx+2])
                                end
                            end
                            
                            annotation = Dict(
                                "class_id" => class_id,
                                "bbox" => [cx, cy, width, height],
                                "keypoints" => keypoints
                            )
                            push!(annotations, annotation)
                        end
                    end
                else
                    # No annotation file, create empty annotation
                    push!(annotations, Dict("class_id" => -1, "bbox" => [0.0, 0.0, 0.0, 0.0], "keypoints" => zeros(Float64, 4, 3)))
                end
            end
        end
        
        return new(image_paths, annotations, transform)
    end
end

"""
Create a DataLoader for batch processing
"""
function create_dataloader(dataset::CourtDataset, batch_size::Int64, shuffle::Bool=false)
    # Create and return a simple data iterator
    n = length(dataset.image_paths)
    indices = collect(1:n)
    
    if shuffle
        Random.shuffle!(indices)
    end
    
    function data_iterator()
        for i in 1:batch_size:n
            batch_indices = indices[i:min(i+batch_size-1, n)]
            batch_images = []
            batch_annotations = []
            
            for idx in batch_indices
                # Load and preprocess image
                img = load(dataset.image_paths[idx])
                img = dataset.transform(img)
                push!(batch_images, img)
                push!(batch_annotations, dataset.annotations[idx])
            end
            
            # Convert to batch
            x = cat(batch_images..., dims=4)  # Shape: H×W×C×B
            x = permutedims(x, [3, 1, 2, 4])  # Shape: C×H×W×B
            
            produce(x, batch_annotations)
        end
    end
    
    return Task(data_iterator)
end

"""
Preprocess image function
"""
function preprocess_image(img, size::Tuple{Int64, Int64})
    # Resize image
    img = imresize(img, size)
    
    # Convert to RGB if needed
    if eltype(img) <: Gray
        img = RGB.(img)
    end
    
    # Normalize
    img = Float32.(channelview(img))
    img = permutedims(img, (2, 3, 1))  # CHW to HWC
    
    # Normalize to [0, 1]
    img = img ./ 255.0
    
    # Apply mean and std normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for c in 1:3
        img[:, :, c] = (img[:, :, c] .- mean[c]) ./ std[c]
    end
    
    return img
end

"""
Run inference on a model
"""
function run_inference(model::AbstractCourtModel, image::Array{Float32})
    # Measure inference time
    start_time = time()
    
    # Run inference based on model type
    if isa(model, CourtKeyNetModel)
        # CourtKeyNet inference
        if endswith(model.model_path, ".onnx")
            # ONNX model inference
            input_name = ONNXRunTime.get_input_names(model.model)[1]
            output_name = ONNXRunTime.get_output_names(model.model)[1]
            
            # Prepare input: expand dims for batch
            input = reshape(image, (size(image)..., 1))
            
            # Run inference
            outputs = ONNXRunTime.run(model.model, Dict(input_name => input))
            keypoints = outputs[output_name][:,:,1]  # First batch item
        else
            # BSON model inference
            input = reshape(image, (size(image)..., 1))
            if model.device == :gpu
                input = gpu(input)
            end
            
            output = model.model(input)
            
            if model.device == :gpu
                output = cpu(output)
            end
            
            # Extract keypoints from model output
            keypoints = output["keypoints"][1]  # First batch item
        end
    elseif isa(model, YOLOv8Model)
        # YOLOv8 inference implementation
        # ...
        keypoints = zeros(Float64, 4, 2)  # Placeholder
    elseif isa(model, HoughTransformModel)
        # Hough Transform implementation (non-neural approach)
        # ...
        keypoints = zeros(Float64, 4, 2)  # Placeholder
    else
        # Default implementation for other models
        keypoints = zeros(Float64, 4, 2)  # Placeholder
    end
    
    end_time = time()
    inference_time = end_time - start_time
    
    # Create result with confidence score of 1.0 (placeholder)
    return PredictionResult(keypoints, 1.0, inference_time)
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
    # Function to calculate polygon area
    function polygon_area(vertices)
        n = size(vertices, 1)
        area = 0.0
        
        for i in 1:n
            j = i % n + 1
            area += vertices[i, 1] * vertices[j, 2]
            area -= vertices[j, 1] * vertices[i, 2]
        end
        
        return abs(area) / 2.0
    end
    
    # Function to check if point is inside polygon
    function point_in_polygon(point, vertices)
        n = size(vertices, 1)
        inside = false
        
        j = n
        for i in 1:n
            if ((vertices[i, 2] > point[2]) != (vertices[j, 2] > point[2])) &&
               (point[1] < vertices[i, 1] + (vertices[j, 1] - vertices[i, 1]) * 
                (point[2] - vertices[i, 2]) / (vertices[j, 2] - vertices[i, 2]))
                inside = !inside
            end
            j = i
        end
        
        return inside
    end
    
    # Function to calculate intersection of two polygons
    function polygon_intersection_area(poly1, poly2)
        # Simple implementation using a grid-based approach
        min_x = min(minimum(poly1[:, 1]), minimum(poly2[:, 1]))
        max_x = max(maximum(poly1[:, 1]), maximum(poly2[:, 1]))
        min_y = min(minimum(poly1[:, 2]), minimum(poly2[:, 2]))
        max_y = max(maximum(poly1[:, 2]), maximum(poly2[:, 2]))
        
        grid_size = 100
        step_x = (max_x - min_x) / grid_size
        step_y = (max_y - min_y) / grid_size
        
        intersection_count = 0
        total_points = 0
        
        for i in 0:grid_size
            for j in 0:grid_size
                x = min_x + i * step_x
                y = min_y + j * step_y
                point = [x, y]
                
                if point_in_polygon(point, poly1) && point_in_polygon(point, poly2)
                    intersection_count += 1
                end
                
                if point_in_polygon(point, poly1) || point_in_polygon(point, poly2)
                    total_points += 1
                end
            end
        end
        
        # Estimate IoU from point counts
        return total_points > 0 ? intersection_count / total_points : 0.0
    end
    
    # Extract visible keypoints
    pred_visible = pred_keypoints[:, 1:2]
    gt_visible = gt_keypoints[gt_keypoints[:, 3] .> 0, 1:2]
    
    # Ensure we have enough points to form polygons
    if size(gt_visible, 1) < 3
        return 0.0
    end
    
    # Calculate IoU
    return polygon_intersection_area(pred_visible, gt_visible)
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
Compute all metrics for a prediction
"""
function compute_metrics(pred::PredictionResult, gt::Dict{String, Any})
    gt_keypoints = gt["keypoints"]
    
    # Compute metrics
    kla_05 = compute_kla(pred.keypoints, gt_keypoints, 0.05)
    kla_10 = compute_kla(pred.keypoints, gt_keypoints, 0.10)
    cd_iou = compute_court_iou(pred.keypoints, gt_keypoints)
    pcr = compute_pcr(pred.keypoints, gt_keypoints, 0.05)
    
    return Dict(
        :kla_05 => kla_05,
        :kla_10 => kla_10,
        :cd_iou => cd_iou,
        :pcr => pcr,
        :inference_time => pred.inference_time
    )
end

"""
Generate plots for benchmark results
"""
function generate_plots(results::DataFrame, output_dir::String)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Create KLA plot
    p1 = bar(results.model, results.kla_05, 
             title="Keypoint Localization Accuracy (KLA@0.05)",
             xlabel="Model", ylabel="Accuracy (%)",
             legend=false, bar_width=0.7,
             formatter=:percent, xtickfontsize=8,
             xticks=:all, xrotation=45,
             color=:blue, alpha=0.8)
    
    # Create CD-IoU plot
    p2 = bar(results.model, results.cd_iou, 
             title="Court Detection IoU",
             xlabel="Model", ylabel="IoU",
             legend=false, bar_width=0.7,
             formatter=:percent, xtickfontsize=8,
             xticks=:all, xrotation=45,
             color=:green, alpha=0.8)
    
    # Create PCR plot
    p3 = bar(results.model, results.pcr, 
             title="Perfect Court Rate",
             xlabel="Model", ylabel="PCR",
             legend=false, bar_width=0.7,
             formatter=:percent, xtickfontsize=8,
             xticks=:all, xrotation=45,
             color=:purple, alpha=0.8)
    
    # Create inference time plot
    p4 = bar(results.model, results.inference_time, 
             title="Inference Time",
             xlabel="Model", ylabel="Time (s)",
             legend=false, bar_width=0.7,
             xtickfontsize=8, xticks=:all, xrotation=45,
             color=:red, alpha=0.8)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
    savefig(combined_plot, joinpath(output_dir, "benchmark_results.png"))
    
    # Create radar chart for KLA, CD-IoU, and PCR
    theta = [0, 2π/3, 4π/3]
    labels = ["KLA@0.05", "CD-IoU", "PCR"]
    
    rp = plot(proj=:polar, legend=:topright, size=(600, 600))
    
    for (i, row) in enumerate(eachrow(results))
        r = [row.kla_05, row.cd_iou, row.pcr]
        # Close the polygon
        r = [r..., r[1]]
        t = [theta..., theta[1]]
        plot!(rp, t, r, label=row.model, linewidth=2, marker=:circle)
    end
    
    savefig(rp, joinpath(output_dir, "radar_chart.png"))
end

"""
Generate HTML report for benchmark results
"""
function generate_html_report(results::DataFrame, config::BenchmarkConfig, output_dir::String)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Generate timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    
    # Create HTML report
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CourtKeyNet Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { background-color: #e6f7ff; font-weight: bold; }
            .container { display: flex; flex-wrap: wrap; }
            .chart { flex: 1; min-width: 400px; margin: 10px; }
            .footer { margin-top: 50px; color: #666; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <h1>CourtKeyNet Benchmark Results</h1>
        <p>Generated on: $(timestamp)</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>KLA@0.05</th>
                <th>KLA@0.10</th>
                <th>CD-IoU</th>
                <th>PCR</th>
                <th>Inference Time (s)</th>
            </tr>
    """
    
    # Add rows for each model
    for (i, row) in enumerate(eachrow(results))
        # Determine if this row should be highlighted (highest values)
        highlight = row.kla_05 == maximum(results.kla_05) ? " class='highlight'" : ""
        
        html *= """
            <tr$(highlight)>
                <td>$(row.model)</td>
                <td>$(round(row.kla_05 * 100, digits=1))%</td>
                <td>$(round(row.kla_10 * 100, digits=1))%</td>
                <td>$(round(row.cd_iou * 100, digits=1))%</td>
                <td>$(round(row.pcr * 100, digits=1))%</td>
                <td>$(round(row.inference_time, digits=4))</td>
            </tr>
        """
    end
    
    html *= """
        </table>
        
        <h2>Visualization</h2>
        <div class="container">
            <div class="chart">
                <img src="benchmark_results.png" alt="Benchmark Results" width="100%">
            </div>
            <div class="chart">
                <img src="radar_chart.png" alt="Radar Chart" width="100%">
            </div>
        </div>
        
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Dataset Path</td><td>$(config.dataset_path)</td></tr>
            <tr><td>Image Size</td><td>$(config.image_size)</td></tr>
            <tr><td>Batch Size</td><td>$(config.batch_size)</td></tr>
            <tr><td>Device</td><td>$(config.device)</td></tr>
        </table>
        
        <div class="footer">
            <p>CourtKeyNet Benchmark Tool - Generated by CourtBench.jl</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    open(joinpath(output_dir, "report.html"), "w") do io
        write(io, html)
    end
end

"""
Generate comprehensive report for benchmark results
"""
function generate_report(results::DataFrame, config::BenchmarkConfig)
    output_dir = config.results_path
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Save raw results as CSV
    CSV.write(joinpath(output_dir, "results.csv"), results)
    
    # Generate plots
    generate_plots(results, output_dir)
    
    # Generate HTML report
    generate_html_report(results, config, output_dir)
    
    # Generate JSON results for API consumption
    open(joinpath(output_dir, "results.json"), "w") do io
        JSON.print(io, Dict(
            "timestamp" => string(now()),
            "config" => Dict(
                "dataset_path" => config.dataset_path,
                "image_size" => config.image_size,
                "batch_size" => config.batch_size,
                "device" => string(config.device)
            ),
            "results" => [
                Dict(
                    "model" => row.model,
                    "kla_05" => row.kla_05,
                    "kla_10" => row.kla_10,
                    "cd_iou" => row.cd_iou,
                    "pcr" => row.pcr,
                    "inference_time" => row.inference_time
                ) for row in eachrow(results)
            ]
        ), 4)  # Pretty-print with 4 spaces
    end
    
    return output_dir
end

"""
Run benchmark on all models
"""
function run_benchmark(config::BenchmarkConfig = BenchmarkConfig())
    # Create results dataframe
    results = DataFrame(
        model = String[],
        kla_05 = Float64[],
        kla_10 = Float64[],
        cd_iou = Float64[],
        pcr = Float64[],
        inference_time = Float64[]
    )
    
    # Create transform function
    transform = img -> preprocess_image(img, config.image_size)
    
    # Create dataset
    dataset = CourtDataset(config.dataset_path, "test", transform)
    
    # Create dataloader
    dataloader = create_dataloader(dataset, config.batch_size, false)
    
    # Run benchmark for each model
    for model_name in config.models_to_test
        println("Benchmarking model: $(model_name)")
        
        try
            # Load model
            model = load_model(model_name, config)
            
            # Initialize metrics
            metric_sums = Dict(metric => 0.0 for metric in config.metrics)
            total_samples = 0
            
            # Run inference on all samples
            @showprogress for (batch_images, batch_annotations) in dataloader
                # Process each image in the batch
                for i in 1:size(batch_images, 4)
                    # Get image and ground truth
                    image = batch_images[:, :, :, i]
                    annotation = batch_annotations[i]
                    
                    # Run inference
                    pred = run_inference(model, image)
                    
                    # Compute metrics
                    metrics = compute_metrics(pred, annotation)
                    
                    # Accumulate metrics
                    for metric in config.metrics
                        metric_sums[metric] += metrics[metric]
                    end
                    
                    total_samples += 1
                end
            end
            
            # Compute average metrics
            avg_metrics = Dict(metric => metric_sums[metric] / total_samples for metric in config.metrics)
            
            # Add to results
            push!(results, (
                model = model_name,
                kla_05 = avg_metrics[:kla_05],
                kla_10 = avg_metrics[:kla_10],
                cd_iou = avg_metrics[:cd_iou],
                pcr = avg_metrics[:pcr],
                inference_time = avg_metrics[:inference_time]
            ))
            
            println("Finished benchmarking $(model_name)")
            
        catch e
            println("Error benchmarking $(model_name): $(e)")
        end
    end
    
    # Generate report
    report_path = generate_report(results, config)
    
    println("Benchmark completed. Results saved to: $(report_path)")
    
    return results
end

end # module