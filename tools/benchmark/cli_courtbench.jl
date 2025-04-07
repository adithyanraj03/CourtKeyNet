#!/usr/bin/env julia

using ArgParse
using CourtBench
using DataFrames
using CSV
using Dates

"""
Parse command-line arguments
"""
function parse_commandline()
    s = ArgParseSettings(
        description="CourtKeyNet Benchmark Tool",
        version="1.0.0",
        add_version=true
    )
    
    @add_arg_table s begin
        "--dataset", "-d"
            help = "Path to dataset directory"
            arg_type = String
            default = "datasets/badminton"
        
        "--results", "-r"
            help = "Path to results directory"
            arg_type = String
            default = "results"
        
        "--models", "-m"
            help = "Models to test (comma-separated)"
            arg_type = String
            default = "courtkeynet,yolov8,hough"
        
        "--batch-size", "-b"
            help = "Batch size"
            arg_type = Int
            default = 32
        
        "--img-size", "-i"
            help = "Image size (WxH)"
            arg_type = String
            default = "640x640"
        
        "--device"
            help = "Device to run inference on (cpu or gpu)"
            arg_type = String
            default = "cpu"
        
        "--compare"
            help = "Compare with existing results file"
            arg_type = String
            default = ""
        
        "--export-format"
            help = "Export format for results (csv, json, html, all)"
            arg_type = String
            default = "all"
        
        "command"
            help = "Command to run (benchmark, analyze, visualize)"
            required = true
    end
    
    return parse_args(s)
end

"""
Parse image size string
"""
function parse_img_size(size_str)
    parts = split(size_str, "x")
    if length(parts) != 2
        error("Invalid image size format. Expected WxH (e.g., 640x640)")
    end
    
    width = parse(Int, parts[1])
    height = parse(Int, parts[2])
    
    return (width, height)
end

"""
Run benchmark command
"""
function run_benchmark_command(args)
    # Parse models
    models = split(args["models"], ',')
    
    # Parse image size
    img_size = parse_img_size(args["img-size"])
    
    # Parse device
    device = Symbol(lowercase(args["device"]))
    if !(device in [:cpu, :gpu])
        error("Invalid device. Expected 'cpu' or 'gpu'")
    end
    
    # Create config
    config = CourtBench.BenchmarkConfig(
        dataset_path = args["dataset"],
        results_path = args["results"],
        models_to_test = models,
        batch_size = args["batch-size"],
        image_size = img_size,
        device = device
    )
    
    # Run benchmark
    results = CourtBench.run_benchmark(config)
    
    # Compare with existing results if requested
    if !isempty(args["compare"])
        if isfile(args["compare"])
            prev_results = CSV.read(args["compare"], DataFrame)
            println("\nComparison with previous results:")
            
            # Find common models
            common_models = intersect(results.model, prev_results.model)
            
            if isempty(common_models)
                println("No common models found for comparison")
            else
                # Create comparison table
                comparison = DataFrame(
                    model = String[],
                    kla_05_diff = Float64[],
                    kla_10_diff = Float64[],
                    cd_iou_diff = Float64[],
                    pcr_diff = Float64[],
                    inference_time_diff = Float64[]
                )
                
                # Calculate differences
                for model in common_models
                    current = results[results.model .== model, :]
                    previous = prev_results[prev_results.model .== model, :]
                    
                    push!(comparison, (
                        model = model,
                        kla_05_diff = current.kla_05[1] - previous.kla_05[1],
                        kla_10_diff = current.kla_10[1] - previous.kla_10[1],
                        cd_iou_diff = current.cd_iou[1] - previous.cd_iou[1],
                        pcr_diff = current.pcr[1] - previous.pcr[1],
                        inference_time_diff = current.inference_time[1] - previous.inference_time[1]
                    ))
                end
                
                # Display comparison
                println(comparison)
                
                # Save comparison
                CSV.write(joinpath(args["results"], "comparison.csv"), comparison)
            end
        else
            println("Comparison file not found: $(args["compare"])")
        end
    end
    
    return results
end

"""
Run analysis command
"""
function run_analysis_command(args)
    # Check if results file exists
    results_file = joinpath(args["results"], "results.csv")
    if !isfile(results_file)
        error("Results file not found: $(results_file)")
    end
    
    # Load results
    results = CSV.read(results_file, DataFrame)
    
    # Create config
    config = CourtBench.BenchmarkConfig(
        dataset_path = args["dataset"],
        results_path = args["results"]
    )
    
    # Generate report
    CourtBench.generate_report(results, config)
    
    println("Analysis completed. Results saved to: $(args["results"])")
    
    return results
end

"""
Run visualization command
"""
function run_visualization_command(args)
    # Check if results file exists
    results_file = joinpath(args["results"], "results.csv")
    if !isfile(results_file)
        error("Results file not found: $(results_file)")
    end
    
    # Load results
    results = CSV.read(results_file, DataFrame)
    
    # Create config
    config = CourtBench.BenchmarkConfig(
        dataset_path = args["dataset"],
        results_path = args["results"]
    )
    
    # Generate plots
    CourtBench.generate_plots(results, args["results"])
    
    println("Visualization completed. Results saved to: $(args["results"])")
    
    return results
end

"""
Main function
"""
function main()
    # Parse command-line arguments
    args = parse_commandline()
    
    # Run command
    if args["command"] == "benchmark"
        results = run_benchmark_command(args)
    elseif args["command"] == "analyze"
        results = run_analysis_command(args)
    elseif args["command"] == "visualize"
        results = run_visualization_command(args)
    else
        error("Unknown command: $(args["command"])")
    end
    
    # Success message
    command_name = uppercase(args["command"])
    println("\n$(command_name) COMPLETED SUCCESSFULLY")
    println("Timestamp: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    
    return results
end

# Run main function if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end