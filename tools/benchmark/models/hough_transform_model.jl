module HoughTransformModel

using Images
using ImageFiltering
using Statistics

# Import the Hough Transform implementation
include("hough.jl")
using .HoughTransform

export HoughModel, predict_court

"""
HoughModel struct for CourtBench compatibility
"""
mutable struct HoughModel
    params::Dict{String, Any}
    
    function HoughModel(params::Dict{String, Any} = Dict())
        # Default parameters
        default_params = Dict(
            "canny_low" => 50,
            "canny_high" => 150,
            "hough_threshold" => 50,
            "min_line_length" => 50,
            "max_line_gap" => 10,
            "angle_threshold" => 2.0,
            "court_ratio_min" => 1.5,
            "court_ratio_max" => 3.0,
            "edge_extend_ratio" => 0.1,
            "cluster_threshold" => 0.05
        )
        
        # Merge provided parameters with defaults
        merged_params = merge(default_params, params)
        
        return new(merged_params)
    end
end

"""
Prediction function that returns keypoints in the expected format
"""
function predict_court(model::HoughModel, img::Array{Float32, 3})
    # Ensure image is in the right format
    if size(img, 3) != 3
        error("Input image must have 3 channels (RGB)")
    end
    
    # Rearrange dimensions if needed (from CHW to HWC)
    if size(img, 1) == 3
        img = permutedims(img, (2, 3, 1))
    end
    
    # Run Hough Transform detection
    result = HoughTransform.detect_court(img, model.params)
    
    # Extract and return keypoints and confidence
    return result.keypoints, result.confidence
end

"""
Method to load Hough model
"""
function load_model(model_path::String; params::Dict{String, Any} = Dict())
    # For Hough transform, we just create a new model with parameters
    return HoughModel(params)
end

end # module