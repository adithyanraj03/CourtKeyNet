module HoughTransform

using Images
using ImageFiltering
using LinearAlgebra
using Statistics
using Clustering

export detect_court

"""
CourtResult struct containing keypoints and confidence
"""
struct CourtResult
    keypoints::Matrix{Float64}  # 4×2 matrix of keypoints (normalized)
    confidence::Float64
end

"""
Detect badminton court using Hough transform
"""
function detect_court(img::Array{T,3}, params::Dict = Dict()) where T <: AbstractFloat
    # Default parameters
    default_params = Dict(
        "canny_low" => 50,
        "canny_high" => 150,
        "hough_threshold" => 50,
        "min_line_length" => 50,
        "max_line_gap" => 10,
        "angle_threshold" => 2.0,  # degrees
        "court_ratio_min" => 1.5,  # width/height ratio min
        "court_ratio_max" => 3.0,  # width/height ratio max
        "edge_extend_ratio" => 0.1, # extend edges by this ratio
        "cluster_threshold" => 0.05 # distance threshold for clustering corners
    )
    
    # Merge provided parameters with defaults
    for (key, value) in default_params
        if !haskey(params, key)
            params[key] = value
        end
    end
    
    # Get image dimensions
    height, width, _ = size(img)
    img_gray = RGB.(img) |> Gray
    
    # Apply Gaussian blur to reduce noise
    img_blurred = imfilter(img_gray, Kernel.gaussian(2))
    
    # Edge detection with Canny
    edges = canny(img_blurred, (params["canny_low"] / 255, params["canny_high"] / 255))
    
    # Hough transform for line detection
    lines = hough_lines(edges, params["hough_threshold"], 
                       params["min_line_length"], 
                       params["max_line_gap"])
    
    # Filter and group lines by orientation (horizontal and vertical)
    h_lines, v_lines = filter_and_group_lines(lines, params["angle_threshold"])
    
    # If not enough lines detected, return empty result
    if length(h_lines) < 2 || length(v_lines) < 2
        # Return default keypoints with low confidence
        return CourtResult(zeros(Float64, 4, 2), 0.1)
    end
    
    # Extend lines to ensure better intersection points
    ext_h_lines = extend_lines(h_lines, width, params["edge_extend_ratio"])
    ext_v_lines = extend_lines(v_lines, height, params["edge_extend_ratio"])
    
    # Find all intersection points between horizontal and vertical lines
    intersections = find_intersections(ext_h_lines, ext_v_lines)
    
    # Filter intersection points to those inside the image
    valid_intersections = filter_intersections(intersections, width, height)
    
    # If not enough intersections, return empty result
    if length(valid_intersections) < 4
        return CourtResult(zeros(Float64, 4, 2), 0.2)
    end
    
    # Cluster intersection points to find court corners
    corner_points = cluster_court_corners(valid_intersections, params["cluster_threshold"], width, height)
    
    # If not enough corners, return empty result
    if length(corner_points) < 4
        return CourtResult(zeros(Float64, 4, 2), 0.3)
    end
    
    # Sort corner points in clockwise order
    sorted_corners = sort_corners_clockwise(corner_points)
    
    # Calculate court aspect ratio for validation
    court_ratio = calculate_court_ratio(sorted_corners)
    
    # Validate court aspect ratio
    if court_ratio < params["court_ratio_min"] || court_ratio > params["court_ratio_max"]
        # Court shape is unreasonable, return with low confidence
        return CourtResult(normalized_keypoints(sorted_corners, width, height), 0.4)
    end
    
    # Normalize keypoints to [0,1] range
    keypoints = normalized_keypoints(sorted_corners, width, height)
    
    # Calculate confidence based on line strength and corner clustering quality
    confidence = calculate_confidence(lines, corner_points)
    
    return CourtResult(keypoints, confidence)
end

"""
Canny edge detection
"""
function canny(img::Array{Gray{T},2}, thresholds::Tuple{Real,Real}) where T <: Real
    # Calculate gradients
    grad_x = imfilter(Float64.(img), Kernel.sobel()[2])
    grad_y = imfilter(Float64.(img), Kernel.sobel()[1])
    
    # Calculate gradient magnitude and direction
    magnitude = sqrt.(grad_x.^2 .+ grad_y.^2)
    direction = atan.(grad_y, grad_x)
    
    # Non-maximum suppression
    suppressed = zeros(size(magnitude))
    for i in 2:size(magnitude, 1)-1
        for j in 2:size(magnitude, 2)-1
            # Quantize direction to 0, 45, 90, or 135 degrees
            theta = direction[i, j] * 180 / π
            if theta < 0
                theta += 180
            end
            
            # Round angle to nearest 45 degrees
            angle = round(Int, theta / 45) * 45
            if angle == 180
                angle = 0
            end
            
            # Check neighbors along gradient direction
            if angle == 0  # East-West direction
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elseif angle == 45  # Northeast-Southwest direction
                neighbors = [magnitude[i+1, j-1], magnitude[i-1, j+1]]
            elseif angle == 90  # North-South direction
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else  # angle == 135, Northwest-Southeast direction
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            end
            
            # Suppress if not a local maximum
            if magnitude[i, j] >= maximum(neighbors)
                suppressed[i, j] = magnitude[i, j]
            end
        end
    end
    
    # Double thresholding
    low, high = thresholds
    strong_edges = suppressed .>= high
    weak_edges = (suppressed .>= low) .& (suppressed .< high)
    
    # Edge tracking by hysteresis
    final_edges = copy(strong_edges)
    
    # Define neighbors
    neighbors = [(i, j) for i in -1:1 for j in -1:1 if !(i == 0 && j == 0)]
    
    # Track edges
    changes = true
    while changes
        changes = false
        for i in 2:size(weak_edges, 1)-1
            for j in 2:size(weak_edges, 2)-1
                if weak_edges[i, j] && !final_edges[i, j]
                    # Check if connected to a strong edge
                    for (di, dj) in neighbors
                        if final_edges[i+di, j+dj]
                            final_edges[i, j] = true
                            changes = true
                            break
                        end
                    end
                end
            end
        end
    end
    
    return final_edges
end

"""
Hough transform for line detection
"""
function hough_lines(edges::BitMatrix, threshold::Int, min_line_length::Int, max_line_gap::Int)
    height, width = size(edges)
    
    # Define Hough space
    diagonal = sqrt(height^2 + width^2)
    rho_range = Int(ceil(diagonal))
    num_rhos = 2 * rho_range + 1
    num_thetas = 180
    thetas = range(0, π, length=num_thetas)
    
    # Initialize accumulator
    accumulator = zeros(Int, num_rhos, num_thetas)
    
    # Populate accumulator
    y_idxs, x_idxs = findall(edges .== 1).I
    for (y, x) in zip(y_idxs, x_idxs)
        for (theta_idx, theta) in enumerate(thetas)
            rho = round(Int, x * cos(theta) + y * sin(theta))
            rho_idx = rho + rho_range + 1
            if 1 <= rho_idx <= num_rhos
                accumulator[rho_idx, theta_idx] += 1
            end
        end
    end
    
    # Find peaks in accumulator
    lines = Tuple{Float64, Float64, Float64, Float64}[]
    for theta_idx in 1:num_thetas
        for rho_idx in 1:num_rhos
            if accumulator[rho_idx, theta_idx] >= threshold
                # Extract line parameters
                rho = rho_idx - rho_range - 1
                theta = thetas[theta_idx]
                
                # Find line points
                line_points = Tuple{Int, Int}[]
                
                # For vertical lines
                if abs(sin(theta)) < 1e-6
                    x = round(Int, rho / cos(theta))
                    for y in 1:height
                        push!(line_points, (y, x))
                    end
                else
                    # For other lines
                    for x in 1:width
                        y = round(Int, (rho - x * cos(theta)) / sin(theta))
                        if 1 <= y <= height
                            push!(line_points, (y, x))
                        end
                    end
                end
                
                # Apply min_line_length filter
                if length(line_points) < min_line_length
                    continue
                end
                
                # Apply max_line_gap filter by finding connected segments
                segments = []
                current_segment = [line_points[1]]
                
                for i in 2:length(line_points)
                    y1, x1 = line_points[i-1]
                    y2, x2 = line_points[i]
                    gap = sqrt((y2 - y1)^2 + (x2 - x1)^2)
                    
                    if gap <= max_line_gap
                        push!(current_segment, line_points[i])
                    else
                        if length(current_segment) >= min_line_length
                            push!(segments, current_segment)
                        end
                        current_segment = [line_points[i]]
                    end
                end
                
                # Add last segment if valid
                if length(current_segment) >= min_line_length
                    push!(segments, current_segment)
                end
                
                # Convert segments to line endpoints
                for segment in segments
                    y1, x1 = segment[1]
                    y2, x2 = segment[end]
                    push!(lines, (Float64(x1), Float64(y1), Float64(x2), Float64(y2)))
                end
            end
        end
    end
    
    return lines
end

"""
Filter lines and group them into horizontal and vertical lines
"""
function filter_and_group_lines(lines, angle_threshold)
    horizontal_lines = []
    vertical_lines = []
    
    for (x1, y1, x2, y2) in lines
        # Calculate line angle in degrees
        angle = abs(atand((y2 - y1), (x2 - x1)))
        
        # Classify as horizontal or vertical
        if angle < angle_threshold || angle > 180 - angle_threshold
            push!(horizontal_lines, (x1, y1, x2, y2))
        elseif abs(angle - 90) < angle_threshold
            push!(vertical_lines, (x1, y1, x2, y2))
        end
    end
    
    return horizontal_lines, vertical_lines
end

"""
Extend lines to ensure better intersection points
"""
function extend_lines(lines, max_dim, extend_ratio)
    extended_lines = []
    
    for (x1, y1, x2, y2) in lines
        # Calculate line direction vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize direction vector
        length = sqrt(dx^2 + dy^2)
        if length < 1e-6
            continue
        end
        
        dx /= length
        dy /= length
        
        # Extend by a percentage of the max dimension
        extend_dist = max_dim * extend_ratio
        
        # Calculate new endpoints
        new_x1 = x1 - dx * extend_dist
        new_y1 = y1 - dy * extend_dist
        new_x2 = x2 + dx * extend_dist
        new_y2 = y2 + dy * extend_dist
        
        push!(extended_lines, (new_x1, new_y1, new_x2, new_y2))
    end
    
    return extended_lines
end

"""
Find intersections between two line sets
"""
function find_intersections(h_lines, v_lines)
    intersections = Tuple{Float64, Float64}[]
    
    for (x1, y1, x2, y2) in h_lines
        for (x3, y3, x4, y4) in v_lines
            # Line intersection formula
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            
            # Check if lines are parallel
            if abs(denom) < 1e-6
                continue
            end
            
            # Calculate intersection point
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
            
            # Check if intersection is within line segments
            if 0 <= ua <= 1 && 0 <= ub <= 1
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                push!(intersections, (x, y))
            end
        end
    end
    
    return intersections
end

"""
Filter intersections to those inside the image
"""
function filter_intersections(intersections, width, height)
    valid_intersections = []
    
    # Add margin to allow intersections slightly outside the image
    margin = 0.1
    
    for (x, y) in intersections
        if -margin * width <= x <= width * (1 + margin) && 
           -margin * height <= y <= height * (1 + margin)
            push!(valid_intersections, (x, y))
        end
    end
    
    return valid_intersections
end

"""
Cluster intersection points to find court corners
"""
function cluster_court_corners(intersections, threshold, width, height)
    if length(intersections) < 4
        return intersections
    end
    
    # Convert to array for clustering
    points = hcat([[x, y] for (x, y) in intersections]...)'
    
    # Normalize coordinates for better clustering
    scaled_points = copy(points)
    scaled_points[:, 1] ./= width
    scaled_points[:, 2] ./= height
    
    # Determine number of clusters (should be 4 for a rectangle)
    # Start with k=4 court corners
    k = 4
    
    # Cluster the intersection points
    result = kmeans(scaled_points', k, init=:kmpp)
    
    # Get cluster centers
    centers = result.centers'
    
    # Un-normalize coordinates
    centers[:, 1] .*= width
    centers[:, 2] .*= height
    
    # Convert back to tuples
    corner_points = [(centers[i, 1], centers[i, 2]) for i in 1:size(centers, 1)]
    
    return corner_points
end

"""
Sort corners in clockwise order, starting from top-left
"""
function sort_corners_clockwise(corners)
    # Find center point
    center_x = sum([x for (x, _) in corners]) / length(corners)
    center_y = sum([y for (_, y) in corners]) / length(corners)
    
    # Sort corners by angle from center
    sorted_corners = sort(corners, by = corner -> begin
        x, y = corner
        atan(y - center_y, x - center_x)
    end)
    
    return sorted_corners
end

"""
Calculate court aspect ratio for validation
"""
function calculate_court_ratio(corners)
    if length(corners) < 4
        return 0.0
    end
    
    # Calculate distances between consecutive corners
    distances = []
    for i in 1:length(corners)
        j = i % length(corners) + 1
        x1, y1 = corners[i]
        x2, y2 = corners[j]
        dist = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        push!(distances, dist)
    end
    
    # Sort distances (should have 2 pairs of similar lengths)
    sort!(distances)
    
    # Calculate aspect ratio (longer side / shorter side)
    width = (distances[3] + distances[4]) / 2
    height = (distances[1] + distances[2]) / 2
    
    if height < 1e-6
        return 0.0
    end
    
    return width / height
end

"""
Normalize keypoints to [0,1] range
"""
function normalized_keypoints(corners, width, height)
    keypoints = zeros(Float64, 4, 2)
    
    for i in 1:min(4, length(corners))
        x, y = corners[i]
        keypoints[i, 1] = clamp(x / width, 0, 1)
        keypoints[i, 2] = clamp(y / height, 0, 1)
    end
    
    return keypoints
end

"""
Calculate confidence based on line strength and corner clustering quality
"""
function calculate_confidence(lines, corners)
    if length(lines) < 4 || length(corners) < 4
        return 0.5
    end
    
    # Base confidence on number of lines detected
    line_confidence = min(1.0, length(lines) / 10)
    
    # Calculate corner quality based on how well they form a rectangle
    # (simplified calculation)
    corner_x = [x for (x, _) in corners]
    corner_y = [y for (_, y) in corners]
    
    # Rectangle property: opposite sides should be parallel and equal length
    if length(corners) == 4
        width1 = sqrt((corner_x[2] - corner_x[1])^2 + (corner_y[2] - corner_y[1])^2)
        width2 = sqrt((corner_x[4] - corner_x[3])^2 + (corner_y[4] - corner_y[3])^2)
        height1 = sqrt((corner_x[3] - corner_x[2])^2 + (corner_y[3] - corner_y[2])^2)
        height2 = sqrt((corner_x[1] - corner_x[4])^2 + (corner_y[1] - corner_y[4])^2)
        
        # Ratio between opposite sides (should be close to 1)
        width_ratio = min(width1, width2) / max(width1, width2)
        height_ratio = min(height1, height2) / max(height1, height2)
        
        corner_confidence = (width_ratio + height_ratio) / 2
    else
        corner_confidence = 0.5
    end
    
    # Combined confidence score
    return (line_confidence + corner_confidence) / 2
end

end # module