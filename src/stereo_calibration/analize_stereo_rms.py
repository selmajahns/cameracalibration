import cv2
import numpy as np
import json
import os
import glob
import matplotlib
# Use 'TkAgg' for interactive GUI display (crucial for VNC and 3D rotation)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

# Add the current working directory (project root) to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import SQUARE_SIZE_MM

def sort_snaps_by_fps(img_pts_L, img_pts_R):
    """
    Farthest Point Sampling (FPS) to ensure maximum spatial coverage 
    of the actual shared Field of View in stereovision.
    """
    num_snaps = len(img_pts_L)
    
    # Calculate 2D centroids (mean between L and R cameras for each snap)
    centroids = []
    for i in range(num_snaps):
        c_L = np.mean(img_pts_L[i], axis=0).flatten()
        c_R = np.mean(img_pts_R[i], axis=0).flatten()
        centroids.append((c_L + c_R) / 2.0)
    centroids = np.array(centroids)
    
    # 1. Find the global centroid and the point closest to it
    global_center = np.mean(centroids, axis=0)
    dists_to_center = np.linalg.norm(centroids - global_center, axis=1)
    first_idx = int(np.argmin(dists_to_center))
    
    sorted_indices = [first_idx]
    unselected = set(range(num_snaps))
    unselected.remove(first_idx)
    
    # Track the minimum distance from each "unselected" point to the "selected" ones
    min_dists = np.linalg.norm(centroids - centroids[first_idx], axis=1)
    
    # 2. Iteratively pick the point farthest from the already selected points
    while unselected:
        farthest_idx = max(unselected, key=lambda idx: min_dists[idx])
        sorted_indices.append(farthest_idx)
        unselected.remove(farthest_idx)
        
        # Update distances
        new_dists = np.linalg.norm(centroids - centroids[farthest_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return sorted_indices

def filter_edge_snaps(img_pts_L, img_pts_R, obj_pts, w, h, margin=15):
    """
    Discards snapshots where at least one checkerboard point 
    is too close to the sensor edges, and returns info about discarded ones.
    """
    valid_L, valid_R, valid_obj = [], [], []
    discarded_info = [] # Stores data for visualization
    
    for i, (pL, pR, pObj) in enumerate(zip(img_pts_L, img_pts_R, obj_pts)):
        pL_2d = pL.reshape(-1, 2)
        pR_2d = pR.reshape(-1, 2)
        
        out_L = np.any(pL_2d[:, 0] < margin) or np.any(pL_2d[:, 0] > w - margin) or \
                np.any(pL_2d[:, 1] < margin) or np.any(pL_2d[:, 1] > h - margin)
                
        out_R = np.any(pR_2d[:, 0] < margin) or np.any(pR_2d[:, 0] > w - margin) or \
                np.any(pR_2d[:, 1] < margin) or np.any(pR_2d[:, 1] > h - margin)
                
        if not (out_L or out_R):
            valid_L.append(pL)
            valid_R.append(pR)
            valid_obj.append(pObj)
        else:
            reason = 'LEFT' if out_L and not out_R else ('RIGHT' if out_R and not out_L else 'BOTH')
            discarded_info.append({
                'index': i,
                'pts_L': pL_2d,
                'pts_R': pR_2d,
                'reason': reason
            })
            
    return valid_L, valid_R, valid_obj, discarded_info

def visualize_discarded(discarded_info, w, h, margin, output_dir="data_analysis/discarded"):
    """
    Generates side-by-side images of the discarded snaps, 
    drawing the safe-zone margin and highlighting the offending points.
    """
    if not discarded_info:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    for info in discarded_info:
        idx = info['index']
        reason = info['reason']
        
        # Create blank black images for GenX320 canvas
        img_L = np.zeros((h, w, 3), dtype=np.uint8)
        img_R = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw the safe-zone margin (red rectangle)
        cv2.rectangle(img_L, (margin, margin), (w - margin, h - margin), (0, 0, 255), 1)
        cv2.rectangle(img_R, (margin, margin), (w - margin, h - margin), (0, 0, 255), 1)
        
        # Helper function to draw points
        def draw_pts(img, pts):
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                # If point is outside margin, draw RED and larger. Otherwise WHITE.
                if x < margin or x > w - margin or y < margin or y > h - margin:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # Red outlier
                else:
                    cv2.circle(img, (x, y), 3, (255, 255, 255), -1) # White normal
                    
        draw_pts(img_L, info['pts_L'])
        draw_pts(img_R, info['pts_R'])
                
        # Combine Left and Right side-by-side
        combined = np.hstack((img_L, img_R))
        
        # Add a text label
        cv2.putText(combined, f"Snap {idx:03d} | Cut by: {reason}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save to disk
        filepath = os.path.join(output_dir, f"discarded_snap_{idx:03d}.png")
        cv2.imwrite(filepath, combined)

def evaluate_metric_error(mtxL, distL, mtxR, distR, R, T, all_img_pts_L, all_img_pts_R, bw, bh):
    """
    Calculates the 3D metric error ON THE ENTIRE DATASET.
    The Median is highly robust and automatically filters out noisy outliers.
    """
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    
    raw_distances = []

    for i in range(len(all_img_pts_L)):
        pL_u = cv2.undistortPoints(all_img_pts_L[i].reshape(-1, 1, 2), mtxL, distL)
        pR_u = cv2.undistortPoints(all_img_pts_R[i].reshape(-1, 1, 2), mtxR, distR)
        
        pts4D = cv2.triangulatePoints(P1, P2, pL_u, pR_u)
        pts3D = (pts4D[:3, :] / pts4D[3, :]).T
        
        # Array Orientation Auto-Detect (prevents diagonal distance calculation)
        grid_test = pts3D.reshape(bh, bw, 3)
        if np.linalg.norm(grid_test[0,0] - grid_test[1,0]) > 0.030: 
            grid3D = pts3D.reshape(bw, bh, 3)
            rows, cols = bw, bh
        else:
            grid3D = grid_test
            rows, cols = bh, bw

        for r in range(rows):
            for c in range(cols):
                if c < cols - 1: # Horizontal edge
                    raw_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r, c+1]))
                if r < rows - 1: # Vertical edge
                    raw_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r+1, c]))
    
    raw_distances = np.array(raw_distances)
    errors = np.abs(raw_distances - (SQUARE_SIZE_MM))
    
    return np.mean(errors), np.median(errors), np.mean(raw_distances)

def run_two_step_analysis(logger):
    logger.info("--- STEREO COVERAGE (FPS) & ASYMPTOTIC CONVERGENCE ---")
    
    files = glob.glob("data_analysis/points_stereo_*.json")
    if not files:
        logger.error("No data files found in data_analysis/.")
        sys.exit(1)
        
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Using dataset: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)

    img_pts_L_raw = [np.array(i).astype(np.float32) for i in data["imgpoints_L"]]
    img_pts_R_raw = [np.array(i).astype(np.float32) for i in data["imgpoints_R"]]
    obj_pts_raw = [np.array(o).astype(np.float32) for o in data["objpoints"]]
    w, h = data.get("width", 320), data.get("height", 320)
    bw, bh = data.get("board_size", (9, 6))

    # Scale alignment for objpoints (ensures metric accuracy)
    json_square_size = np.linalg.norm(obj_pts_raw[0][0] - obj_pts_raw[0][1])
    scale_factor = SQUARE_SIZE_MM / json_square_size
    obj_pts_raw = [pts * scale_factor for pts in obj_pts_raw]

    # --- FILTER AND VISUALIZE OUTLIERS ---
    margin_px = 10
    img_pts_L_raw, img_pts_R_raw, obj_pts_raw, discarded_info = filter_edge_snaps(
        img_pts_L_raw, img_pts_R_raw, obj_pts_raw, w, h, margin=margin_px
    )
    logger.info(f"Margin Filter applied: discarded {len(discarded_info)} snaps too close to the edges.")
    
    if discarded_info:
        visualize_discarded(discarded_info, w, h, margin=margin_px)
        logger.info(f"Visualizations of discarded snaps saved in 'data_analysis/discarded/'.")
    def load_cam(side):
        with open(f"config/camera_{side}.json", 'r') as f:
            d = json.load(f)
        return np.array(d.get("K", d.get("camera_matrix"))).reshape(3,3).astype(np.float32), \
               np.array(d.get("D", d.get("dist_coeffs"))).astype(np.float32)

    try:
        mtx_L, dist_L = load_cam("left")
        mtx_R, dist_R = load_cam("right")
    except FileNotFoundError as e:
        logger.error(f"Missing intrinsic calibration file: {e}")
        sys.exit(1)

    # Sort using Farthest Point Sampling
    sorted_idx = sort_snaps_by_fps(img_pts_L_raw, img_pts_R_raw)
    
    obj_pts = [obj_pts_raw[i] for i in sorted_idx]
    img_pts_L = [img_pts_L_raw[i] for i in sorted_idx]
    img_pts_R = [img_pts_R_raw[i] for i in sorted_idx]

    # Keep a full unsorted copy for the global validation test
    all_img_pts_L = img_pts_L_raw
    all_img_pts_R = img_pts_R_raw

    total = len(obj_pts)
    steps = range(15, total + 1, 1)
    results = {"count": [], "rms_raw": [], "rms_clean": [], "baseline": [], "mae_metric": [], "median_metric": []}
    
    final_params = {}

    logger.info("Evaluating models and testing on the ENTIRE FOV volume...")

    for count in steps:
        curr_obj = obj_pts[:count]
        curr_L = img_pts_L[:count]
        curr_R = img_pts_R[:count]

        # Step A: Raw Calibration
        res = cv2.stereoCalibrateExtended(
            curr_obj, curr_L, curr_R, mtx_L, dist_L, mtx_R, dist_R, (w, h),
            None, None, None, None, None, flags=cv2.CALIB_FIX_INTRINSIC
        )
        ret_raw, _, _, _, _, _, _, _, _, perViewErrors = res
        
        # Outlier Filtering (Mean + 1 StdDev)
        errors = np.mean(perViewErrors, axis=1).flatten()
        threshold = np.mean(errors) + np.std(errors)
        
        f_obj, f_L, f_R = [], [], []
        for i, err in enumerate(errors):
            if err < threshold:
                f_obj.append(curr_obj[i])
                f_L.append(curr_L[i])
                f_R.append(curr_R[i])

        # Step B: Clean Calibration
        ret_clean, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            f_obj, f_L, f_R, mtx_L, dist_L, mtx_R, dist_R, (w, h),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        # Global Metric Validation (Test on all 100 snaps)
        current_mae, current_median, measured_size = evaluate_metric_error(
            mtx_L, dist_L, mtx_R, dist_R, R, T, all_img_pts_L, all_img_pts_R, bw, bh
        )
        baseline_mm = float(np.linalg.norm(T))
        
        results["count"].append(count)
        results["rms_raw"].append(ret_raw)
        results["rms_clean"].append(ret_clean)
        results["baseline"].append(baseline_mm)
        results["mae_metric"].append(current_mae)
        results["median_metric"].append(current_median)
        
        # Continuously overwrite params. 
        # By the end of the loop, it holds the maximally converged model.
        final_params = {
            "R": R.tolist(), "T": T.tolist(), 
            "E": E.tolist(), "F": F.tolist(),
            "RMS": float(ret_clean),                      # <-- Cast a float standard
            "Metric_Median_mm": float(current_median),    # <-- Cast a float standard
            "Metric_MAE_mm": float(current_mae),          # <-- Cast a float standard
            "n_images_used": int(count),                  # <-- Cast a int standard
            "n_images_kept": int(len(f_obj))              # <-- Cast a int standard
        }
        
        logger.info(f"Snap {count:03d} | Clean RMS: {ret_clean:.4f} px | Global Median Error: {current_median:.4f} mm | Baseline: {baseline_mm:.2f} mm")

    # --- FINAL LOGGING ---
    logger.info("==================================================")
    logger.info("🏆 FINAL EXTRINSIC PARAMETERS (Max Convergence Model)")
    logger.info(f"Global Median Error: {final_params['Metric_Median_mm']:.4f} mm")
    logger.info(f"Input Snapshots:     {final_params['n_images_used']} (Kept post-filter: {final_params['n_images_kept']})")
    logger.info(f"Reprojection RMS:    {final_params['RMS']:.4f} pixels")
    logger.info(f"Baseline (T):        {np.linalg.norm(np.array(final_params['T'])):.2f} mm")
    logger.info("==================================================")

    # Save Configuration
    config_path = os.path.join("config", "stereo_params.json")
    os.makedirs("config", exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(final_params, f, indent=4)
    logger.info(f"Optimal parameters saved successfully to: {config_path}")

    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 1. RMS Plot
    ax1.plot(results["count"], results["rms_raw"], 'o--', label='Raw RMS', color='gray', alpha=0.5)
    ax1.plot(results["count"], results["rms_clean"], 'o-', label='Clean RMS', color='tab:blue', linewidth=2)
    ax1.set_title("RMS Convergence (FPS Sorting)", fontsize=12)
    ax1.set_xlabel("Input Sample Size")
    ax1.set_ylabel("Global RMS (pixels)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # The chosen model is the last step
    chosen_count = results["count"][-1]

    # 2. Baseline Plot
    ax2.plot(results["count"], results["baseline"], 's-', color='tab:red', linewidth=2)
    ax2.axvline(chosen_count, color='green', linestyle='--', alpha=0.8, label=f'Chosen Model ({chosen_count} snaps)')
    ax2.set_title("Baseline Stability", fontsize=12)
    ax2.set_xlabel("Input Sample Size")
    ax2.set_ylabel("Estimated Baseline (mm)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Metric Error Plot
    ax3.plot(results["count"], results["median_metric"], 'D-', color='teal', linewidth=2, label='Median (Global Test)')
    ax3.plot(results["count"], results["mae_metric"], 'o--', color='orange', alpha=0.5, label='Mean MAE (Global Test)')
    
    # Highlight the final chosen step
    ax3.plot(chosen_count, results["median_metric"][-1], marker='*', markersize=15, color='gold', markeredgecolor='black')
    ax3.axvline(chosen_count, color='gold', linestyle='--', alpha=0.8)
    
    ax3.set_title("Global Metric Error 3D (Convergence)", fontsize=12)
    ax3.set_xlabel("Input Sample Size")
    ax3.set_ylabel("Error (mm)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join("data_analysis", "rms_stereo_plot_metric_fps.png")
    plt.savefig(plot_path)
    logger.info(f"Convergence plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Initialize logger for stereo analysis
    logger = setup_logging("analyze_stereo_rms", "stereo")
    run_two_step_analysis(logger)
