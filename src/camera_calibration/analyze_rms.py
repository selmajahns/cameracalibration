import numpy as np
import cv2
import json
import os
import glob
import matplotlib.pyplot as plt
import argparse
import sys

# Adds the current working directory (project root) to sys.path
sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from src.utils.settings import FLAGS

def sort_snaps_by_fps(imgpoints_all):
    """
    Farthest Point Sampling (FPS) to ensure maximum spatial coverage 
    of the single camera's Field of View.
    """
    num_snaps = len(imgpoints_all)
    
    # Calculate 2D centroids for each snap
    centroids = []
    for imgp in imgpoints_all:
        imgp_arr = np.array(imgp, dtype=np.float32).reshape(-1, 2)
        centroids.append(np.mean(imgp_arr, axis=0))
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

def analyze_rms_fps_asymptotic(objpoints_all, imgpoints_all, img_size, camera_side, logger):
    logger.info(f"--- SINGLE CAMERA ASYMPTOTIC CONVERGENCE ({camera_side.upper()}) ---")
    
    # 1. SORTING PHASE: Farthest Point Sampling
    sorted_idx = sort_snaps_by_fps(imgpoints_all)
    obj_pts = [objpoints_all[i] for i in sorted_idx]
    img_pts = [imgpoints_all[i] for i in sorted_idx]
    
    # 2. BATCH PROCESSING (Iterative evaluation)
    OUTLIER_MULTIPLIER = 1.5

    stats_snapshots = []
    stats_rms = []
    stats_std = []
    
    # This dictionary will be continuously overwritten and end up holding the model 
    # computed on the maximum number of images (asymptotic convergence).
    final_results = {"mtx": None, "dist": None, "count": 0, "used": 0, "rms": 0.0}

    logger.info(f"Processing FPS sorted pool for {camera_side} camera...")
    
    total = len(obj_pts)
    steps = range(10, total + 1, 1)

    for count in steps:
        curr_obj = [np.array(o, dtype=np.float32) for o in obj_pts[:count]]
        curr_img = [np.array(i, dtype=np.float32) for i in img_pts[:count]]

        try:
            # Stage 1: Initial Calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(curr_obj, curr_img, img_size, None, None, flags=FLAGS)

            # Stage 2: Refinement (Outlier Removal based on Reprojection Error)
            errors = []
            for i in range(len(curr_obj)):
                p2, _ = cv2.projectPoints(curr_obj[i], rvecs[i], tvecs[i], mtx, dist)
                errors.append(cv2.norm(curr_img[i], p2, cv2.NORM_L2) / len(p2))

            median_err = np.median(errors)
            # Filter out snapshots with extreme reprojection errors
            refined_indices = [i for i, err in enumerate(errors) if err <= median_err * OUTLIER_MULTIPLIER]
            ref_obj = [curr_obj[i] for i in refined_indices]
            ref_img = [curr_img[i] for i in refined_indices]

            # Re-calibrate with clean data
            ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(ref_obj, ref_img, img_size, None, None, flags=FLAGS)

            # Calculate Standard Deviation for the plot
            ref_errors = []
            for i in range(len(ref_obj)):
                p2, _ = cv2.projectPoints(ref_obj[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
                ref_errors.append(cv2.norm(ref_img[i], p2, cv2.NORM_L2) / len(p2))

            curr_std = np.std(ref_errors)
            
            stats_snapshots.append(count)
            stats_rms.append(ret_r)
            stats_std.append(curr_std)

            # Overwrite the final results. At the end of the loop, it will hold the N=100 model.
            final_results = {
                "mtx": mtx_r, 
                "dist": dist_r, 
                "count": count, 
                "used": len(ref_obj),
                "rms": ret_r
            }

            logger.info(f"Snap {count:03d} | Clean RMS: {ret_r:.4f} ± {curr_std:.4f} px | Kept: {len(ref_obj)}/{count}")

        except cv2.error as e:
            logger.error(f"OpenCV Error at count {count}: {e}")
            break

    # 3. OUTPUT GENERATION
    logger.info("==================================================")
    logger.info(f"🏆 FINAL INTRINSIC PARAMETERS (Max Convergence Model)")
    logger.info(f"Camera:              {camera_side.upper()}")
    logger.info(f"Snapshots provided:  {final_results['count']} (Kept post-filter: {final_results['used']})")
    logger.info(f"Final RMS:           {final_results['rms']:.4f} pixels")
    logger.info("==================================================")

    # Save to JSON config file
    os.makedirs("config", exist_ok=True)
    cfg_path = os.path.join("config", f"camera_{camera_side}.json")
    cfg_data = {
        "type": "pinhole", 
        "width": img_size[0], 
        "height": img_size[1],
        "K": final_results['mtx'].tolist(),
        "D": final_results['dist'].tolist(),
        "RMS": final_results['rms'],
        "n_images_used": final_results['used']
    }
    with open(cfg_path, 'w') as f:
        json.dump(cfg_data, f, indent=4)
    logger.info(f"Optimal parameters saved to: {cfg_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(stats_snapshots, stats_rms, yerr=stats_std, fmt='-o', color='teal', ecolor='orange', capsize=3, label='Clean RMS ± StdDev')
    
    # Mark the chosen model (the last point)
    chosen_count = stats_snapshots[-1]
    plt.plot(chosen_count, stats_rms[-1], marker='*', markersize=15, color='gold', markeredgecolor='black')
    plt.axvline(chosen_count, color='gold', linestyle='--', alpha=0.8, label=f'Chosen Convergence Model ({chosen_count} snaps)')

    plt.title(f'RMS Progression - {camera_side.upper()} (FPS Asymptotic Convergence)')
    plt.xlabel("Input Sample Size")
    plt.ylabel("Reprojection Error (pixels)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join("data_analysis", f"rms_plot_{camera_side}_fps.png")
    plt.savefig(plot_path)
    logger.info(f"Convergence plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=["left", "right"])
    args = parser.parse_args()
    
    logger = setup_logging("analyze_rms", args.side)

    data_dir = "data_analysis"
    files = glob.glob(os.path.join(data_dir, f"points_{args.side}_*.json"))
    if not files:
        logger.error(f"No point files found for {args.side} camera in {data_dir}/.")
        sys.exit(1)

    latest = max(files, key=os.path.getctime)
    logger.info(f"Loading dataset: {latest}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    # Extract dimensions from the JSON to ensure accurate K matrices, defaulting to GenX320 resolution
    w, h = data.get("width", 320), data.get("height", 320)
    
    analyze_rms_fps_asymptotic(data['objpoints'], data['imgpoints'], (w, h), args.side, logger)
