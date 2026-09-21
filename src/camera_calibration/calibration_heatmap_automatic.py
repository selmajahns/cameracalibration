import cv2
import numpy as np
import json
import time
import os
import argparse
from metavision_core.event_io import EventsIterator
from datetime import datetime
import sys

# Adds the current working directory (project root) to sys.path
sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from src.utils.settings import (
    SERIAL_LEFT, SERIAL_RIGHT, DELTA_T, CHECKERBOARD_ROWS, CHECKERBOARD_COLS,
    SQUARE_SIZE_M, COOLDOWN_SECONDS, MAX_SINGLE_SNAP_RMS, MIN_REQUIRED_SNAPS, BIAS_INCREMENT
)   

def configure_biases(iterator, logger):
    """ Hardware access to increase bias_diff_on/off by a fixed offset """
    try:
        device = iterator.reader.device
        biases = device.get_i_ll_biases()
        if biases:
            current_on = biases.get("bias_diff_on")
            current_off = biases.get("bias_diff_off")
            biases.set("bias_diff_on", current_on + BIAS_INCREMENT)
            biases.set("bias_diff_off", current_off + BIAS_INCREMENT)
            logger.info(f"Biases set: ON={current_on + BIAS_INCREMENT}, OFF={current_off + BIAS_INCREMENT}")
    except Exception as e:
        logger.warning(f"Could not configure hardware biases: {e}")

def save_prophesee_json(filename, width, height, mtx, dist, logger):
    """ Save calibration data in Metavision SDK compatible format """
    os.makedirs("config", exist_ok=True)
    full_path = os.path.join("config", filename)
    data = {
        "type": "pinhole", "width": width, "height": height,
        "K": mtx.flatten().tolist(),
        "D": dist.flatten().tolist()
    }
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Calibration parameters saved to: {full_path}")

def generate_point_heatmap(accumulated_mask):
    """ Generate a jet-colored heatmap based on accumulated corner detections """
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def save_points_json(side, objpoints, imgpoints, width, height, logger):
    """ Save extracted points for offline analysis """
    os.makedirs("data_analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join("data_analysis", f"points_{side}_{timestamp}.json")
    data = {
        "side": side, "width": width, "height": height,
        "objpoints": [op.tolist() for op in objpoints],
        "imgpoints": [ip.tolist() for ip in imgpoints]
    }
    with open(full_path, 'w') as f:
        json.dump(data, f)
    logger.info(f"Raw points saved to: {full_path}")

def main():
    parser = argparse.ArgumentParser(description="Automated Heatmap Calibration for GenX320")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--mode", choices=["calibrate", "analyze"], required=True)
    args = parser.parse_args()

    logger = setup_logging("heatmap_auto", args.side)
    serial = SERIAL_LEFT if args.side == "left" else SERIAL_RIGHT
    
    mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T) 
    configure_biases(mv_it, logger)
    height, width = mv_it.get_size()
    
    # Calibration flags
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2) * SQUARE_SIZE_M

    objpoints, imgpoints = [], []
    point_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps, last_cap = 0, time.time()

    logger.info(f"Streaming {args.side.upper()} camera. Move checkerboard. Press 'q' to stop.")

    for evs in mv_it:
        if evs.size == 0: continue
        
        # Filter for ON events (Polarity 1)
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs[evs['p']==1]['y'], evs[evs['p']==1]['x']] = 255
        
        # Pre-process for Sector-Based (SB) detector
        im_vis = cv2.bitwise_not(cv2.dilate(im, np.ones((2,2), np.uint8), iterations=1))
        display = cv2.cvtColor(im_vis, cv2.COLOR_GRAY2BGR)

        # Detect corners
        ret, corners = cv2.findChessboardCornersSB(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_sb)
        
        if ret:
            cv2.drawChessboardCorners(display, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners, ret)
            if time.time() - last_cap > COOLDOWN_SECONDS:
                objpoints.append(objp)
                imgpoints.append(corners)
                valid_snaps += 1
                last_cap = time.time()
                for c in corners:
                    cv2.circle(point_mask, (int(c[0][0]), int(c[0][1])), 2, 1.0, -1)
                logger.info(f"Snap {valid_snaps} captured.")

        # GUI
        heatmap_display = generate_point_heatmap(point_mask)
        cv2.putText(display, f"Snaps: {valid_snaps}", (10, 30), 1, 1.5, (255, 255, 0), 2)
        cv2.imshow("Acquisition Stream", display)
        cv2.imshow("Coverage Heatmap", heatmap_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

    if valid_snaps >= MIN_REQUIRED_SNAPS:
        save_points_json(args.side, objpoints, imgpoints, width, height, logger)
        
        if args.mode == "calibrate":
            logger.info("Starting two-stage calibration with outlier rejection...")
            res = cv2.calibrateCameraExtended(objpoints, imgpoints, (width, height), None, None)
            ret, mtx, dist, rvecs, tvecs, _, _, viewErr = res
            
            # Statistical threshold: Mean + StdDev
            thresh = np.mean(viewErr) + np.std(viewErr)
            f_obj, f_img = [], []
            for i, err in enumerate(viewErr.flatten()):
                if err < thresh and err < MAX_SINGLE_SNAP_RMS:
                    f_obj.append(objpoints[i])
                    f_img.append(imgpoints[i])
            
            if len(f_obj) >= MIN_REQUIRED_SNAPS:
                ret_f, mtx_f, dist_f, _, _ = cv2.calibrateCamera(f_obj, f_img, (width, height), None, None)
                logger.info(f"Final Refined RMS: {ret_f:.4f} (Original: {ret:.4f})")
                save_prophesee_json(f"camera_{args.side}.json", width, height, mtx_f, dist_f, logger)
            else:
                logger.error(f"Too many outliers. Only {len(f_obj)} snaps remained.")
    else:
        logger.error(f"Not enough snapshots ({valid_snaps}/20). Calibration aborted.")

if __name__ == "__main__":
    main()
