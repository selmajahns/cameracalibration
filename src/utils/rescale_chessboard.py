import json
import numpy as np
import os
import glob
import sys
import argparse
from datetime import datetime

# Add the project root to sys.path to load the custom logger
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging

DATA_FOLDER = 'data_analysis'

def get_latest_points_json(logger):
    """Finds the most recent raw dataset."""
    files = glob.glob(os.path.join(DATA_FOLDER, 'points_stereo_*.json'))
    if not files:
        logger.error(f"No original dataset found in {DATA_FOLDER}")
        sys.exit(1)
    
    # Restituisce il file più recente
    return max(files, key=os.path.getctime)

def main():
    parser = argparse.ArgumentParser(description="Rescale objpoints in stereo points JSON")
    parser.add_argument("--new-size", type=float, required=True, 
                        help="New square size in millimeters (e.g., 16.46)")
    args = parser.parse_args()

    logger = setup_logging("rescale_points", "stereo")

    json_path = get_latest_points_json(logger)
    logger.info(f"Loading dataset: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Calcola la dimensione del vecchio quadretto leggendo la distanza tra i primi due punti
    p0 = np.array(data['objpoints'][0][0])
    p1 = np.array(data['objpoints'][0][1])
    old_size_m = np.linalg.norm(p1 - p0)
    old_size_mm = old_size_m * 1000.0
    
    # 2. Calcola il fattore di conversione
    new_size_m = args.new_size / 1000.0
    scale_factor = new_size_m / old_size_m
    
    logger.info(f"Detected OLD square size: {old_size_mm:.4f} mm")
    logger.info(f"Target NEW square size:   {args.new_size:.4f} mm")
    logger.info(f"Applying scale factor:    {scale_factor:.6f}")

    # 3. Moltiplica tutti gli objpoints per il fattore di scala
    new_objpoints = []
    for snap in data['objpoints']:
        new_snap = []
        for pt in snap:
            new_pt = [pt[0] * scale_factor, pt[1] * scale_factor, pt[2] * scale_factor]
            new_snap.append(new_pt)
        new_objpoints.append(new_snap)

    data['objpoints'] = new_objpoints

    # 4. Salva il nuovo JSON con un nuovo timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"points_stereo_{timestamp}.json"
    new_filepath = os.path.join(DATA_FOLDER, new_filename)

    with open(new_filepath, 'w') as f:
        # Non uso indent=4 per mantenere il file compatto esattamente come l'originale
        json.dump(data, f)

    logger.info(f"Successfully saved scaled points to: {new_filepath}")
    logger.info(">>> You can now safely run 'analyze_stereo_rms.py' to generate the new calibration.")

if __name__ == "__main__":
    main()
