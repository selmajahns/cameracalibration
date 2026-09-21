# config/settings.py

import cv2

# ==========================================
# MASTER THESIS PROJECT - GLOBAL SETTINGS
# ==========================================

# --- HARDWARE SENSORS ---
SERIAL_LEFT = "genx320 11-003c"  #
SERIAL_RIGHT = "genx320 10-003c" #
IMG_WIDTH = 320
IMG_HEIGHT = 320
DELTA_T = 20000                  # Accumulation window in microseconds
STEREO_DELTA_T = 30000           # Accumulation window in microseconds for stereo system
TRACKER_DELTA_T = 20000          # Accumulation window in microseconds for tracking

# --- CALIBRATION BOARD ---
CHECKERBOARD_ROWS = 6     
CHECKERBOARD_COLS = 9  
SQUARE_SIZE_MM = 16.6           # Real physical size of a square (Aggiorna con la tua misura esatta!)
SQUARE_SIZE_M = SQUARE_SIZE_MM / 1000.0

# --- ALGORITHM TUNING ---
COOLDOWN_SECONDS = 3.0          # Minimum time between valid detections to prevent rapid-fire captures
MAX_SINGLE_SNAP_RMS = 1.0       # Maximum RMS error for a single snap to be considered valid
MIN_REQUIRED_SNAPS = 20         # Minimum number of valid snaps required for a successful calibration
MIN_EVENTS_THRESHOLD = 10       #
MAX_SYNC_DIFF_US = 200          #
BIAS_INCREMENT = 0              #
BIAS_INCREMENT_STEREO = 10      #
BIAS_INCREMENT_TRACKER = 10     #
FLAGS = cv2.CALIB_FIX_K3        # Calibration flag to fix the K3 distortion coefficient
FLAGS_STEREO = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
MAX_TRACK_DISTANCE = 0.02       # Max allowed distance (in meters) a particle can move between frames
MAX_AGE = 3                     # Frames to keep a track alive if temporarily lost
MIN_PARTICLE_AREA = 5           # Minimum pixels to consider a cluster as a valid particle

# --- DIRECTORIES ---
DATA_FOLDER = "data_analysis"   #
CONFIG_FOLDER = "config"
LOG_FOLDER = "logs"
RECORD_FOLDER = "data"