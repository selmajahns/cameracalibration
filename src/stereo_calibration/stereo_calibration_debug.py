#!/usr/bin/env python3
"""
Stereo calibration - DEBUG edition
==================================

Same idea as stereo_calibration_automatic.py, but built to show WHY a snap is
(not) being captured.  Run it from the project root, exactly like the original:

    python src/stereo_calibration/stereo_calibration_debug.py --mode calibrate

What you see
------------
* Window "Detector input": the exact left | right images that are handed to
  OpenCV's findChessboardCornersSB (after polarity filter, morphology, invert).
  Each side has a GREEN border when the board was found and a RED border when
  not, plus event count, fill %, and a status word.
* Window "Coverage heatmap": where corners have been captured so far.
* Console / log file: every 5 s a [status] line and, when something is wrong,
  a [hint] line saying which stage is failing (no data / no sync / few events /
  board not found / only one camera sees it).

About the flashing board
------------------------
The board page (checkerboard_8128 2.html) swaps black/white every 400 ms, so a
camera only sees events for a few ms after each flip; most 30 ms frames are
EMPTY, which is normal.  To keep the window readable, the panels HOLD the last
frame that had a real burst of events (the bar shows how old it is).  With
polarity ON (default) each flip shows one phase of the board as filled squares;
BOTH would fill every square and the pattern disappears - do not use BOTH here.
Roughly half of the flips can arrive split over two frames (screen refresh vs the
30 ms window); those attempts fail, which is normal - only both=0 for a long
time is a problem.

Live keys (click a window first)
--------------------------------
  q      quit (and calibrate if enough snaps)
  p      cycle polarity used for the image:  ON -> OFF -> BOTH
  m      cycle morphology:                   close3 -> dilate2 -> none
  i      toggle invert (detector wants dark squares on a light background)
  s      SCAN board sizes on the current frame (finds the right --rows/--cols)
  SPACE  save the current raw + detector-input images to ./debug_frames/

Safety: the result is written to config/stereo_params_<timestamp>.json.
config/stereo_params.json is only replaced if you pass --overwrite (the old one
is backed up first).
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from queue import Empty

import cv2
import numpy as np

sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from src.utils import settings as S

POLARITY_MODES = ["ON", "OFF", "BOTH"]
MORPH_MODES = ["close3", "dilate2", "none"]

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 255)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)


# ----------------------------------------------------------------------------
# Image helpers (pure functions - easy to test without cameras)
# ----------------------------------------------------------------------------
def select_events(evs, polarity_mode):
    """Filter a structured event array by polarity mode (ON / OFF / BOTH)."""
    if polarity_mode == "ON":
        return evs[evs["p"] == 1]
    if polarity_mode == "OFF":
        return evs[evs["p"] == 0]
    return evs


def events_to_image(evs, width, height):
    """Binary accumulation image: 255 wherever at least one event fired."""
    im = np.zeros((height, width), np.uint8)
    if evs.size:
        x, y = evs["x"], evs["y"]
        ok = (x < width) & (y < height)
        im[y[ok], x[ok]] = 255
    return im


def preprocess(im, morph, invert):
    """Turn the event image into what the chessboard detector receives."""
    if morph == "close3":
        out = cv2.morphologyEx(im, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    elif morph == "dilate2":
        out = cv2.dilate(im, np.ones((2, 2), np.uint8), iterations=1)
    else:
        out = im
    return cv2.bitwise_not(out) if invert else out


def detect(proc, pattern, flags):
    """findChessboardCornersSB wrapper that never raises."""
    try:
        ret, corners = cv2.findChessboardCornersSB(proc, pattern, flags)
    except cv2.error:
        return False, None
    return bool(ret), corners


def generate_point_heatmap(mask):
    vis = np.clip(mask * 20.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heat[vis == 0] = [0, 0, 0]
    return heat


def put(img, text, org, color=WHITE, scale=0.45, thick=1):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_panel(name, proc, status, corners, pattern, ev_count, fill_pct, scale):
    """One side (L or R) of the 'Detector input' window."""
    panel = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    color = {"FOUND": GREEN, "NOT FOUND": RED, "FEW EVENTS": ORANGE}.get(status, GREY)
    if status == "FOUND" and corners is not None:
        cv2.drawChessboardCorners(panel, pattern, (corners * scale).astype(np.float32), True)
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), color, 4)
    put(panel, f"{name}: {status}", (8, 20), color, 0.6, 2)
    put(panel, f"events {ev_count}   pixels on {fill_pct:.1f}%", (8, 40))
    return panel


def make_display(panel_l, panel_r, bar_lines):
    gap = np.full((panel_l.shape[0], 4, 3), 60, np.uint8)
    top = np.hstack([panel_l, gap, panel_r])
    bar = np.zeros((22 * len(bar_lines) + 8, top.shape[1], 3), np.uint8)
    for i, (txt, col) in enumerate(bar_lines):
        put(bar, txt, (8, 20 + 22 * i), col)
    return np.vstack([top, bar])


def scan_pattern_sizes(proc_l, proc_r, logger, lo=4, hi=10):
    """
    Find the largest inner-corner grid OpenCV can detect in each camera's current frame.

    OpenCV also detects SUB-grids of a bigger board, so the *largest* detected size is the
    board's real size - but only if that camera sees the whole board in this frame.  Sizes
    are searched from large to small and each camera stops at its first hit (fast).
    Orientation does not matter for detection, so only rows <= cols is tried.
    """
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    cands = sorted({(min(a, b), max(a, b)) for a in range(lo, hi + 1) for b in range(lo, hi + 1)},
                   key=lambda ab: (-ab[0] * ab[1], ab))
    logger.info(f"[scan] searching grids {lo}..{hi} on the last board burst (a few seconds, the windows freeze meanwhile) ...")
    best = {}
    for key, img in (("LEFT", proc_l), ("RIGHT", proc_r)):
        best[key] = next((ab for ab in cands if detect(img, ab, flags)[0]), None)
        logger.info(f"[scan] {key}: largest grid found = {best[key] if best[key] else 'none'}")
    if best["LEFT"] is None or best["RIGHT"] is None:
        miss = [k for k, v in best.items() if v is None]
        logger.warning(f"[scan] no board grid found at all in {' and '.join(miss)}. Look at that panel: are the "
                       f"squares clean, complete and inside the image? Try keys p / m / i, or a different distance.")
    elif best["LEFT"] == best["RIGHT"]:
        a, b = best["LEFT"]
        logger.info(f"[scan] both cameras agree: {a} x {b} inner corners -> "
                    f"rerun with  --rows {a} --cols {b}   ({a + 1} x {b + 1} squares)")
    else:
        big = max(best.values(), key=lambda ab: ab[0] * ab[1])
        logger.warning(f"[scan] cameras disagree (LEFT {best['LEFT']}, RIGHT {best['RIGHT']}). The larger one, "
                       f"{big}, is probably the real board; the other camera does not see the WHOLE board in "
                       f"this frame (cut off at an image edge?). Press 's' again on a burst where both panels show "
                       f"the complete board.")


# ----------------------------------------------------------------------------
# Config / IO helpers
# ----------------------------------------------------------------------------
def load_intrinsics(filename, logger):
    path = os.path.join("config", filename)
    if not os.path.exists(path):
        logger.error(f"{path} not found (run from the project root).")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    mtx = np.array(data.get("K", data.get("camera_matrix"))).reshape(3, 3).astype(np.float32)
    dist = np.array(data.get("D", data.get("dist_coeffs"))).astype(np.float32)
    return mtx, dist, data["width"], data["height"]


def save_points_json(objpoints, ipl, ipr, width, height, logger):
    os.makedirs("data_analysis", exist_ok=True)
    path = os.path.join("data_analysis", f"points_stereo_{datetime.now():%Y%m%d_%H%M%S}.json")
    data = {
        "width": width, "height": height,
        "objpoints": [op.tolist() for op in objpoints],
        "imgpoints_L": [ip.tolist() for ip in ipl],
        "imgpoints_R": [ip.tolist() for ip in ipr],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info(f"Stereo points saved for offline analysis: {path}")


def save_stereo_params(path, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F, width, height, logger):
    data = {
        "width": width, "height": height,
        "camera_left": {"K": mtx_l.tolist(), "D": dist_l.tolist()},
        "camera_right": {"K": mtx_r.tolist(), "D": dist_r.tolist()},
        "stereo": {"R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Stereo calibration parameters saved to: {path}")


def save_debug_frames(raw_l, raw_r, proc_l, proc_r, meta, logger):
    os.makedirs("debug_frames", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, img in (("L_raw", raw_l), ("R_raw", raw_r), ("L_detector_input", proc_l), ("R_detector_input", proc_r)):
        cv2.imwrite(os.path.join("debug_frames", f"{stamp}_{name}.png"), img)
    with open(os.path.join("debug_frames", f"{stamp}_settings.txt"), "w") as f:
        f.write(meta + "\n")
    logger.info(f"Saved debug images to debug_frames/{stamp}_*.png")


def run_calibration(objpoints, ipl, ipr, mtx_l, dist_l, mtx_r, dist_r, size, args, logger):
    logger.info("Starting stereo calibration with outlier rejection...")
    # R, T, E, F, perViewErrors must be passed (as None) - the Python binding of
    # stereoCalibrateExtended requires them; the original script left them out and crashed here.
    res = cv2.stereoCalibrateExtended(objpoints, ipl, ipr, mtx_l, dist_l, mtx_r, dist_r,
                                      size, None, None, None, None, None, flags=cv2.CALIB_FIX_INTRINSIC)
    # Newer OpenCV returns 12 values (adds rvecs/tvecs), older ones 10 -> take first and last only.
    ret, per_view = res[0], res[-1]
    errors = np.mean(per_view, axis=1).flatten()
    threshold = np.mean(errors) + np.std(errors)

    f_obj, f_l, f_r = [], [], []
    for i, err in enumerate(errors):
        if err < threshold:
            f_obj.append(objpoints[i]); f_l.append(ipl[i]); f_r.append(ipr[i])
        else:
            logger.info(f"Snap {i} discarded: RMS {err:.4f} > {threshold:.4f}")

    logger.info(f"Refining with {len(f_obj)} snapshots...")
    ret_f, m1, d1, m2, d2, R, T, E, F = cv2.stereoCalibrate(
        f_obj, f_l, f_r, mtx_l, dist_l, mtx_r, dist_r, size, flags=cv2.CALIB_FIX_INTRINSIC)
    logger.info(f"Final Stereo RMS: {ret_f:.4f} px (before outlier rejection: {ret:.4f})")
    logger.info(f"Baseline |T| = {float(np.linalg.norm(T)) * 1000:.1f} mm  "
                f"(the original rig measured about 119 mm - a very different number means a wrong square "
                f"size or a bad calibration)")

    os.makedirs("config", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("config", f"stereo_params_{stamp}.json")
    save_stereo_params(out, m1, d1, m2, d2, R, T, E, F, size[0], size[1], logger)
    if args.overwrite:
        live = os.path.join("config", "stereo_params.json")
        if os.path.exists(live):
            backup = os.path.join("config", f"stereo_params.backup_{stamp}.json")
            shutil.copy2(live, backup)
            logger.info(f"Backed up the previous file to {backup}")
        shutil.copy2(out, live)
        logger.info(f"Replaced {live}")
    else:
        logger.info("config/stereo_params.json was NOT changed. Copy the new file over it "
                    "(or rerun with --overwrite) when you are happy with the RMS.")


class Gui:
    """Thin wrapper so a missing/broken OpenCV GUI (e.g. opencv-python-headless) never kills the run."""

    def __init__(self, enabled, logger):
        self.enabled, self.logger = enabled, logger

    def _fail(self, e):
        self.logger.error(f"OpenCV window functions failed ({str(e).strip().splitlines()[0][:110]}). Is "
                          f"opencv-python-headless installed? Continuing WITHOUT windows - rely on the "
                          f"[status] / [hint] lines.")
        self.enabled = False

    def key(self):
        if not self.enabled:
            return 255
        try:
            return cv2.waitKey(1) & 0xFF
        except cv2.error as e:
            self._fail(e)
            return 255

    def show(self, name, img):
        if not self.enabled:
            return
        try:
            cv2.imshow(name, img)
        except cv2.error as e:
            self._fail(e)

    def close(self):
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ----------------------------------------------------------------------------
# Diagnosis
# ----------------------------------------------------------------------------
def diagnose(win, last_dt, args, logger, waiting_l=False, waiting_r=False):
    # A packet that is being held while the other side is late also proves that camera is alive.
    alive_l = win["pk_L"] > 0 or waiting_l
    alive_r = win["pk_R"] > 0 or waiting_r
    if not (alive_l and alive_r):
        who = "LEFT" if not alive_l else "RIGHT"
        if not alive_l and not alive_r:
            who = "BOTH cameras"
        logger.warning(f"[hint] No data from {who} in the last 5 s. Check connections/power, the serials in "
                       f"settings.py, and that no other program (Metavision Studio) has the camera open. "
                       f"Also look above for a 'Critical error' line from the sensor thread.")
    elif win["synced"] == 0:
        logger.warning(f"[hint] Both cameras stream, but their timestamps are never within {args.max_sync_us} us "
                       f"(last difference {last_dt} us). That points to the HARDWARE SYNC: check the sync cable "
                       f"between the boards and that RIGHT is master / LEFT is slave. "
                       f"(Quick test: rerun with --max-sync-us 40000 to see whether detection works at all.)")
    elif win["attempts"] == 0:
        logger.warning(f"[hint] Not one synced frame with >= {args.min_events} events in 5 s. The board page "
                       f"flips every 400 ms, so you should see ~2 bursts per second. Check: page is open and "
                       f"full-screen, screen brightness high, both cameras point at the screen and are in "
                       f"focus, and try --bias 0 (the +{S.BIAS_INCREMENT_STEREO} default was tuned for another "
                       f"setup) or lower --min-events.")
    elif win["attempts"] > 0 and win["both"] == 0:
        if win["found_L"] == 0 and win["found_R"] == 0:
            logger.warning("[hint] Board NOT found in either camera. Check in the window: (1) do you see clean "
                           "filled squares? (2) is --rows/--cols the number of INNER corners of your board? "
                           "Press 's' to scan sizes. (3) try keys p / m / i to change the image, and distance "
                           "20-35 cm so squares are >= ~8 px.")
        elif win["found_L"] == 0 or win["found_R"] == 0:
            side = "RIGHT" if win["found_L"] > 0 else "LEFT"
            logger.warning(f"[hint] Only one camera finds the board; {side} never does. The board must be fully "
                           f"visible in the overlap of both views. Compare the two panels.")
        else:
            logger.warning("[hint] Each camera finds the board, but never in the same frame. The board is probably "
                           "at the edge of the overlap area of the two views - move it toward the centre/farther.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stereo calibration with live debugging overlay")
    p.add_argument("--mode", choices=["calibrate", "analyze"], default="calibrate",
                   help="'calibrate' computes stereo params, 'analyze' only saves the points.")
    p.add_argument("--rows", type=int, default=S.CHECKERBOARD_ROWS, help="inner corners (default from settings.py)")
    p.add_argument("--cols", type=int, default=S.CHECKERBOARD_COLS, help="inner corners (default from settings.py)")
    p.add_argument("--square-mm", type=float, default=S.SQUARE_SIZE_MM, help="printed square size in mm")
    p.add_argument("--polarity", choices=POLARITY_MODES, default="ON",
                   help="events used for the image. Use ON or OFF with the flashing board (BOTH fills every square)")
    p.add_argument("--min-events", type=int, default=200,
                   help="a frame counts as a 'board burst' only if BOTH cameras have at least this many events")
    p.add_argument("--morph", choices=MORPH_MODES, default="close3")
    p.add_argument("--no-invert", action="store_true", help="do not invert the detector image")
    p.add_argument("--delta-t", type=int, default=S.STEREO_DELTA_T, help="accumulation window in us")
    p.add_argument("--bias", type=int, default=S.BIAS_INCREMENT_STEREO, help="bias_diff_on/off increment (0 = leave)")
    p.add_argument("--max-sync-us", type=int, default=S.MAX_SYNC_DIFF_US)
    p.add_argument("--min-snaps", type=int, default=S.MIN_REQUIRED_SNAPS)
    p.add_argument("--cooldown", type=float, default=S.COOLDOWN_SECONDS)
    p.add_argument("--no-exhaustive", action="store_true", help="faster detection (less robust)")
    p.add_argument("--scale", type=int, default=2, help="window zoom factor")
    p.add_argument("--headless", action="store_true", help="no windows (logs only)")
    p.add_argument("--overwrite", action="store_true", help="also replace config/stereo_params.json (backup kept)")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("stereo_calibration_debug")

    rows, cols = args.rows, args.cols
    pattern = (rows, cols)
    square_m = args.square_mm / 1000.0
    polarity = args.polarity
    morph = args.morph
    invert = not args.no_invert
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    if not args.no_exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE

    logger.info(f"OpenCV {cv2.__version__}, numpy {np.__version__}")
    logger.info(f"Serials: LEFT={S.SERIAL_LEFT}  RIGHT={S.SERIAL_RIGHT}")
    logger.info(f"Board: {rows} x {cols} INNER corners, square {args.square_mm} mm | "
                f"delta_t {args.delta_t} us | bias +{args.bias} | max sync {args.max_sync_us} us")
    logger.info(f"Image: polarity={polarity} morph={morph} invert={invert}")

    try:
        from src.utils.camera_streamer import EventReaderThread
    except ImportError as e:
        logger.error(f"Cannot import the Metavision Python bindings ({e}). "
                     f"Is the OpenEB/Metavision python package on PYTHONPATH for this interpreter?")
        sys.exit(1)

    mtx_l, dist_l, width, height = load_intrinsics("camera_left.json", logger)
    mtx_r, dist_r, _, _ = load_intrinsics("camera_right.json", logger)
    logger.info("Using intrinsics from config/camera_left.json and camera_right.json (FIXED during stereo "
                "calibration). If these came from other cameras, calibrate each camera first.")

    # Threads deliver ALL events; polarity is chosen live in this script.
    t_L = EventReaderThread(S.SERIAL_LEFT, args.delta_t, role="SLAVE_LEFT", logger=logger,
                            bias_increment=args.bias, filter_polarity=None)
    t_R = EventReaderThread(S.SERIAL_RIGHT, args.delta_t, role="MASTER_RIGHT", logger=logger,
                            bias_increment=args.bias, filter_polarity=None)
    t_L.start()
    time.sleep(0.5)  # let the slave arm before the master starts the clock
    t_R.start()

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_m

    objpoints, ipl, ipr = [], [], []
    point_mask = np.zeros((height, width), np.float32)
    last_cap = time.time()

    tot, win = Counter(), Counter()
    gui = Gui(not args.headless, logger)

    def bump(k):
        tot[k] += 1
        win[k] += 1

    pkt_l = pkt_r = None
    last_dt = 0
    det_ms = 0.0
    last_report = time.time()
    shown_any = False
    held = None                 # (panel_l, panel_r, time) of the last frame with a real burst of events
    burst_frames = None         # raw/proc images of that same burst (used by keys 's' and SPACE)

    logger.info("Running. Click a window and press q to stop, s to scan board sizes, SPACE to save images.")

    try:
        while True:
            if t_L.error or t_R.error:
                logger.error("A sensor thread reported an error (see 'Critical error' above). Stopping.")
                break
            if not (t_L.is_alive() and t_R.is_alive()):
                logger.error("A sensor thread has stopped. Stopping.")
                break

            # ---- keys -------------------------------------------------------
            key = gui.key()
            if key == ord("q"):
                break
            elif key == ord("p"):
                polarity = POLARITY_MODES[(POLARITY_MODES.index(polarity) + 1) % 3]
                logger.info(f"polarity -> {polarity}")
            elif key == ord("m"):
                morph = MORPH_MODES[(MORPH_MODES.index(morph) + 1) % 3]
                logger.info(f"morph -> {morph}")
            elif key == ord("i"):
                invert = not invert
                logger.info(f"invert -> {invert}")
            elif key == ord("s") and burst_frames is not None:
                scan_pattern_sizes(burst_frames[2], burst_frames[3], logger)
            elif key == ord("s"):
                logger.warning("[scan] no board burst captured yet - wait until the panels show a frame.")
            elif key == 32 and burst_frames is not None:
                save_debug_frames(*burst_frames,
                                  f"pattern={pattern} polarity={polarity} morph={morph} invert={invert} "
                                  f"delta_t={args.delta_t} bias={args.bias}", logger)

            # ---- periodic status -------------------------------------------
            now = time.time()
            if now - last_report >= 5.0:
                logger.info(f"[status] last 5s: packets L={win['pk_L']} R={win['pk_R']} | synced={win['synced']} "
                            f"unsynced_drops={win['unsynced']} | tried={win['attempts']} "
                            f"found L={win['found_L']} R={win['found_R']} both={win['both']} "
                            f"few_events={win['low_events']} | snaps={len(objpoints)}/{args.min_snaps} "
                            f"| detect {det_ms:.0f} ms")
                diagnose(win, last_dt, args, logger, pkt_l is not None, pkt_r is not None)
                win.clear()
                last_report = now

            # ---- fetch packets (never throw one away just because the other side is late)
            if pkt_l is None:
                try:
                    pkt_l = t_L.q.get(timeout=0.01); bump("pk_L")
                except Empty:
                    pass
            if pkt_r is None:
                try:
                    pkt_r = t_R.q.get(timeout=0.01); bump("pk_R")
                except Empty:
                    pass
            if pkt_l is None or pkt_r is None:
                if gui.enabled and not shown_any:
                    ph = np.zeros((120, 640, 3), np.uint8)
                    put(ph, "Waiting for both cameras ...", (10, 40), WHITE, 0.7, 2)
                    put(ph, f"packets so far  L={tot['pk_L']}  R={tot['pk_R']}", (10, 80))
                    gui.show("Detector input", ph)
                continue

            ts_l, ts_r = pkt_l[1], pkt_r[1]
            last_dt = ts_l - ts_r
            synced = abs(last_dt) <= args.max_sync_us

            # ---- build images ---------------------------------------------
            ev_l = select_events(pkt_l[0], polarity)
            ev_r = select_events(pkt_r[0], polarity)
            raw_l = events_to_image(ev_l, width, height)
            raw_r = events_to_image(ev_r, width, height)
            proc_l = preprocess(raw_l, morph, invert)
            proc_r = preprocess(raw_r, morph, invert)
            fill_l = 100.0 * np.count_nonzero(raw_l) / raw_l.size
            fill_r = 100.0 * np.count_nonzero(raw_r) / raw_r.size

            status_l = status_r = "NOT SYNCED"
            corners_l = corners_r = None

            if synced:
                bump("synced")
                if ev_l.size >= args.min_events and ev_r.size >= args.min_events:
                    bump("attempts")
                    burst_frames = (raw_l, raw_r, proc_l, proc_r)
                    t0 = time.perf_counter()
                    ok_l, corners_l = detect(proc_l, pattern, flags)
                    ok_r, corners_r = detect(proc_r, pattern, flags)
                    det_ms = (time.perf_counter() - t0) * 1000.0
                    status_l = "FOUND" if ok_l else "NOT FOUND"
                    status_r = "FOUND" if ok_r else "NOT FOUND"
                    if ok_l: bump("found_L")
                    if ok_r: bump("found_R")
                    if ok_l and ok_r:
                        bump("both")
                        if now - last_cap >= args.cooldown:
                            objpoints.append(objp)
                            ipl.append(corners_l)
                            ipr.append(corners_r)
                            last_cap = now
                            for c in corners_l:
                                cv2.circle(point_mask, (int(c[0][0]), int(c[0][1])), 2, 1.0, -1)
                            logger.info(f"Snap {len(objpoints)} captured (sync diff {abs(last_dt)} us)")
                else:
                    bump("low_events")
                    status_l = status_r = "IDLE"   # no flip in this 30 ms window: normal for a blinking board
                pkt_l = pkt_r = None
            else:
                # drop whichever packet is older and try to line up with the next one
                bump("unsynced")
                if ts_l < ts_r:
                    pkt_l = None
                else:
                    pkt_r = None

            # ---- display ---------------------------------------------------
            if gui.enabled:
                pl = make_panel("LEFT", proc_l, status_l, corners_l, pattern, ev_l.size, fill_l, args.scale)
                pr = make_panel("RIGHT", proc_r, status_r, corners_r, pattern, ev_r.size, fill_r, args.scale)
                # Hold the last burst on screen: a blinking board is only visible for a few ms per flip.
                if status_l in ("FOUND", "NOT FOUND") or held is None:
                    held = (pl, pr, now)
                pl, pr, held_t = held
                age = now - held_t
                cd = max(0.0, args.cooldown - (now - last_cap))
                bar = [
                    (f"SYNC {'OK' if synced else 'LOST'}  dt={last_dt:+d} us (limit {args.max_sync_us})   "
                     f"detect {det_ms:.0f} ms   panels show the last board burst, {age:.1f}s ago",
                     (GREEN if synced else RED) if age < 1.5 else ORANGE),
                    (f"snaps {len(objpoints)}/{args.min_snaps}   next snap in {cd:.1f}s   "
                     f"found so far: L {tot['found_L']}  R {tot['found_R']}  both {tot['both']}  "
                     f"of {tot['attempts']} tries", WHITE),
                    (f"board {rows}x{cols}   pol={polarity}  morph={morph}  invert={'on' if invert else 'off'}"
                     f"   keys: q quit | p pol | m morph | i invert | s scan | SPACE save", GREY),
                ]
                gui.show("Detector input", make_display(pl, pr, bar))
                heat = cv2.resize(generate_point_heatmap(point_mask), None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_NEAREST)
                put(heat, f"Snaps: {len(objpoints)}", (8, 20), WHITE, 0.6, 2)
                gui.show("Coverage heatmap", heat)
                shown_any = True

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info(f"TOTAL: packets L={tot['pk_L']} R={tot['pk_R']} | synced={tot['synced']} | tried={tot['attempts']} "
                    f"found L={tot['found_L']} R={tot['found_R']} both={tot['both']} | snaps={len(objpoints)}")
        logger.info("Shutting down sensor threads...")
        t_R.stop(); t_L.stop()
        t_R.join(timeout=5); t_L.join(timeout=5)
        gui.close()

    if len(objpoints) >= args.min_snaps:
        save_points_json(objpoints, ipl, ipr, width, height, logger)
        if args.mode == "calibrate":
            try:
                run_calibration(objpoints, ipl, ipr, mtx_l, dist_l, mtx_r, dist_r, (width, height), args, logger)
            except cv2.error as e:
                logger.error(f"OpenCV calibration failed: {e}")
        else:
            logger.info("Mode 'analyze' complete. Parameters were not updated.")
    else:
        logger.error(f"Not enough snaps captured ({len(objpoints)}/{args.min_snaps}). "
                     f"See the [hint] lines above for the failing stage.")


if __name__ == "__main__":
    main()
