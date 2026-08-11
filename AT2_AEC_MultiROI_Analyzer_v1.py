import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi
import io

# ============================================================
# CONSTANTS
# ============================================================

PIXEL_SIZE_UM = 0.2128
PIXEL_SIZE_MM = PIXEL_SIZE_UM / 1000.0
PIXEL_AREA_MM2 = PIXEL_SIZE_MM ** 2


# ============================================================
# HELPERS
# ============================================================

def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def is_near(p1, p2, r=10):
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2)) <= r


def nearest_point_index(points, x, y, max_dist):
    if not points:
        return None
    d = [np.hypot(px - x, py - y) for px, py in points]
    i = int(np.argmin(d))
    return i if d[i] <= max_dist else None


# ============================================================
# HSV CALIBRATION (ORIGINAL IMAGE)
# ============================================================

def compute_hsv_range(points, hsv_img, radius=5):
    if not points:
        return None

    vals = []
    h_img, w_img = hsv_img.shape[:2]

    for x, y in points:
        x0 = max(0, x - radius)
        x1 = min(w_img, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h_img, y + radius + 1)
        region = hsv_img[y0:y1, x0:x1]
        if region.size:
            vals.append(region.reshape(-1, 3))

    if not vals:
        return None

    vals = np.vstack(vals)
    h, s, v = vals[:, 0], vals[:, 1], vals[:, 2]

    # circular mean hue
    h_rad = h * np.pi / 90.0
    angle = np.arctan2(np.mean(np.sin(h_rad)), np.mean(np.cos(h_rad)))
    if angle < 0:
        angle += 2 * np.pi
    h_center = (np.degrees(angle) / 2.0) % 180.0

    s_center = float(np.median(s))
    v_center = float(np.median(v))

    tol_h, tol_s, tol_v = 12, 45, 45

    hmin = int(round(h_center - tol_h))
    hmax = int(round(h_center + tol_h))
    smin = max(0, int(round(s_center - tol_s)))
    smax = min(255, int(round(s_center + tol_s)))
    vmin = max(0, int(round(v_center - tol_v)))
    vmax = min(255, int(round(v_center + tol_v)))

    return (hmin, hmax, smin, smax, vmin, vmax)


def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        return cv2.inRange(hsv_img,
                           np.array([hmin, smin, vmin]),
                           np.array([hmax, smax, vmax]))
    low = cv2.inRange(hsv_img,
                      np.array([0, smin, vmin]),
                      np.array([hmax, smax, vmax]))
    high = cv2.inRange(hsv_img,
                       np.array([hmin, smin, vmin]),
                       np.array([179, smax, vmax]))
    return cv2.bitwise_or(low, high)


# ============================================================
# AEC DETECTION (ORIGINAL IMAGE)
# ============================================================

def detect_aec_centers(image_rgb, hsv_range, blur_kernel=5, min_area=25, morph_size=3):
    proc = image_rgb.copy()

    if blur_kernel > 1:
        k = ensure_odd(blur_kernel)
        proc = cv2.GaussianBlur(proc, (k, k), 0)

    hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    mask = apply_hue_wrap(hsv, *hsv_range)

    if morph_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))
        centers.append((cx, cy))

    return centers, mask


# ============================================================
# DBSCAN (ORIGINAL COORDINATES)
# ============================================================

def dbscan_analysis(points, eps_um, min_samples):
    if len(points) == 0:
        return {"labels": [], "cluster_count": 0, "cluster_sizes": [], "cluster_centers": []}

    eps_px = eps_um / PIXEL_SIZE_UM
    xy = np.asarray(points, float)

    model = DBSCAN(eps=eps_px, min_samples=min_samples)
    labels = model.fit_predict(xy)

    clusters = sorted(set(labels) - {-1})
    sizes = [int(np.sum(labels == c)) for c in clusters]
    centers = [xy[labels == c].mean(axis=0) for c in clusters]

    return {
        "labels": labels.tolist(),
        "cluster_count": len(clusters),
        "cluster_sizes": sizes,
        "cluster_centers": centers
    }


# ============================================================
# VORONOI (ONLY ON ROI FINISH)
# ============================================================

def voronoi_areas_mm2(points, image_shape):
    if len(points) < 3:
        return [np.nan] * len(points)

    h, w = image_shape[:2]
    pts = np.asarray(points, float)

    try:
        vor = Voronoi(pts)
    except Exception:
        return [np.nan] * len(points)

    def clip(poly, axis, value, keep_greater):
        out = []
        for i in range(len(poly)):
            A, B = poly[i], poly[(i + 1) % len(poly)]
            a_in = (A[axis] >= value) if keep_greater else (A[axis] <= value)
            b_in = (B[axis] >= value) if keep_greater else (B[axis] <= value)
            if a_in and b_in:
                out.append(B)
            elif a_in and not b_in:
                t = (value - A[axis]) / (B[axis] - A[axis])
                out.append(A + t * (B - A))
            elif not a_in and b_in:
                t = (value - A[axis]) / (B[axis] - A[axis])
                out.append(A + t * (B - A))
                out.append(B)
        return out

    def area(poly):
        if len(poly) < 3:
            return 0.0
        p = np.asarray(poly)
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    areas = []
    for i in range(len(pts)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            c = pts[i]
            s = max(w, h) * 10
            poly = np.array([[c[0]-s, c[1]-s], [c[0]+s, c[1]-s],
                             [c[0]+s, c[1]+s], [c[0]-s, c[1]+s]])
        else:
            poly = np.asarray([vor.vertices[v] for v in region])

        poly = clip(poly, 0, 0, True)
        poly = clip(poly, 0, w-1, False)
        poly = clip(poly, 1, 0, True)
        poly = clip(poly, 1, h-1, False)

        areas.append(area(poly) * PIXEL_AREA_MM2)

    return areas


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="AT2 / AEC Multi-ROI Analyzer", layout="wide")
st.title("🧬 AT2 / AEC Multi-ROI Spatial Analyzer")

# SESSION STATE
defaults = {
    "slide_no": "",
    "current_roi_index": 0,
    "roi_points": [],
    "auto_points": [],
    "aec_cal_points": [],
    "history": [],
    "current_calibration": None,
    "finished_rois": [],
    "slide_summary": [],
    "slide_objects": [],
    "split_stage": 0,
    "split_clicks": [],
    "split_selected": None
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Slide")
slide_no = st.sidebar.text_input("Slide-Nummer", st.session_state.slide_no)
st.session_state.slide_no = slide_no

st.sidebar.header("AEC-Erkennung")
blur_kernel = ensure_odd(st.sidebar.slider("Blur", 1, 21, 5, step=2))
min_area = st.sidebar.number_input("Min Area (px²)", 1, 10000, 25)
morph_size = ensure_odd(st.sidebar.slider("Morph Size", 1, 11, 3, step=2))
calib_radius = st.sidebar.slider("Kalibrierungsradius", 1, 15, 5)

st.sidebar.header("Manuelle Korrektur")
edit_radius = st.sidebar.slider("Auswahlradius", 3, 30, 10)
circle_radius = st.sidebar.slider("Anzeige-Radius", 2, 20, 6)

st.sidebar.header("Clusteranalyse")
dbscan_eps_um = st.sidebar.number_input("DBSCAN eps (µm)", 1.0, 500.0, 50.0)
dbscan_min_samples = st.sidebar.number_input("DBSCAN min_samples", 1, 50, 3)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader("ROI-Bilder hochladen", type=["jpg", "png", "tif"], accept_multiple_files=True)

if not uploaded_files:
    st.stop()

# Reset batch if new files
file_key = tuple((f.name, f.size) for f in uploaded_files)
if file_key != st.session_state.get("file_key"):
    st.session_state.file_key = file_key
    st.session_state.current_roi_index = 0
    st.session_state.finished_rois = []
    st.session_state.slide_summary = []
    st.session_state.slide_objects = []
    st.session_state.roi_points = []
    st.session_state.auto_points = []
    st.session_state.aec_cal_points = []
    st.session_state.history = []
    st.session_state.split_stage = 0


# ============================================================
# LOAD CURRENT ROI
# ============================================================

idx = st.session_state.current_roi_index
current_file = uploaded_files[idx]

image_orig = np.array(Image.open(io.BytesIO(current_file.getvalue())).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]

display_width = st.sidebar.slider("Bildbreite", 500, 2000, 1200)
scale = display_width / W_orig
display_height = int(H_orig * scale)

image_disp = cv2.resize(image_orig, (display_width, display_height), interpolation=cv2.INTER_AREA)

st.subheader(f"Slide {slide_no} — ROI {idx+1}/{len(uploaded_files)}: {current_file.name}")

# ============================================================
# CALIBRATION
# ============================================================

st.markdown("### AEC-Kalibrierung")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Kalibrierpunkte zurücksetzen"):
        st.session_state.aec_cal_points = []
        st.experimental_rerun()

with c2:
    if st.button("AEC erkennen"):
        if len(st.session_state.aec_cal_points) == 0:
            st.warning("Bitte Kalibrierpunkte setzen.")
        else:
            # scale calibration points back to original
            cal_orig = [(int(x/scale), int(y/scale)) for x, y in st.session_state.aec_cal_points]
            hsv_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)
            hsv_range = compute_hsv_range(cal_orig, hsv_orig, radius=calib_radius)

            if hsv_range is None:
                st.error("Kalibrierung fehlgeschlagen.")
            else:
                st.session_state.current_calibration = hsv_range
                detected, mask = detect_aec_centers(image_orig, hsv_range,
                                                    blur_kernel, min_area, morph_size)
                st.session_state.auto_points = detected
                st.session_state.roi_points = detected.copy()
                st.session_state.history = []
                st.success(f"{len(detected)} AEC erkannt.")

with c3:
    if st.button("ROI neu starten"):
        st.session_state.roi_points = []
        st.session_state.auto_points = []
        st.session_state.aec_cal_points = []
        st.session_state.history = []
        st.session_state.split_stage = 0
        st.experimental_rerun()


# ============================================================
# MODE
# ============================================================

MODES = ["👁 Kontrolle", "➕ Hinzufügen", "🗑 Löschen", "✂ Split", "🎯 Kalibrierpunkt"]
mode = st.radio("Modus", MODES, horizontal=True)

# Reset split mode if user switches mode
if mode != "✂ Split":
    st.session_state.split_stage = 0
    st.session_state.split_clicks = []
    st.session_state.split_selected = None


# ============================================================
# DISPLAY IMAGE
# ============================================================

marked = image_disp.copy()

# calibration points
for x, y in st.session_state.aec_cal_points:
    cv2.circle(marked, (x, y), circle_radius, (0, 255, 255), -1)

# final points
for i, (ox, oy) in enumerate(st.session_state.roi_points, start=1):
    dx, dy = int(ox * scale), int(oy * scale)
    cv2.circle(marked, (dx, dy), circle_radius, (0, 0, 255), 2)
    cv2.putText(marked, str(i), (dx+circle_radius+2, dy-circle_radius-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

coords = streamlit_image_coordinates(Image.fromarray(marked),
                                     key=f"roi_canvas_{slide_no}_{idx}",
                                     width=display_width)

# ============================================================
# CLICK LOGIC
# ============================================================

if coords:
    x_disp, y_disp = coords["x"], coords["y"]
    x_orig, y_orig = int(x_disp / scale), int(y_disp / scale)

    # calibration
    if mode == "🎯 Kalibrierpunkt":
        st.session_state.aec_cal_points.append((x_disp, y_disp))
        st.experimental_rerun()

    # add
    elif mode == "➕ Hinzufügen":
        st.session_state.history.append(st.session_state.roi_points.copy())
        st.session_state.roi_points.append((x_orig, y_orig))
        st.experimental_rerun()

    # delete
    elif mode == "🗑 Löschen":
        i = nearest_point_index(st.session_state.roi_points, x_orig, y_orig, edit_radius)
        if i is not None:
            st.session_state.history.append(st.session_state.roi_points.copy())
            st.session_state.roi_points.pop(i)
            st.experimental_rerun()

    # split
    elif mode == "✂ Split":
        if st.session_state.split_stage == 0:
            sel = nearest_point_index(st.session_state.roi_points, x_orig, y_orig, edit_radius)
            if sel is None:
                st.warning("Bitte auf die verschmolzene Zelle klicken.")
            else:
                st.session_state.split_selected = sel
                st.session_state.split_stage = 1
                st.info("Jetzt zwei neue Zentren anklicken.")
                st.experimental_rerun()

        elif st.session_state.split_stage == 1:
            old = st.session_state.roi_points[st.session_state.split_selected]
            if not is_near(old, (x_orig, y_orig), 5):
                st.session_state.split_clicks.append((x_orig, y_orig))
                if len(st.session_state.split_clicks) == 2:
                    st.session_state.history.append(st.session_state.roi_points.copy())
                    st.session_state.roi_points.pop(st.session_state.split_selected)
                    st.session_state.roi_points.extend(st.session_state.split_clicks)
                    st.session_state.split_stage = 0
                    st.session_state.split_clicks = []
                    st.session_state.split_selected = None
