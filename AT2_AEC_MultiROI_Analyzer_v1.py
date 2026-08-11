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
    return np.linalg.norm(np.asarray(p1, float) - np.asarray(p2, float)) <= float(r)


def nearest_point_index(points, x, y, max_dist):
    if not points:
        return None
    d = [np.hypot(float(px) - x, float(py) - y) for px, py in points]
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
        x0 = max(0, int(x) - radius)
        x1 = min(w_img, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(h_img, int(y) + radius + 1)
        region = hsv_img[y0:y1, x0:x1]
        if region.size:
            vals.append(region.reshape(-1, 3))

    if not vals:
        return None

    vals = np.vstack(vals)
    h = vals[:, 0].astype(float)
    s = vals[:, 1].astype(float)
    v = vals[:, 2].astype(float)

    h_rad = h * np.pi / 90.0
    sin_mean = np.mean(np.sin(h_rad))
    cos_mean = np.mean(np.cos(h_rad))
    angle = np.arctan2(sin_mean, cos_mean)
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
    hmin, hmax, smin, smax, vmin, vmax = map(int, (hmin, hmax, smin, smax, vmin, vmax))
    if hmin <= hmax:
        return cv2.inRange(
            hsv_img,
            np.array([hmin, smin, vmin], dtype=np.uint8),
            np.array([hmax, smax, vmax], dtype=np.uint8)
        )
    low = cv2.inRange(
        hsv_img,
        np.array([0, smin, vmin], dtype=np.uint8),
        np.array([hmax, smax, vmax], dtype=np.uint8)
    )
    high = cv2.inRange(
        hsv_img,
        np.array([hmin, smin, vmin], dtype=np.uint8),
        np.array([179, smax, vmax], dtype=np.uint8)
    )
    return cv2.bitwise_or(low, high)


# ============================================================
# AEC DETECTION (ORIGINAL IMAGE)
# ============================================================

def detect_aec_centers(
    image_rgb,
    hsv_range,
    blur_kernel=5,
    min_area=25,
    morph_size=3
):
    proc = image_rgb.copy()

    if blur_kernel > 1:
        k = ensure_odd(blur_kernel)
        proc = cv2.GaussianBlur(proc, (k, k), 0)

    hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    mask = apply_hue_wrap(hsv, *hsv_range)

    if morph_size > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_size, morph_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours_info = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < float(min_area):
            continue
        M = cv2.moments(c)
        if M.get("m00", 0) == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((int(round(cx)), int(round(cy))))

    return centers, mask


# ============================================================
# DBSCAN (ORIGINAL COORDINATES)
# ============================================================

def dbscan_analysis(points, eps_um, min_samples):
    if len(points) == 0:
        return {
            "labels": [],
            "cluster_count": 0,
            "cluster_sizes": [],
            "cluster_centers": []
        }

    eps_px = float(eps_um) / PIXEL_SIZE_UM
    xy = np.asarray(points, dtype=float)

    model = DBSCAN(eps=float(eps_px), min_samples=int(min_samples))
    labels = model.fit_predict(xy)

    unique = sorted(set(labels))
    clusters = [x for x in unique if x != -1]

    sizes = []
    centers = []
    for label in clusters:
        pts = xy[labels == label]
        sizes.append(int(len(pts)))
        centers.append(pts.mean(axis=0))

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
    pts = np.asarray(points, dtype=float)

    try:
        vor = Voronoi(pts)
    except Exception:
        return [np.nan] * len(points)

    def clip_polygon(poly, axis, value, keep_greater):
        if not poly:
            return []
        result = []
        for i in range(len(poly)):
            A = poly[i]
            B = poly[(i + 1) % len(poly)]
            a_inside = (A[axis] >= value) if keep_greater else (A[axis] <= value)
            b_inside = (B[axis] >= value) if keep_greater else (B[axis] <= value)
            if a_inside and b_inside:
                result.append(B)
            elif a_inside and not b_inside:
                denom = B[axis] - A[axis]
                if denom != 0:
                    t = (value - A[axis]) / denom
                    result.append(A + t * (B - A))
            elif not a_inside and b_inside:
                denom = B[axis] - A[axis]
                if denom != 0:
                    t = (value - A[axis]) / denom
                    result.append(A + t * (B - A))
                    result.append(B)
        return result

    def polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        p = np.asarray(poly)
        return 0.5 * abs(
            np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
            np.dot(p[:, 1], np.roll(p[:, 0], -1))
        )

    areas_px2 = []
    for i in range(len(pts)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        if not region or -1 in region:
            center = pts[i]
            scale = max(w, h) * 10.0
            poly = np.array([
                [center[0] - scale, center[1] - scale],
                [center[0] + scale, center[1] - scale],
                [center[0] + scale, center[1] + scale],
                [center[0] - scale, center[1] + scale]
            ], dtype=float)
        else:
            poly = np.asarray([vor.vertices[v] for v in region], dtype=float)

        poly_list = [p for p in poly]
        poly_list = clip_polygon(poly_list, 0, 0, True)
        poly_list = clip_polygon(poly_list, 0, w - 1, False)
        poly_list = clip_polygon(poly_list, 1, 0, True)
        poly_list = clip_polygon(poly_list, 1, h - 1, False)

        areas_px2.append(polygon_area(poly_list))

    return [a * PIXEL_AREA_MM2 for a in areas_px2]


# ============================================================
# ROI SUMMARY / OBJECTS
# ============================================================

def nearest_neighbor_stats(points):
    if len(points) < 2:
        return np.nan, np.nan

    pts = np.asarray(points, dtype=float)
    distances = []
    for i in range(len(pts)):
        d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
        d[i] = np.inf
        distances.append(np.min(d))
    distances = np.asarray(distances) * PIXEL_SIZE_UM
    return float(np.mean(distances)), float(np.median(distances))


def calculate_roi_summary(
    slide_no,
    roi_name,
    points,
    image_shape,
    dbscan_eps_um,
    dbscan_min_samples
):
    h, w = image_shape[:2]
    roi_area_mm2 = (w * h) * PIXEL_AREA_MM2

    count = len(points)
    density = count / roi_area_mm2 if roi_area_mm2 > 0 else np.nan

    db = dbscan_analysis(points, dbscan_eps_um, dbscan_min_samples)
    cluster_count = db["cluster_count"]
    cluster_density = (
        cluster_count / roi_area_mm2
        if roi_area_mm2 > 0 else np.nan
    )
    sizes = db["cluster_sizes"]

    vor_areas = voronoi_areas_mm2(points, image_shape)
    valid_vor = np.asarray(
        [x for x in vor_areas if np.isfinite(x) and x > 0],
        dtype=float
    )

    if len(valid_vor):
        mean_vor = float(np.mean(valid_vor))
        sd_vor = float(np.std(valid_vor, ddof=1)) if len(valid_vor) > 1 else 0.0
        cv_vor = sd_vor / mean_vor if mean_vor > 0 else np.nan
        median_vor = float(np.median(valid_vor))
    else:
        mean_vor = np.nan
        sd_vor = np.nan
        cv_vor = np.nan
        median_vor = np.nan

    mean_nn, median_nn = nearest_neighbor_stats(points)

    return {
        "Slide": slide_no,
        "ROI": roi_name,
        "ROI_Width_px": w,
        "ROI_Height_px": h,
        "ROI_Area_mm2": roi_area_mm2,
        "AEC_Count": count,
        "AEC_per_mm2": density,
        "DBSCAN_eps_um": dbscan_eps_um,
        "DBSCAN_min_samples": dbscan_min_samples,
        "Cluster_Count": cluster_count,
        "Clusters_per_mm2": cluster_density,
        "Mean_Cluster_Size": float(np.mean(sizes)) if sizes else np.nan,
        "Median_Cluster_Size": float(np.median(sizes)) if sizes else np.nan,
        "Max_Cluster_Size": int(max(sizes)) if sizes else 0,
        "Mean_Voronoi_Area_mm2": mean_vor,
        "SD_Voronoi_Area_mm2": sd_vor,
        "CV_Voronoi_Area": cv_vor,
        "Median_Voronoi_Area_mm2": median_vor,
        "Mean_Nearest_Neighbor_um": mean_nn,
        "Median_Nearest_Neighbor_um": median_nn
    }


def make_objects_dataframe(slide_no, roi_name, points, image_shape):
    rows = []
    vor_areas = voronoi_areas_mm2(points, image_shape)

    for i, (x, y) in enumerate(points, start=1):
        row = {
            "Slide": slide_no,
            "ROI": roi_name,
            "Cell_ID": i,
            "X_px": int(x),
            "Y_px": int(y),
            "X_um": float(x * PIXEL_SIZE_UM),
            "Y_um": float(y * PIXEL_SIZE_UM),
            "Source": "final",
            "Voronoi_Area_mm2": vor_areas[i - 1] if i - 1 < len(vor_areas) else np.nan
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="AT2 / AEC Multi-ROI Spatial Analyzer v2",
    layout="wide"
)

st.title("🧬 AT2 / AEC Multi-ROI Spatial Analyzer v2")

defaults = {
    "slide_no": "",
    "current_roi_index": 0,
    "roi_points": [],
    "auto_points": [],
    "aec_cal_points": [],
    "history": [],
    "current_calibration": None,
    "slide_summary": [],
    "slide_objects": [],
    "finished_rois": [],
    "display_width": 1200,
    "last_file_key": None,
    "split_stage": 0,
    "split_clicks": [],
    "split_selected": None,
    "last_click": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_current_roi():
    for key in [
        "roi_points",
        "auto_points",
        "aec_cal_points",
        "history",
        "current_calibration",
        "split_stage",
        "split_clicks",
        "split_selected",
        "last_click"
    ]:
        st.session_state[key] = [] if isinstance(defaults[key], list) else None
    st.session_state.roi_finished = False


def snapshot():
    return [(int(x), int(y)) for x, y in st.session_state.roi_points]


def undo():
    hist = st.session_state.history
    if hist:
        st.session_state.roi_points = hist.pop()
        st.session_state.split_stage = 0
        st.session_state.split_clicks = []
        st.session_state.split_selected = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("1. Slide")

slide_input = st.sidebar.text_input(
    "Slide-Nummer",
    value=st.session_state.slide_no,
    placeholder="z.B. S001"
)
st.session_state.slide_no = slide_input

st.sidebar.markdown("---")
st.sidebar.header("2. Automatische AEC-Erkennung")

blur_kernel = ensure_odd(
    st.sidebar.slider("Blur", 1, 21, 5, step=2)
)
min_area = st.sidebar.number_input(
    "Mindestfläche der AEC-Kontur (px²)",
    min_value=1,
    max_value=10000,
    value=25,
    step=5
)
morph_size = ensure_odd(
    st.sidebar.slider("Morphologie", 1, 11, 3, step=2)
)
calib_radius = st.sidebar.slider(
    "AEC-Kalibrierungsradius",
    min_value=1,
    max_value=15,
    value=5
)

st.sidebar.markdown("---")
st.sidebar.header("3. Manuelle Korrektur")

edit_radius = st.sidebar.slider(
    "Auswahlradius",
    min_value=3,
    max_value=30,
    value=10
)
circle_radius = st.sidebar.slider(
    "Anzeige-Radius",
    min_value=2,
    max_value=20,
    value=6
)

st.sidebar.markdown("---")
st.sidebar.header("4. Clusteranalyse")

dbscan_eps_um = st.sidebar.number_input(
    "DBSCAN eps (µm)",
    min_value=1.0,
    max_value=500.0,
    value=50.0,
    step=1.0
)
dbscan_min_samples = st.sidebar.number_input(
    "DBSCAN min_samples",
    min_value=1,
    max_value=50,
    value=3,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Kalibrierung: 1 px = 0,2128 µm\n\n"
    "Hämatoxylin wird nicht analysiert."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "📂 Mehrere ROI-Bilder hochladen",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Bitte mehrere ROI-Bilder eines Slides hochladen.")
    st.stop()

if not st.session_state.slide_no.strip():
    st.warning("Bitte zuerst eine Slide-Nummer eingeben.")
    st.stop()

file_key = tuple((f.name, f.size) for f in uploaded_files)
if file_key != st.session_state.last_file_key:
    st.session_state.last_file_key = file_key
    st.session_state.current_roi_index = 0
    st.session_state.slide_summary = []
    st.session_state.slide_objects = []
    st.session_state.finished_rois = []
    reset_current_roi()


# ============================================================
# CURRENT ROI
# ============================================================

idx = st.session_state.current_roi_index
if idx >= len(uploaded_files):
    idx = len(uploaded_files) - 1
    st.session_state.current_roi_index = idx

current_file = uploaded_files[idx]

st.subheader(
    f"Slide {st.session_state.slide_no} — "
    f"ROI {idx + 1}/{len(uploaded_files)}: {current_file.name}"
)

file_bytes = current_file.getvalue()
image_orig = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]

display_width = st.sidebar.slider(
    "Bildbreite",
    500,
    2000,
    min(st.session_state.display_width, 2000),
    100
)
st.session_state.display_width = display_width

scale = display_width / W_orig
display_height = max(1, int(round(H_orig * scale)))

image_disp = cv2.resize(
    image_orig,
    (display_width, display_height),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# CALIBRATION / AUTO DETECTION
# ============================================================

st.markdown("### AEC-Kalibrierung")

cal_col1, cal_col2, cal_col3 = st.columns(3)

with cal_col1:
    if st.button("🎯 AEC-Kalibrierpunkte zurücksetzen"):
        st.session_state.aec_cal_points = []
        st.experimental_rerun()

with cal_col2:
    if st.button("🧬 AEC erkennen"):
        if len(st.session_state.aec_cal_points) < 1:
            st.warning("Bitte mindestens einen AEC-Kalibrierpunkt setzen.")
        else:
            cal_orig = [
                (int(x / scale), int(y / scale))
                for x, y in st.session_state.aec_cal_points
            ]
            hsv_orig = cv2.cvtColor(image_orig, cv2.COLOR_RGB2HSV)
            hsv_range = compute_hsv_range(
                cal_orig,
                hsv_orig,
                radius=calib_radius
            )

            if hsv_range is None:
                st.error("AEC-Kalibrierung fehlgeschlagen.")
            else:
                st.session_state.current_calibration = hsv_range
                detected, mask = detect_aec_centers(
                    image_orig,
                    hsv_range,
                    blur_kernel=blur_kernel,
                    min_area=min_area,
                    morph_size=morph_size
                )
                st.session_state.auto_points = list(detected)
                st.session_state.roi_points = list(detected)
                st.session_state.history = []
                st.session_state.roi_finished = False
                st.success(
                    f"{len(st.session_state.roi_points)} "
                    f"AEC-Kandidaten erkannt."
                )

with cal_col3:
    if st.button("🔄 ROI-Erkennung neu starten"):
        reset_current_roi()
        st.experimental_rerun()


# ============================================================
# MODE
# ============================================================

MODES = [
    "👁️ Kontrolle",
    "➕ Zelle hinzufügen",
    "🗑️ Zelle löschen",
    "✂️ Siamese Zelle splitten",
    "🎯 AEC-Kalibrierpunkt"
]

mode = st.radio(
    "Korrekturmodus",
    MODES,
    horizontal=True
)

if mode != "✂️ Siamese Zelle splitten":
    st.session_state.split_stage = 0
    st.session_state.split_clicks = []
    st.session_state.split_selected = None


# ============================================================
# DISPLAY IMAGE WITH POINTS
# ============================================================

marked = image_disp.copy()

for x_disp, y_disp in st.session_state.aec_cal_points:
    cv2.circle(
        marked,
        (int(x_disp), int(y_disp)),
        max(3, circle_radius),
        (0, 255, 255),
        -1
    )

for i, (x_orig, y_orig) in enumerate(st.session_state.roi_points, start=1):
    x_disp = int(x_orig * scale)
    y_disp = int(y_orig * scale)
    cv2.circle(
        marked,
        (x_disp, y_disp),
        circle_radius,
        (0, 0, 255),
        2
    )
    cv2.putText(
        marked,
        str(i),
        (x_disp + circle_radius + 2, y_disp - circle_radius - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 0),
        1,
        cv2.LINE_AA
    )

coords = streamlit_image_coordinates(
    Image.fromarray(marked),
    key=f"roi_canvas_{st.session_state.slide_no}_{idx}",
    width=display_width
)


# ============================================================
# CLICK LOGIC (DEBOUNCED)
# ============================================================

if coords:
    click = (coords["x"], coords["y"])
    if click != st.session_state.last_click:
        st.session_state.last_click = click

        x_disp = int(coords["x"])
        y_disp = int(coords["y"])
        x_orig = int(x_disp / scale)
        y_orig = int(y_disp / scale)

        if mode == "🎯 AEC-Kalibrierpunkt":
            st.session_state.aec_cal_points.append((x_disp, y_disp))
            st.experimental_rerun()

        elif mode == "➕ Zelle hinzufügen":
            st.session_state.history.append(snapshot())
            st.session_state.roi_points.append((x_orig, y_orig))
            st.experimental_rerun()

        elif mode == "🗑️ Zelle löschen":
            i = nearest_point_index(
                st.session_state.roi_points,
                x_orig, y_orig,
                edit_radius
            )
            if i is not None:
                st.session_state.history.append(snapshot())
                removed = st.session_state.roi_points.pop(i)
                st.toast(f"Zelle {i + 1} gelöscht: {removed}")
                st.experimental_rerun()
            else:
                st.warning("Keine Zelle in der Nähe des Klicks.")

        elif mode == "✂️ Siamese Zelle splitten":
            if st.session_state.split_stage == 0:
                selected = nearest_point_index(
                    st.session_state.roi_points,
                    x_orig, y_orig,
                    edit_radius
                )
                if selected is None:
                    st.warning(
                        "Bitte zuerst direkt auf den Mittelpunkt "
                        "der zusammengefassten Zelle klicken."
                    )
                else:
                    st.session_state.split_selected = selected
                    st.session_state.split_clicks = []
                    st.session_state.split_stage = 1
                    st.info(
                        "Jetzt zwei Zellzentren anklicken: "
                        "erstes Zentrum, dann zweites Zentrum."
                    )
                    st.experimental_rerun()
            elif st.session_state.split_stage == 1:
                selected_index = st.session_state.split_selected
                if selected_index is not None:
                    old = st.session_state.roi_points[selected_index]
                    if not is_near(old, (x_orig, y_orig), 5):
                        st.session_state.split_clicks.append((x_orig, y_orig))
                        if len(st.session_state.split_clicks) >= 2:
                            st.session_state.history.append(snapshot())
                            if 0 <= selected_index < len(st.session_state.roi_points):
                                st.session_state.roi_points.pop(selected_index)
                            st.session_state.roi_points.extend(
                                st.session_state.split_clicks[:2]
                            )
                            st.session_state.split_selected = None
                            st.session_state.split_clicks = []
                            st.session_state.split_stage = 0
                            st.success("✂️ Siamese Zelle in 2 Zellen geteilt.")
                            st.experimental_rerun()
                        else:
                            st.experimental_rerun()


# ============================================================
# CONTROL BUTTONS
# ============================================================

b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("↶ Undo"):
        undo()
        st.experimental_rerun()

with b2:
    if st.button("🧹 Alle Zellen entfernen"):
        st.session_state.history.append(snapshot())
        st.session_state.roi_points = []
        st.experimental_rerun()

with b3:
    if st.button("↩️ Auto-Erkennung wiederherstellen"):
        st.session_state.history.append(snapshot())
        st.session_state.roi_points = list(st.session_state.auto_points)
        st.experimental_rerun()

with b4:
    if st.button("✅ ROI fertig"):
        if len(st.session_state.roi_points) == 0:
            st.warning("Keine Zellen vorhanden.")
        else:
            points = list(st.session_state.roi_points)
            summary = calculate_roi_summary(
                st.session_state.slide_no,
                current_file.name,
                points,
                image_orig.shape,
                dbscan_eps_um,
                dbscan_min_samples
            )
            objects = make_objects_dataframe(
                st.session_state.slide_no,
                current_file.name,
                points,
                image_orig.shape
            )
            st.session_state.slide_summary = [
                x for x in st.session_state.slide_summary
                if x["ROI"] != current_file.name
            ]
            st.session_state.slide_objects = [
                x for x in st.session_state.slide_objects
                if x["ROI"] != current_file.name
            ]
            st.session_state.slide_summary.append(summary)
            st.session_state.slide_objects.extend(
                objects.to_dict("records")
            )
            if current_file.name not in st.session_state.finished_rois:
                st.session_state.finished_rois.append(current_file.name)
            st.session_state.roi_finished = True
            st.success(
                f"ROI abgeschlossen: {len(points)} finale AEC-Zellen."
            )


# ============================================================
# ROI NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns(3)

with nav1:
    if st.button("⬅️ Vorheriges ROI", disabled=(idx == 0)):
        st.session_state.current_roi_index -= 1
        reset_current_roi()
        st.experimental_rerun()

with nav2:
    st.write(
        f"**Fertig:** "
        f"{len(st.session_state.finished_rois)} / "
        f"{len(uploaded_files)}"
    )

with nav3:
    if st.button(
        "➡️ Nächstes ROI",
        disabled=(idx >= len(uploaded_files) - 1)
    ):
        st.session_state.current_roi_index += 1
        reset_current_roi()
        st.experimental_rerun()


# ============================================================
# CURRENT RESULT
# ============================================================

st.markdown("---")
st.subheader("📊 Aktuelles ROI")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Finale AEC-Zellen",
        len(st.session_state.roi_points)
    )

with col2:
    area_mm2 = (
        image_orig.shape[0] *
        image_orig.shape[1] *
        PIXEL_AREA_MM2
    )
    st.metric(
        "ROI-Fläche (mm²)",
        f"{area_mm2:.4f}"
    )

with col3:
    density = (
        len(st.session_state.roi_points) / area_mm2
        if area_mm2 > 0 else 0
    )
    st.metric(
        "AEC / mm²",
        f"{density:.3f}"
    )


# ============================================================
# SLIDE RESULTS
# ============================================================

st.markdown("---")
st.subheader(
    f"📋 Slide {st.session_state.slide_no} — bisher abgeschlossene ROIs"
)

if st.session_state.slide_summary:
    df_summary = pd.DataFrame(st.session_state.slide_summary)
    st.dataframe(df_summary, use_container_width=True)

    csv_summary = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Slide-Summary als CSV herunterladen",
        csv_summary,
        file_name=f"{st.session_state.slide_no}_summary.csv",
        mime="text/csv"
    )

if st.session_state.slide_objects:
    df_objects = pd.DataFrame(st.session_state.slide_objects)
    st.dataframe(df_objects, use_container_width=True)

    csv_objects = df_objects.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Objekt-Liste als CSV herunterladen",
        csv_objects,
        file_name=f"{st.session_state.slide_no}_objects.csv",
        mime="text/csv"
    )
