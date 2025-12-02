import streamlit as st
import pickle, os, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="MedAlert — ICU Live", initial_sidebar_state="expanded")

# -----------------------
# CONFIG
# -----------------------
CACHE_DEFAULT = "patient_probs.pkl"   # expected cache file (list of dicts with keys: patient_file, hours, probs, n_hours, onset_hour)
CSV_ROOT = "sepsis_csv_dataset"      
MAX_SPARK = 20
DEFAULT_CARDS = 100
MIN_SPEED = 0.25
MAX_SPEED = 2.0

# -----------------------
# HELPERS: load cache + safe wrappers
# -----------------------
@st.cache_data
def load_patient_cache(path):
    """Load patient_probs.pkl (cached). Returns list-of-dicts or None, err."""
    try:
        if not os.path.exists(path):
            return None, f"Cache not found: {path}"
        with open(path, "rb") as f:
            data = pickle.load(f)
        # quick validation
        if not isinstance(data, list):
            return None, "Cache format invalid (expected list)."
        return data, None
    except Exception as e:
        return None, f"Failed to load cache: {e}"

def prob_at(entry, hour):
    # entry["hours"] and entry["probs"] expected aligned lists
    hours = entry.get("hours", [])
    probs = entry.get("probs", [])
    if not hours or not probs:
        return float("nan")
    # find last index where hours[idx] <= hour
    idx = None
    for i,h in enumerate(hours):
        if h <= hour:
            idx = i
        else:
            break
    return probs[idx] if idx is not None else float("nan")

# -----------------------
# SESSION INIT (safe)
# -----------------------
if "ui_ready" not in st.session_state:
    st.session_state.ui_ready = False      # will be set True after first render
    st.session_state.play_idx = 0          # simulated current hour index
    st.session_state.playing = False      # whether autoplay is running
    st.session_state.selected = None      # selected patient id
    st.session_state.patients = None      # cached patient objects
    st.session_state.cache_path = CACHE_DEFAULT
    st.session_state.cards = DEFAULT_CARDS
    st.session_state.threshold = 0.5
    st.session_state.speed = 1.0

# -----------------------
# SIDEBAR: controls + upload
# -----------------------
with st.sidebar:
    st.title("MedAlert Controls")
    st.markdown("**Data / Files**")
    uploaded_cache = st.file_uploader("Upload patient_probs.pkl (optional)", type=["pkl"])
    if uploaded_cache is not None:
        # write to local file and reload
        local_cache_path = Path(CACHE_DEFAULT)
        with open(local_cache_path, "wb") as f:
            f.write(uploaded_cache.getbuffer())
        st.success(f"Saved {local_cache_path}")
        # clear cached loader so it picks up new file
        load_patient_cache.clear()
        st.experimental_rerun()

    st.markdown("---")
    dataset_choice = st.selectbox("Dataset view:", ["training_setA", "training_setB", "combined", "random sample"])
    cards = st.select_slider("Cards to display", options=[25,50,75,100,150,200], value=st.session_state.cards)
    st.session_state.cards = int(cards)
    st.markdown("---")
    st.markdown("**Playback**")
    speed = st.slider("Playback speed (seconds per simulated hour)", MIN_SPEED, MAX_SPEED, float(st.session_state.speed), step=0.25)
    st.session_state.speed = float(speed)
    threshold = st.slider("Alert threshold", 0.01, 0.99, float(st.session_state.threshold), step=0.01)
    st.session_state.threshold = float(threshold)
    autoplay = st.checkbox("Autoplay after UI ready", value=True)
    st.markdown("---")
    st.markdown("Session")
    if st.button("Reset simulation"):
        st.session_state.play_idx = 0
        st.session_state.selected = None
        st.session_state.playing = False
    if st.button("Play / Pause"):
        st.session_state.playing = not st.session_state.playing

# -----------------------
# LOAD CACHE (main)
# -----------------------
patients, err = load_patient_cache(st.session_state.cache_path)
if err:
    st.error(err)
    st.stop()

# keep a dict index for fast lookup
patient_index = {p["patient_file"]: p for p in patients}
all_ids = list(patient_index.keys())

# subset pick function
def pick_subset(choice):
    n = len(all_ids)
    if choice == "training_setA":
        return all_ids[:min(20000, n)]
    if choice == "training_setB":
        return all_ids[20000:40000] if n>20000 else all_ids
    if choice == "combined":
        return all_ids
    # random sample
    rng = np.random.default_rng(42)
    take = min(2000, n)
    return list(rng.choice(all_ids, size=take, replace=False))

subset_ids = pick_subset(dataset_choice)

# -----------------------
# RANK + DISPLAY LIST
# -----------------------
cur_hour = st.session_state.play_idx

# compute ranking by current probability (fast)
ranked = []
for pid in subset_ids:
    p = prob_at(patient_index[pid], cur_hour)
    ranked.append((pid, -1.0 if math.isnan(p) else p))
ranked.sort(key=lambda x: x[1], reverse=True)

display_ids = [pid for pid,_ in ranked[:st.session_state.cards]]
top_spark = display_ids[:min(MAX_SPARK, len(display_ids))]
color_only = display_ids[len(top_spark):]

# -----------------------
# PAGE LAYOUT
# -----------------------
left_col, right_col = st.columns([3,1])
with left_col:
    st.header("MedAlert — ICU Live Replay")
    st.markdown(f"**Simulated hour:** {cur_hour}   •   **Cards:** {len(display_ids)}   •   **Dataset:** {dataset_choice}")
    st.markdown("Use **Inspect** buttons to pause playback and open the inspector.")

    # Top sparkline grid
    st.subheader("Priority patients (sparklines)")
    cols_per_row = 4
    for i in range(0, len(top_spark), cols_per_row):
        row = st.columns(cols_per_row)
        for j, pid in enumerate(top_spark[i:i+cols_per_row]):
            col = row[j]
            entry = patient_index[pid]
            pnow = prob_at(entry, cur_hour)
            valid_idxs = [k for k,h in enumerate(entry["hours"]) if h <= cur_hour]
            y = np.array(entry["probs"])[valid_idxs[-40:]] if valid_idxs else np.array([])
            # plotly sparkline
            fig = go.Figure()
            if y.size:
                fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(width=2)))
                fig.add_hline(y=st.session_state.threshold, line=dict(color="red", dash="dash"))
            fig.update_layout(height=90, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
            # status text
            if math.isnan(pnow):
                status = ("No data", "orange")
            elif pnow >= st.session_state.threshold:
                status = ("ALERT", "red")
            elif pnow >= 0.75 * st.session_state.threshold:
                status = ("WATCH", "goldenrod")
            else:
                status = ("OK", "green")
            with col:
                st.markdown(f"**{pid}**")
                st.markdown(f"<div style='font-weight:600;color:{status[1]}'>{status[0]} {'' if math.isnan(pnow) else f'• {pnow:.2f}'}</div>", unsafe_allow_html=True)
                # keep ploty small
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                if st.button("Inspect", key=f"inspect_{pid}"):
                    st.session_state.selected = pid
                    st.session_state.playing = False

    # color-only grid
    st.subheader("Other patients")
    cols_per_row = 6
    for i in range(0, len(color_only), cols_per_row):
        row = st.columns(cols_per_row)
        for j, pid in enumerate(color_only[i:i+cols_per_row]):
            col = row[j]
            entry = patient_index[pid]
            pnow = prob_at(entry, cur_hour)
            if math.isnan(pnow):
                label = "No data"
                bg = "#efefef"
            elif pnow >= st.session_state.threshold:
                label = f"ALERT {pnow:.2f}"
                bg = "#ffdce0"
            elif pnow >= 0.75*st.session_state.threshold:
                label = f"WATCH {pnow:.2f}"
                bg = "#fff4e0"
            else:
                label = f"OK {pnow:.2f}"
                bg = "#eafaf0"
            with col:
                st.markdown(f"<div style='background:{bg};padding:6px;border-radius:6px;text-align:center'><b>{pid}</b><div style='font-size:12px;margin-top:6px'>{label}</div></div>", unsafe_allow_html=True)
                if st.button("V", key=f"v_{pid}"):
                    st.session_state.selected = pid
                    st.session_state.playing = False

with right_col:
    st.header("Inspector")
    sel = st.session_state.selected
    if sel is None:
        st.info("Click 'Inspect' on any card to view details.")
    else:
        entry = patient_index[sel]
        st.markdown(f"### {sel}")
        st.write(f"Total hours: {entry.get('n_hours','?')}")
        if entry.get("onset_hour") is not None:
            st.write(f"Sepsis onset hour: {entry['onset_hour']}")
        # probability timeline
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=entry["hours"], y=entry["probs"], mode="lines", name="prob"))
        figp.add_hline(y=st.session_state.threshold, line=dict(color="red", dash="dash"))
        if entry.get("onset_hour") is not None:
            figp.add_vline(x=entry["onset_hour"], line=dict(color="green", dash="dash"))
        figp.update_layout(height=300)
        st.plotly_chart(figp, use_container_width=True, config={"displayModeBar": False})

        # optional: try load patient CSV from local CSV_ROOT (if exists)
        csv_a = Path(CSV_ROOT) / "training_setA" / "training_setA" / (sel.replace(".csv","") + ".csv")
        csv_b = Path(CSV_ROOT) / "training_setB" / "training_setB" / (sel.replace(".csv","") + ".csv")
        csv_path = csv_a if csv_a.exists() else (csv_b if csv_b.exists() else None)
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                st.write("Latest vitals (last row):")
                st.dataframe(df.tail(1))
            except Exception as e:
                st.write("Failed to load patient CSV:", e)

        if st.button("Close Inspector"):
            st.session_state.selected = None
            st.session_state.playing = True

# -----------------------
# SAFE AUTOPLAY INITIALIZATION
# -----------------------
if not st.session_state.ui_ready:
    st.session_state.ui_ready = True
    # start autoplay only after first render if user enabled
    if autoplay:
        # small delay to let DOM settle
        time.sleep(0.3)
        st.session_state.playing = True

# -----------------------
# PLAYBACK TICK (safe)
# -----------------------
if st.session_state.ui_ready and st.session_state.playing:
    # find a conservative max hours (from display_list)
    max_hours = 0
    for pid in display_ids:
        h = patient_index[pid].get("n_hours", 0)
        if isinstance(h, (int, float)) and h > max_hours:
            max_hours = int(h)
    # advance index safely
    if st.session_state.play_idx < max_hours - 1:
        st.session_state.play_idx += 1
    # throttle playback
    time.sleep(st.session_state.speed)
    st.experimental_rerun()

# -----------------------
# FOOTER: quick metrics
# -----------------------
st.markdown("---")
# quick aggregate metrics around current threshold
septic_total = sum(1 for p in patients if p.get("onset_hour") is not None)
septic_detected = sum(1 for p in patients if any([(pp >= st.session_state.threshold) for pp in p.get("probs",[])]))
st.write(f"Septic patients total: **{septic_total}** • Patients with any prob >= threshold: **{septic_detected}**")
st.caption("Tip: reduce number of cards if the UI becomes sluggish.")
