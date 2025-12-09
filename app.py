import streamlit as st
import pickle, os, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    layout="wide",
    page_title="MedAlert — ICU Live Dashboard",
    initial_sidebar_state="expanded"
)

# ===============================
# CONSTANTS
# ===============================
CACHE_DEFAULT = "patient_probs.pkl"  # expected ICU-wide prob cache file
CSV_ROOT = "sepsis_csv_dataset"      # unzipped patient CSV dataset
MAX_SPARK = 20                       # number of sparkline cards
DEFAULT_CARDS = 100                  # number of patient tiles initially visible
MIN_SPEED = 0.25                     # min playback delay (sec)
MAX_SPEED = 2.0                      # max playback delay (sec)

# ===============================
# LOAD PATIENT CACHE WITH STREAMLIT CACHING
# ===============================
@st.cache_data
def load_patient_cache(path):
    """Load precomputed patient probability timelines."""
    try:
        if not os.path.exists(path):
            return None, f"Cache not found at: {path}"
        with open(path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            return None, "Invalid cache format — expected list of dicts."

        return data, None
    except Exception as e:
        return None, f"Failed to load cache: {e}"

# ===============================
# UTILITY — probability lookup at specific hour
# ===============================
def prob_at(entry, hour):
    hours = entry.get("hours", [])
    probs = entry.get("probs", [])
    if not hours or not probs:
        return float("nan")

    idx = None
    for i, h in enumerate(hours):
        if h <= hour:
            idx = i
        else:
            break

    return probs[idx] if idx is not None else float("nan")

# ===============================
# SESSION INIT
# ===============================
if "ui_ready" not in st.session_state:
    st.session_state.ui_ready = False
    st.session_state.play_idx = 0
    st.session_state.playing = False
    st.session_state.selected = None
    st.session_state.cache_path = CACHE_DEFAULT
    st.session_state.cards = DEFAULT_CARDS
    st.session_state.threshold = 0.50
    st.session_state.speed = 1.0

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("⚙️ MedAlert Controls")
    st.markdown("---")

    st.subheader("🔌 Load ICU Data Cache")
    uploaded_cache = st.file_uploader("Upload patient_probs.pkl", type=["pkl"])
    if uploaded_cache:
        with open(CACHE_DEFAULT, "wb") as f:
            f.write(uploaded_cache.getbuffer())
        load_patient_cache.clear()
        st.success("Cache updated. Reloading…")
        st.experimental_rerun()

    st.markdown("---")

    dataset_choice = st.selectbox(
        "Dataset Selection:",
        ["training_setA", "training_setB", "combined", "random sample"]
    )

    st.session_state.cards = int(
        st.select_slider(
            "Cards to display",
            options=[25, 50, 75, 100, 150, 200],
            value=st.session_state.cards
        )
    )

    st.markdown("---")
    st.subheader("🎬 Playback")

    st.session_state.speed = float(
        st.slider("Playback speed (sec per hour)", MIN_SPEED, MAX_SPEED, st.session_state.speed, step=0.25)
    )

    st.session_state.threshold = float(
        st.slider("Alert threshold", 0.01, 0.99, st.session_state.threshold, step=0.01)
    )

    autoplay = st.checkbox("Autoplay when UI ready", value=True)

    st.markdown("---")

    if st.button("Reset Simulation"):
        st.session_state.play_idx = 0
        st.session_state.selected = None
        st.session_state.playing = False

    if st.button("Play / Pause"):
        st.session_state.playing = not st.session_state.playing


# ===============================
# LOAD ICU PATIENT PROBABILITIES
# ===============================
patients, err = load_patient_cache(st.session_state.cache_path)
if err:
    st.error(err)
    st.stop()

patient_index = {p["patient_file"]: p for p in patients}
all_ids = list(patient_index.keys())

def pick_subset(choice):
    n = len(all_ids)
    if choice == "training_setA":
        return all_ids[:min(20000, n)]
    if choice == "training_setB":
        return all_ids[20000:40000] if n > 20000 else all_ids
    if choice == "combined":
        return all_ids
    # random sample
    rng = np.random.default_rng(42)
    take = min(2000, n)
    return list(rng.choice(all_ids, size=take, replace=False))

subset_ids = pick_subset(dataset_choice)

# =================================
# ICU TICK / CURRENT HOUR
# =================================
cur_hour = st.session_state.play_idx

ranked = []
for pid in subset_ids:
    p = prob_at(patient_index[pid], cur_hour)
    ranked.append((pid, -1 if math.isnan(p) else p))

ranked.sort(key=lambda x: x[1], reverse=True)

display_ids = [pid for pid, _ in ranked[:st.session_state.cards]]
top_spark = display_ids[:min(MAX_SPARK, len(display_ids))]
others = display_ids[len(top_spark):]

# =================================
# MAIN LAYOUT
# =================================
left, right = st.columns([3, 1])

# LEFT SIDE — ICU GRID
with left:
    st.header("🩺 MedAlert — ICU Live Replay")
    st.markdown(f"**Hour:** {cur_hour}  •  **Cards:** {len(display_ids)}  •  **Dataset:** {dataset_choice}")

    st.subheader("High Priority Patients")
    cols_per_row = 4

    for i in range(0, len(top_spark), cols_per_row):
        row = st.columns(cols_per_row)
        for j, pid in enumerate(top_spark[i: i+cols_per_row]):
            col = row[j]
            entry = patient_index[pid]
            p_now = prob_at(entry, cur_hour)

            # sparkline
            valid = [k for k, h in enumerate(entry["hours"]) if h <= cur_hour]
            y = np.array(entry["probs"])[valid[-40:]] if valid else np.array([])

            fig = go.Figure()
            if y.size:
                fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(width=2)))
                fig.add_hline(y=st.session_state.threshold, line=dict(color="red", dash="dash"))

            fig.update_layout(
                height=90,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )

            if math.isnan(p_now):
                status = ("No Data", "gray")
            elif p_now >= st.session_state.threshold:
                status = ("ALERT", "red")
            elif p_now >= 0.75 * st.session_state.threshold:
                status = ("WATCH", "orange")
            else:
                status = ("OK", "green")

            with col:
                st.markdown(f"**{pid}**")
                st.markdown(
                    f"<div style='color:{status[1]};font-weight:600'>{status[0]} "
                    f"{'' if math.isnan(p_now) else f'• {p_now:.2f}'}</div>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                if st.button("Inspect", key=f"inspect_{pid}"):
                    st.session_state.selected = pid
                    st.session_state.playing = False

    st.subheader("Remaining Patients")

    cols_per_row = 6
    for i in range(0, len(others), cols_per_row):
        row = st.columns(cols_per_row)
        for j, pid in enumerate(others[i:i+cols_per_row]):
            entry = patient_index[pid]
            p_now = prob_at(entry, cur_hour)

            if math.isnan(p_now):
                label = "No Data"
                bg = "#eee"
            elif p_now >= st.session_state.threshold:
                label = f"ALERT {p_now:.2f}"
                bg = "#ffd4d4"
            elif p_now >= 0.75 * st.session_state.threshold:
                label = f"WATCH {p_now:.2f}"
                bg = "#fff0c2"
            else:
                label = f"OK {p_now:.2f}"
                bg = "#e5f9e0"

            with row[j]:
                st.markdown(
                    f"<div style='background:{bg};padding:6px;border-radius:6px;text-align:center'>"
                    f"<b>{pid}</b><div style='font-size:12px'>{label}</div></div>",
                    unsafe_allow_html=True
                )

                if st.button("V", key=f"view_{pid}"):
                    st.session_state.selected = pid
                    st.session_state.playing = False

# =================================
# RIGHT PANEL — INSPECTOR
# =================================
with right:
    st.header("Inspector")

    if st.session_state.selected is None:
        st.info("Click any 'Inspect' button to view details.")
    else:
        pid = st.session_state.selected
        entry = patient_index[pid]

        st.subheader(pid)
        st.write(f"Total Hours: {entry.get('n_hours')}")
        if entry.get("onset_hour") is not None:
            st.write(f"Sepsis onset: {entry['onset_hour']}")

        # probability timeline
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=entry["hours"], y=entry["probs"], mode="lines"))
        figp.add_hline(y=st.session_state.threshold, line=dict(color="red", dash="dash"))

        if entry.get("onset_hour") is not None:
            figp.add_vline(x=entry["onset_hour"], line=dict(color="green", dash="dash"))

        figp.update_layout(height=280)
        st.plotly_chart(figp, use_container_width=True, config={"displayModeBar": False})

        # try loading raw patient CSV
        csv_a = Path(CSV_ROOT) / "training_setA" / "training_setA" / pid
        csv_b = Path(CSV_ROOT) / "training_setB" / "training_setB" / pid
        csv_path = csv_a if csv_a.exists() else (csv_b if csv_b.exists() else None)

        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                st.write("Latest Vitals:")
                st.dataframe(df.tail(1))
            except Exception as e:
                st.write("Error loading vitals:", e)

        if st.button("Close Inspector"):
            st.session_state.selected = None
            st.session_state.playing = True

# =================================
# AUTOPLAY TICK
# =================================
if not st.session_state.ui_ready:
    st.session_state.ui_ready = True
    if autoplay:
        time.sleep(0.3)
        st.session_state.playing = True

if st.session_state.ui_ready and st.session_state.playing:
    max_hours = max([patient_index[pid].get("n_hours", 0) for pid in display_ids])

    if st.session_state.play_idx < max_hours - 1:
        st.session_state.play_idx += 1

    time.sleep(st.session_state.speed)
    st.experimental_rerun()

# =================================
# FOOTER
# =================================
st.markdown("---")
septic_total = sum(1 for p in patients if p.get("onset_hour") is not None)
septic_detected = sum(1 for p in patients if any(pp >= st.session_state.threshold for pp in p["probs"]))
st.write(f"Total septic patients: **{septic_total}** • Patients above threshold: **{septic_detected}**")
