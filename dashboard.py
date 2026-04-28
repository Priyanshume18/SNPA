import streamlit as st
import pandas as pd
import time
from pathlib import Path

from anpr.config import Config
from anpr import plate as plate_mod

st.set_page_config(page_title="ANPR Admin Dashboard", page_icon="🗄️", layout="wide")

config = Config.from_file()
PLATES_FILE = Path(config.access.authorized_plates_file)
CSV_LOG_FILE = Path(config.logging.log_file)
TXT_LOG_FILE = Path(config.logging.app_log_file)


# ------------------------------------------------------------------ #
# File helpers — single source of truth
#
# Rule: the file ONLY ever contains normalized plate strings, one per
# line, no comments, no blanks, no spaces, no mixed case.
# Both the dashboard and the pipeline call plate_mod.normalize() before
# reading or writing, so they can never disagree on what's in the file.
# ------------------------------------------------------------------ #

def _read_plates() -> list[str]:
    """
    Read the file and return a sorted list of normalized plate strings.
    Any line that doesn't parse as a valid plate (comments, blanks, junk)
    is silently ignored — it will be dropped on the next write.
    """
    if not PLATES_FILE.exists():
        return []
    content = PLATES_FILE.read_text(encoding="utf-8")
    return sorted(plate_mod.load_plates_from_text(content))


def _write_plates(plates: list[str]) -> None:
    """
    Overwrite the file with *plates* — normalized, sorted, one per line.
    This is the ONLY function that writes the file, so the file is always
    clean after any operation.
    """
    PLATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write header comment so the file is still human-readable
    lines = ["# Authorized plates — managed by dashboard. One per line.\n"]
    lines += [f"{p}\n" for p in sorted(set(plates))]
    PLATES_FILE.write_text("".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ #
# Callbacks
# ------------------------------------------------------------------ #

def add_new_plate():
    raw = st.session_state.get("new_plate_input", "").strip()
    st.session_state.new_plate_input = ""
    if not raw:
        return

    normalised = plate_mod.normalize(raw)
    if not normalised:
        st.session_state.action_msg = ("error",
            f"❌ '{raw}' is not a valid Indian plate format. "
            f"Expected formats: MH12AB1234, KA05MJ7777, 22BH1234AA")
        return

    current = _read_plates()
    if normalised in current:
        st.session_state.action_msg = ("warning",
            f"⚠️ {normalised} is already in the authorized list.")
        return

    _write_plates(current + [normalised])
    st.session_state.action_msg = ("success",
        f"✅ {normalised} added. Pipeline picks it up within 30 s.")


def remove_plate(plate: str):
    current = _read_plates()
    updated = [p for p in current if p != plate]
    if len(updated) == len(current):
        # Plate wasn't in the list — nothing to do
        return
    _write_plates(updated)
    st.session_state.action_msg = ("success",
        f"🗑️ {plate} removed. Pipeline picks it up within 30 s.")


# ------------------------------------------------------------------ #
# Data loaders
# ------------------------------------------------------------------ #

def load_csv_logs() -> pd.DataFrame:
    if not CSV_LOG_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_LOG_FILE)
        return df.tail(15).iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_terminal_logs() -> str:
    if not TXT_LOG_FILE.exists():
        return "Waiting for main.py to start logging..."
    try:
        lines = TXT_LOG_FILE.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-15:])
    except Exception:
        return "Error reading log file."

# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #

st.title("SNPA Admin Dashboard")

col_caption, col_toggle = st.columns([4, 1])
with col_caption:
    st.caption(f"Plates file: `{PLATES_FILE.resolve()}`")
with col_toggle:
    auto_refresh = st.toggle("Auto Refresh", value=True)

st.divider()

col_left, col_right = st.columns([1, 1.5])

# ── LEFT: Authorized plate management ──────────────────────────────
with col_left:
    st.subheader("Authorized Plates")

    # Action feedback
    if "action_msg" in st.session_state:
        kind, msg = st.session_state.pop("action_msg")
        {"success": st.success, "warning": st.warning, "error": st.error}[kind](msg)

    # Add plate
    with st.container(border=True):
        st.text_input(
            "Add new plate",
            key="new_plate_input",
            placeholder="MH12AB1234",
        )
        st.button(
            "Add",
            on_click=add_new_plate,
            use_container_width=True,
        )

    plates = _read_plates()
    st.caption(f"{len(plates)} authorized")

    with st.expander("Manage list", expanded=True):
        if plates:
            for p in plates:
                col_plate, col_btn = st.columns([5, 1])
                col_plate.code(p, language=None)
                col_btn.button(
                    "Remove",
                    key=f"del_{p}",
                    on_click=remove_plate,
                    args=(p,),
                )
        else:
            st.caption("No entries")

    with st.expander("File contents", expanded=False):
        if PLATES_FILE.exists():
            st.code(PLATES_FILE.read_text(encoding="utf-8"), language="text")
        else:
            st.caption("File not found")

# ── RIGHT: Logs ────────────────────────────────────────────────────
with col_right:
    st.subheader("Access Logs")

    log_df = load_csv_logs()
    if not log_df.empty:
        status_col = next((c for c in log_df.columns if c.lower() == "status"), None)

        if status_col:
            def _color_status(val):
                v = str(val).upper()
                if v == "AUTHORIZED":
                    return "color: #2e7d32"
                if v == "UNAUTHORIZED":
                    return "color: #c62828"
                return ""

            st.dataframe(
                log_df.style.map(_color_status, subset=[status_col]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No logs available")

    st.divider()
    st.subheader("Application Logs")
    st.code(load_terminal_logs(), language="text")

# ── Auto-refresh ────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(2)
    st.rerun()