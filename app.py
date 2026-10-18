import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
import streamlit as st

st.set_page_config(page_title="WingLab | Module 1.1", page_icon="✈️", layout="wide")

st.title("WingLab — Multi-Station Parametric Flying-Wing Generator")
st.caption(
    "Module 1.1 | Editable spanwise stations, smooth geometry, multiple views, and CSV export. "
    "Preliminary geometry tool only; not flight-validated."
)

DEFAULT_STATIONS = pd.DataFrame({
    "eta": [0.00, 0.25, 0.50, 0.75, 1.00],
    "leading_edge_x_m": [0.00, 0.025, 0.070, 0.145, 0.240],
    "chord_m": [0.20, 0.18, 0.145, 0.090, 0.040],
    "z_m": [0.00, 0.004, 0.012, 0.030, 0.060],
    "twist_deg": [0.0, -0.5, -1.5, -2.0, -2.0],
})


def naca4(code: str = "0012", n: int = 81):
    """Return a common cosine-spaced x grid and normalized NACA 4-digit upper/lower z."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Airfoil must be a four-digit NACA code.")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    if not (0.0 < t < 0.40):
        raise ValueError("NACA thickness must be greater than 0% and less than 40%.")
    if m > 0 and p == 0:
        raise ValueError("Cambered NACA profiles require a nonzero camber-position digit.")

    beta = np.linspace(0.0, np.pi, n)
    x = (1.0 - np.cos(beta)) / 2.0
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m > 0:
        front = x < p
        rear = ~front
        yc[front] = m / p**2 * (2 * p * x[front] - x[front]**2)
        dyc_dx[front] = 2 * m / p**2 * (p - x[front])
        yc[rear] = m / (1 - p)**2 * (
            1 - 2 * p + 2 * p * x[rear] - x[rear]**2
        )
        dyc_dx[rear] = 2 * m / (1 - p)**2 * (p - x[rear])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    zu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    zl = yc - yt * np.cos(theta)
    return x, np.interp(x, xu, zu), np.interp(x, xl, zl)


def validate_stations(stations: pd.DataFrame, half_span: float):
    required = ["eta", "leading_edge_x_m", "chord_m", "z_m", "twist_deg"]
    missing = [col for col in required if col not in stations.columns]
    if missing:
        raise ValueError(f"Missing station columns: {', '.join(missing)}")

    df = stations[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isna().any().any():
        raise ValueError("All station cells must contain valid numbers.")
    if not np.all(np.isfinite(df.to_numpy(dtype=float))):
        raise ValueError("Station values must all be finite.")
    if len(df) < 3:
        raise ValueError("Define at least three stations: root, intermediate, and tip.")
    eta = df["eta"].to_numpy()
    if not np.isclose(eta[0], 0.0, atol=1e-8):
        raise ValueError("The first station must be at eta = 0.0 (centerline/root).")
    if not np.isclose(eta[-1], 1.0, atol=1e-8):
        raise ValueError("The last station must be at eta = 1.0 (wingtip).")
    if np.any(np.diff(eta) <= 0):
        raise ValueError("eta values must be strictly increasing with no duplicates.")
    if np.any(eta < 0) or np.any(eta > 1):
        raise ValueError("eta must be between 0 and 1.")
    if np.any(df["chord_m"].to_numpy() <= 0):
        raise ValueError("Every station chord must be positive.")
    if half_span <= 0:
        raise ValueError("Half-span must be positive.")
    return df


def generate_wing(stations: pd.DataFrame, half_span: float, airfoil: str,
                  n_span: int = 81, n_chord: int = 81):
    """Interpolate editable half-wing stations and mirror them across centerline."""
    df = validate_stations(stations, half_span)
    eta_station = df["eta"].to_numpy()
    station_values = {
        "leading_edge_x_m": df["leading_edge_x_m"].to_numpy(),
        "chord_m": df["chord_m"].to_numpy(),
        "z_m": df["z_m"].to_numpy(),
        "twist_deg": df["twist_deg"].to_numpy(),
    }

    eta_dense = np.linspace(0.0, 1.0, n_span)
    interp = {}
    for key, values in station_values.items():
        interp[key] = PchipInterpolator(eta_station, values)(eta_dense)

    chord_half = interp["chord_m"]
    if not np.all(np.isfinite(chord_half)) or np.any(chord_half <= 0):
        raise ValueError("Interpolated chord became nonpositive or non-finite. Adjust stations.")

    # Smoothly interpolated half-wing values, then mirrored for full-span geometry.
    eta_full = np.concatenate((-eta_dense[:0:-1], eta_dense))
    y_full = eta_full * half_span
    def mirror(values):
        return np.concatenate((values[:0:-1], values))

    le_x = mirror(interp["leading_edge_x_m"])
    chord = mirror(chord_half)
    z_base = mirror(interp["z_m"])
    twist = np.deg2rad(mirror(interp["twist_deg"]))

    x_norm, zu_norm, zl_norm = naca4(airfoil, n_chord)
    local_x = (x_norm[None, :] - 0.25) * chord[:, None]
    local_x = np.broadcast_to(local_x, (len(y_full), n_chord))

    def surface(z_norm):
        local_z = z_norm[None, :] * chord[:, None]
        angle = twist[:, None]
        x_rot = local_x * np.cos(angle) + local_z * np.sin(angle)
        z_rot = -local_x * np.sin(angle) + local_z * np.cos(angle)
        X = le_x[:, None] + 0.25 * chord[:, None] + x_rot
        Y = np.broadcast_to(y_full[:, None], X.shape).copy()
        Z = z_base[:, None] + z_rot
        return X, Y, Z

    Xu, Yu, Zu = surface(zu_norm)
    Xl, Yl, Zl = surface(zl_norm)

    all_arrays = [Xu, Yu, Zu, Xl, Yl, Zl]
    if not all(np.all(np.isfinite(a)) for a in all_arrays):
        raise ValueError("Generated coordinates contain NaN or infinite values.")
    if np.any(chord <= 0):
        raise ValueError("Generated chord must remain positive.")
    if not np.allclose(Yu, Yl):
        raise ValueError("Upper/lower surface span coordinates do not match.")
    if not np.allclose(Xu[:, 0], Xl[:, 0], atol=1e-6):
        raise ValueError("Leading-edge closure check failed.")

    # This is a geometric planform integral, not an aerodynamic calculation.
    area_half = half_span * np.trapz(chord_half, eta_dense)
    area = 2.0 * area_half
    aspect_ratio = (2.0 * half_span)**2 / area
    # MAC for arbitrary chord distribution: integral(c^2 dy) / integral(c dy).
    mac = np.trapz(chord_half**2, eta_dense) / np.trapz(chord_half, eta_dense)
    y_mac = (
        half_span
        * np.trapz(eta_dense * chord_half**2, eta_dense)
        / np.trapz(chord_half**2, eta_dense)
    )

    return {
        "Xu": Xu, "Yu": Yu, "Zu": Zu, "Xl": Xl, "Yl": Yl, "Zl": Zl,
        "y": y_full, "eta": eta_full, "chord": chord, "le_x": le_x,
        "z_base": z_base, "twist_rad": twist,
        "area": area, "aspect_ratio": aspect_ratio,
        "mac": mac, "y_mac_abs": y_mac,
        "station_df": df,
    }


def surface_coordinate_frame(wing):
    frames = []
    for side in ("upper", "lower"):
        X, Y, Z = wing["X" + side[0]], wing["Y" + side[0]], wing["Z" + side[0]]
        ns, nc = X.shape
        frames.append(pd.DataFrame({
            "surface": side,
            "span_station_index": np.repeat(np.arange(ns), nc),
            "chord_station_index": np.tile(np.arange(nc), ns),
            "x_m": X.ravel(), "y_m": Y.ravel(), "z_m": Z.ravel(),
        }))
    return pd.concat(frames, ignore_index=True)


with st.sidebar:
    st.header("Global parameters")
    full_span = st.slider("Full wingspan (m)", 0.4, 3.0, 1.2, 0.05)
    half_span = full_span / 2.0
    airfoil = st.selectbox("NACA four-digit airfoil", ["0012", "2412", "4412", "0015", "0018"])
    n_span = st.select_slider("Points per half-span", options=[21, 41, 61, 81, 121], value=81)
    n_chord = st.select_slider("Chordwise points", options=[21, 41, 61, 81, 101], value=81)
    st.caption("Edit station values in the table. eta=0 is the centerline; eta=1 is the wingtip.")
    st.caption("X, Z and chord are in metres; twist is in degrees.")

st.subheader("Editable spanwise control stations")
st.write(
    "Add or edit rows. Keep `eta` strictly increasing, starting at 0.0 and ending at 1.0. "
    "Intermediate stations control the local leading-edge position, chord, vertical offset, and twist."
)
edited_stations = st.data_editor(
    DEFAULT_STATIONS,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="station_editor",
    column_config={
        "eta": st.column_config.NumberColumn("eta (0=root, 1=tip)", min_value=0.0, max_value=1.0, step=0.05, format="%.3f"),
        "leading_edge_x_m": st.column_config.NumberColumn("Leading edge X (m)", format="%.4f"),
        "chord_m": st.column_config.NumberColumn("Chord C (m)", min_value=0.0001, format="%.4f"),
        "z_m": st.column_config.NumberColumn("Vertical offset Z (m)", format="%.4f"),
        "twist_deg": st.column_config.NumberColumn("Twist θ (deg)", format="%.2f"),
    },
)

try:
    wing = generate_wing(edited_stations, half_span, airfoil, n_span, n_chord)
except (ValueError, TypeError, KeyError) as exc:
    st.error(f"Geometry validation failed: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Full span", f"{full_span:.3f} m")
c2.metric("Planform area", f"{wing['area']:.4f} m²")
c3.metric("Aspect ratio", f"{wing['aspect_ratio']:.3f}")
c4.metric("Mean aerodynamic chord", f"{wing['mac']:.4f} m")

st.success("Implemented checks passed: finite coordinates, positive interpolated chord, station ordering, and leading-edge closure.")
st.warning(
    "These are not comprehensive CAD checks. Self-intersection, watertightness, structural integrity, "
    "aerodynamics, stability, manufacturability, and flight safety are not certified by this app."
)

with st.spinner("Preparing views..."):
    tab3d, tabtop, tabfront, tabside, tabdata = st.tabs(
        ["3D view", "Top view", "Front view", "Side view", "Coordinate data"]
    )

    with tab3d:
        fig3d = go.Figure()
        for X, Y, Z, name in (
            (wing["Xu"], wing["Yu"], wing["Zu"], "Upper surface"),
            (wing["Xl"], wing["Yl"], wing["Zl"], "Lower surface"),
        ):
            fig3d.add_trace(go.Surface(
                x=X, y=Y, z=Z, name=name, showscale=False, opacity=0.92,
                hovertemplate="X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<extra>" + name + "</extra>",
            ))
        # Spanwise station section curves.
        eta_sta = edited_stations["eta"].to_numpy(dtype=float)
        for eta_value in eta_sta:
            idx = int(np.argmin(np.abs(wing["eta"] - eta_value)))
            xs = np.concatenate([wing["Xu"][idx], wing["Xl"][idx, ::-1]])
            ys = np.concatenate([wing["Yu"][idx], wing["Yl"][idx, ::-1]])
            zs = np.concatenate([wing["Zu"][idx], wing["Zl"][idx, ::-1]])
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                name=f"Station eta={eta_value:.2f}", line={"width": 3},
            ))
        fig3d.update_layout(
            title="Full-span 3D geometry",
            scene={
                "xaxis_title": "X — chordwise (m)",
                "yaxis_title": "Y — spanwise (m)",
                "zaxis_title": "Z — vertical (m)",
                "aspectmode": "data",
                "camera": {"eye": {"x": 1.6, "y": 1.8, "z": 1.3}},
            },
            height=650, margin={"l": 0, "r": 0, "t": 45, "b": 0},
        )
        st.plotly_chart(fig3d, use_container_width=True)

    def add_planform_lines(fig, X, Y, label, color):
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", name=label, line={"color": color, "width": 2}))

    with tabtop:
        fig = go.Figure()
        add_planform_lines(fig, wing["Xu"][:, 0], wing["Yu"][:, 0], "Leading edge", "#20c997")
        add_planform_lines(fig, wing["Xu"][:, -1], wing["Yu"][:, -1], "Trailing edge", "#ffb020")
        fig.update_layout(title="Top view: X vs Y", xaxis_title="X (m)", yaxis_title="Y (m)", height=560)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with tabfront:
        fig = go.Figure()
        add_planform_lines(fig, wing["Yu"][:, 0], wing["Zu"][:, 0], "Leading edge", "#20c997")
        add_planform_lines(fig, wing["Yu"][:, -1], wing["Zu"][:, -1], "Trailing edge", "#ffb020")
        fig.update_layout(title="Front view: Y vs Z", xaxis_title="Y (m)", yaxis_title="Z (m)", height=560)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with tabside:
        center_idx = int(np.argmin(np.abs(wing["y"])))
        tip_idx = -1
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([wing["Xu"][tip_idx], wing["Xl"][tip_idx, ::-1]]),
            y=np.concatenate([wing["Zu"][tip_idx], wing["Zl"][tip_idx, ::-1]]),
            mode="lines", name="Tip section",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([wing["Xu"][center_idx], wing["Xl"][center_idx, ::-1]]),
            y=np.concatenate([wing["Zu"][center_idx], wing["Zl"][center_idx, ::-1]]),
            mode="lines", name="Root section",
        ))
        fig.update_layout(title="Side view: root and tip sections", xaxis_title="X (m)", yaxis_title="Z (m)", height=560)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with tabdata:
        coords = surface_coordinate_frame(wing)
        st.dataframe(coords.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(coords):,} surface points.")

st.subheader("Exports")
coords = surface_coordinate_frame(wing)
stations_export = edited_stations.copy()
stations_export["y_station_m_one_side"] = stations_export["eta"] * half_span

left, right = st.columns(2)
left.download_button(
    "Download surface coordinates (CSV)",
    data=coords.to_csv(index=False).encode("utf-8"),
    file_name="wing_surface_coordinates.csv",
    mime="text/csv",
)
right.download_button(
    "Download control stations (CSV)",
    data=stations_export.to_csv(index=False).encode("utf-8"),
    file_name="wing_control_stations.csv",
    mime="text/csv",
)

st.caption(
    "Coordinates are point grids in metres, not a STEP solid or validated watertight mesh. "
    "Area and aspect ratio use numerical integration of the interpolated chord distribution."
)
