WingLab — Module 1.1
A Streamlit application for preliminary flying-wing geometry using editable spanwise control stations.
What's new in Module 1.1
- Editable station table with eta, leading-edge X, chord C, vertical offset Z, and twist θ.
- Shape-preserving PCHIP interpolation between stations.
- Mirrored left/right wing geometry.
- NACA four-digit section generation.
- Interactive 3D, top, front, and root/tip side views.
- Basic station, coordinate, chord, and leading-edge closure checks.
- Numerical planform area, aspect ratio, and mean aerodynamic chord estimates.
- CSV export of surface point grids and control stations.
Requirements
- Python 3.10+ recommended
- See requirements.txt
Run locally
Windows PowerShell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
Deploy to Streamlit Community Cloud
Commit app.py and requirements.txt in the app's configured repository root. In Streamlit Cloud, set the main file path to app.py, then reboot/redeploy so dependencies install.
Control station definitions
Each table row defines a station on one half-wing:
- eta: normalized spanwise location from 0 (centerline/root) to 1 (tip).
- leading_edge_x_m: leading-edge X offset in metres.
- chord_m: local chord in metres.
- z_m: vertical offset in metres.
- twist_deg: local twist in degrees.
Station eta values must be strictly increasing, with first row eta=0 and last row eta=1. Add intermediate rows to control the shape. Chord must remain positive. The geometry is mirrored about the centerline, using the same X, Z, chord, and twist on both sides.
Coordinate conventions
- X: chordwise, positive aft in the generated coordinate convention.
- Y: spanwise, negative on the left wing and positive on the right wing.
- Z: vertical.
- Length units: metres; twist units: degrees.
- Sweep/curvature is controlled by the station leading-edge X values.
- Spanwise interpolation uses SciPy PCHIP, a shape-preserving piecewise cubic interpolator.
Limitations
This is a preliminary geometry prototype. It does not calculate aerodynamic forces or predict lift-to-drag ratio. It does not provide structural analysis, static stability, trim, control effectiveness, material allowables, fatigue, buckling, CAD-kernel solid construction, or flight certification. Current checks are not a proof of self-intersection-free or watertight geometry. Do not use it alone to manufacture or fly an aircraft.
Troubleshooting
- ModuleNotFoundError: ensure requirements.txt is in the configured app root and contains all dependencies; reboot after commit.
- App entry point: Streamlit Cloud should run app.py.
- If the station editor reports invalid input, ensure eta begins at 0, ends at 1, is strictly increasing, and chord values are positive.
- Send the full Streamlit logs and screenshot for runtime errors.
