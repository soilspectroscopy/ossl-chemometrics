# pyrefly: ignore [missing-import]
from shiny import App, render, ui, reactive
import pandas as pd
import io
import re
import sys
import json
import ssl
from pathlib import Path

# --------------------------------------------------------
# Environment setup
# --------------------------------------------------------
IS_WASM = sys.platform == "emscripten"

if not IS_WASM:
    ssl._create_default_https_context = ssl._create_unverified_context

# --------------------------------------------------------
# NOTE: duckdb and pyarrow are intentionally NOT imported
# here at module level.  They are imported inside the
# reactive effect that first needs them (_load_metadata).
# This means Pyodide will not fetch those heavy wheels
# until the user clicks "Load data", so the UI becomes
# interactive much sooner on first load.
# --------------------------------------------------------

# --------------------------------------------------------
# HTTP fetch helper — works in both WASM and native
# --------------------------------------------------------
def _http_get(url: str, headers: dict | None = None) -> bytes:
    """Fetch a URL and return raw bytes. Works in WASM and native."""
    if IS_WASM:
        # pyrefly: ignore [missing-import]
        import js
        xhr = js.XMLHttpRequest.new()
        xhr.open("GET", url, False)   # synchronous
        if headers:
            for k, v in headers.items():
                xhr.setRequestHeader(k, v)
        xhr.responseType = "arraybuffer"
        xhr.send(None)
        if xhr.status not in (200, 206):
            raise RuntimeError(f"HTTP {xhr.status} fetching {url}")
        return bytes(xhr.response.to_py())
    else:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req) as resp:
            return resp.read()


def fetch_parquet_as_arrow(url: str):
    """Download a full Parquet file and return a PyArrow Table."""
    # pyrefly: ignore [missing-import]
    import pyarrow.parquet as pq
    buf = io.BytesIO(_http_get(url))
    return pq.read_table(buf)


def fetch_json(url: str) -> dict:
    """Download and parse a JSON file."""
    raw = _http_get(url)
    return json.loads(raw.decode("utf-8"))


# --------------------------------------------------------
# Load local CSVs once at startup (cheap — no parquet)
# --------------------------------------------------------
try:
    csv_path = Path(__file__).parent / "input" / "ossl_individual_datasets_urls_v1.3.csv"
    urls_df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading URL list: {e}")
    urls_df = pd.DataFrame()

try:
    overview_path = Path(__file__).parent / "input" / "ossl_listed_libraries_overview.csv"
    overview_df = pd.read_csv(overview_path)
except Exception as e:
    print(f"Error reading overview CSV: {e}")
    overview_df = pd.DataFrame()

# --------------------------------------------------------
# Metadata JSON — loaded once at startup.
# The JSON lives next to app.py (or on your bucket).
# If it cannot be found/fetched the app falls back to
# the old behaviour of fetching column names from parquet.
# --------------------------------------------------------
_METADATA_JSON_PATH = Path(__file__).parent / "ossl_metadata.json"
_METADATA_JSON_URL  = None   # set this if hosting on a bucket, e.g.:
# _METADATA_JSON_URL = "https://storage.googleapis.com/your-bucket/ossl_metadata.json"

def _load_metadata_json() -> dict | None:
    """Try to load ossl_metadata.json from disk or remote URL."""
    # 1. Local file (fastest, works in native and WASM if bundled)
    if _METADATA_JSON_PATH.exists():
        try:
            with open(_METADATA_JSON_PATH) as f:
                data = json.load(f)
            print(f"Metadata JSON loaded from disk ({len(data.get('datasets', {}))} datasets)")
            return data
        except Exception as e:
            print(f"Could not parse local metadata JSON: {e}")

    # 2. Remote URL
    if _METADATA_JSON_URL:
        try:
            data = fetch_json(_METADATA_JSON_URL)
            print(f"Metadata JSON loaded from URL ({len(data.get('datasets', {}))} datasets)")
            return data
        except Exception as e:
            print(f"Could not fetch remote metadata JSON: {e}")

    print("No metadata JSON found — will fall back to parquet schema fetching")
    return None


METADATA = _load_metadata_json()   # None if not available

# Populate dataset_codes from metadata if available, otherwise from CSV
if METADATA and "datasets" in METADATA:
    dataset_codes = sorted(METADATA["datasets"].keys())
else:
    if not urls_df.empty and "dataset_code" in urls_df.columns:
        dataset_codes = sorted(urls_df["dataset_code"].unique().tolist())
    else:
        dataset_codes = []


# --------------------------------------------------------
# Small pure-Python helpers
# --------------------------------------------------------
def _is_spectral(col_name: str) -> bool:
    """Return True if col_name is a numeric string (a spectral band)."""
    try:
        float(col_name)
        return True
    except (ValueError, TypeError):
        return False


def _make_spectrum_svg(df: pd.DataFrame) -> str | None:
    """
    Build a lightweight inline SVG showing the mean spectrum ± 1 std.
    Uses only pandas/numpy — no matplotlib, no extra imports.
    Returns None if there are no spectral columns.
    """
    import numpy as np

    spec_cols = sorted(
        [c for c in df.columns if _is_spectral(c)],
        key=float,
    )
    if not spec_cols:
        return None

    xs = [float(c) for c in spec_cols]
    arr = df[spec_cols].to_numpy(dtype=float)
    ys_mean = np.nanmean(arr, axis=0)
    ys_std  = np.nanstd(arr, axis=0)
    n_rows  = arr.shape[0]

    W, H, PX, PY = 600, 160, 36, 16   # viewBox dims and padding

    x_min, x_max = min(xs), max(xs)
    y_vals_all = list(ys_mean - ys_std) + list(ys_mean + ys_std)
    y_min = min(v for v in y_vals_all if not (v != v))  # skip NaN
    y_max = max(v for v in y_vals_all if not (v != v))
    y_rng = y_max - y_min or 1.0
    x_rng = x_max - x_min or 1.0

    def sx(v): return PX + (v - x_min) / x_rng * (W - 2 * PX)
    def sy(v): return H - PY - (v - y_min) / y_rng * (H - 2 * PY)

    # Std envelope (upper then lower reversed = closed polygon)
    upper = [(sx(x), sy(m + s)) for x, m, s in zip(xs, ys_mean, ys_std)]
    lower = [(sx(x), sy(m - s)) for x, m, s in zip(xs, ys_mean, ys_std)]
    envelope_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower[::-1])

    # Mean line
    mean_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys_mean))

    # Axis tick labels (x-axis: 5 ticks, y-axis: 3 ticks)
    x_ticks = [x_min + i * x_rng / 4 for i in range(5)]
    y_ticks = [y_min, (y_min + y_max) / 2, y_max]
    x_tick_svg = "".join(
        f'<text x="{sx(v):.1f}" y="{H - 2}" text-anchor="middle" '
        f'font-size="10" fill="#888">{v:.0f}</text>'
        for v in x_ticks
    )
    y_tick_svg = "".join(
        f'<text x="{PX - 4}" y="{sy(v):.1f}" text-anchor="end" '
        f'dominant-baseline="central" font-size="10" fill="#888">{v:.3f}</text>'
        for v in y_ticks
    )

    return f"""
<div style="margin-top:8px">
  <div style="font-size:11px;color:#888;margin-bottom:2px">
    Mean spectrum (n={n_rows:,}). Shaded area = ±1 SD.
  </div>
  <svg viewBox="0 0 {W} {H}" width="100%"
       style="display:block;border-radius:6px;background:#fafafa">
    <!-- std envelope -->
    <polygon points="{envelope_pts}"
             fill="#667eea" fill-opacity="0.15" stroke="none"/>
    <!-- mean line -->
    <polyline points="{mean_pts}"
              fill="none" stroke="#667eea" stroke-width="1.5"
              stroke-linejoin="round" stroke-linecap="round"/>
    <!-- axis lines -->
    <line x1="{PX}" y1="{PY}" x2="{PX}" y2="{H-PY}"
          stroke="#ddd" stroke-width="0.8"/>
    <line x1="{PX}" y1="{H-PY}" x2="{W-PX}" y2="{H-PY}"
          stroke="#ddd" stroke-width="0.8"/>
    {x_tick_svg}
    {y_tick_svg}
  </svg>
</div>
"""


# --------------------------------------------------------
# UI helpers shared across tabs
# --------------------------------------------------------

def _premium_card(*children, title: str | None = None):
    inner = []
    if title:
        inner.append(ui.div(title, class_="section-title"))
    inner.extend(children)
    return ui.div(*inner, class_="premium-card")


# --------------------------------------------------------
# Tab: Explore
# --------------------------------------------------------
data_overview_content = ui.div(
    _premium_card(
        ui.p(
            "The ",
            ui.tags.a("Open Soil Spectral Library (OSSL)",
                      href="https://docs.soilspectroscopy.org/", target="_blank"),
            " is a global compilation of soil spectral datasets paired with "
            "reference laboratory measurements, assembled from contributions "
            "worldwide and made freely available for research and modelling.",
        ),
        ui.p(
            "All datasets share a common structure with three linked tables: ",
            ui.tags.b("Soilsite"), " (sampling location and date), ",
            ui.tags.b("Soillab"), " (measured soil properties), and ",
            ui.tags.b("Spectra"), " (raw absorbance or reflectance values across "
            "NIR, VisNIR, or MIR ranges at evenly spaced intervals).",
        ),
        ui.p(
            "Use the tabs above to work through the workflow: browse the available "
            "datasets here, then go to ",
            ui.tags.b("Prepare"), " to filter, join, and export a dataset tailored "
            "to your needs. If you have your own spectra, use ",
            ui.tags.b("My Data"), " to resample them to a standard interval. "
            "Finally, open ",
            ui.tags.b("Analyse"), " to launch one of the mdatools chemometric apps "
            "directly in your browser.",
        ),
        ui.p(
            "For full details on each dataset, including collection protocols and "
            "license information, see the ",
            ui.tags.a("'Soil spectral libraries' section",
                      href="https://docs.soilspectroscopy.org/libraries.html",
                      target="_blank"),
            " of the OSSL Manual.",
        ),
    ),
    _premium_card(
        ui.output_data_frame("datasets_overview_table"),
        title="Available datasets",
    ),
)


# --------------------------------------------------------
# Tab: Prepare
# --------------------------------------------------------
data_selection_content = ui.div(

    # 1. Data Selection
    _premium_card(
        ui.layout_columns(
            ui.input_select("dataset_code", "Dataset code",
                            choices=dataset_codes, width="100%"),
            ui.output_ui("spectra_selector"),
            ui.input_action_button("load_metadata", "Load data",
                                   class_="btn-premium w-100 mt-4"),
            col_widths=(4, 4, 4),
        ),
        title="1. Select dataset and spectral region",
    ),

    # 2. Soil properties
    _premium_card(
        ui.p("Select a soil property (or many). Rows with no valid values for the "
             "selected property will be automatically removed.",
             class_="text-muted small"),
        ui.output_ui("lab_column_selector"),
        ui.layout_columns(
            ui.div(
                ui.input_radio_buttons(
                    "transform_type", "Apply transformation (optional)",
                    choices={"none": "No transformation",
                             "sqrt": "Square root (sqrt)",
                             "log1p": "Log(1+x)"},
                    selected="none", inline=True,
                ),
                ui.input_action_button("run_soil_only", "Preview",
                                       class_="btn-premium w-100 mb-2"),
            ),
            ui.div(
                ui.p("Summary statistics of selected properties"),
                ui.output_table("summary_stats_table"),
            ),
            col_widths=(3, 6),
        ),
        title="2. Soil properties",
    ),

    # 3 & 4. Advanced filtering — collapsible
    ui.div(
        ui.tags.details(
            ui.tags.summary(
                ui.span("3 & 4. Advanced filtering (optional) — site & spectral metadata",
                        style="font-size:1rem;font-weight:600;color:#4a5568;cursor:pointer;"),
                style="list-style:none;display:flex;align-items:center;gap:8px;"
                      "padding:14px 20px;border-radius:12px;"
                      "background:white;border:1px solid #e9ecef;"
                      "box-shadow:0 4px 6px rgba(0,0,0,0.05);",
            ),
            # 3. Site filtering
            ui.div(
                _premium_card(
                    ui.layout_columns(
                        ui.div(
                            ui.p("Select site columns to filter. Leave empty to keep all rows.",
                                 class_="text-muted small"),
                            ui.output_ui("site_column_selector"),
                        ),
                        ui.div(
                            ui.p("Highlight the values to keep", class_="text-muted small"),
                            ui.output_ui("site_level_filters"),
                        ),
                        col_widths=(6, 6),
                    ),
                    ui.input_action_button("run_soil_site", "Recalculate statistics",
                                           class_="btn-premium w-25 mt-2"),
                    title="3. Site filtering",
                ),
                # 4. Spectral metadata filtering
                _premium_card(
                    ui.layout_columns(
                        ui.div(
                            ui.p("Select columns to filter. Leave empty to keep all rows.",
                                 class_="text-muted small"),
                            ui.output_ui("spec_column_selector"),
                        ),
                        ui.div(
                            ui.p("Highlight the unique values to keep.",
                                 class_="text-muted small"),
                            ui.output_ui("spec_level_filters"),
                        ),
                        col_widths=(6, 6),
                    ),
                    title="4. Spectral metadata filtering (when available)",
                ),
                style="margin-top:8px;",
            ),
        ),
        style="margin-bottom:20px;",
    ),

    # 5. Join & Export
    _premium_card(
        ui.layout_columns(
            ui.div(
                ui.input_text("id_col", "Common ID column",
                              value="id.layer_local_c", width="100%"),
                ui.input_select(
                    "spec_interval", "Spectral resolution",
                    choices={"2": "Every 2 units (nm or cm⁻¹)", "10": "Every 10 units (nm or cm⁻¹)"},
                    selected="2", width="100%",
                ),
                ui.layout_columns(
                    ui.input_numeric("spec_min", "Min (nm or cm⁻¹)", value=0),
                    ui.input_numeric("spec_max", "Max (nm or cm⁻¹)", value=10000),
                    col_widths=(6, 6),
                ),
                ui.input_select(
                    "join_type", "Join type",
                    choices={"inner": "Only data with spectra",
                             "full":  "Keep all data, add spectra where available"},
                    selected="inner", width="100%",
                ),
                ui.input_action_button("run_join", "Join with available spectra",
                                       class_="btn-premium w-100 mb-3"),
                ui.output_ui("export_column_selector"),
                ui.input_text("dl_filename", "Filename to save",
                              value="ossl_filtered_data.csv", width="100%"),
                ui.download_button("download_csv", "Download processed dataset",
                                   class_="btn-success-premium w-100"),
            ),
            ui.div(
                ui.output_ui("spectrum_preview"),   # inline SVG — above the table
                ui.div(style="height:14px;"),        # breathing room
                ui.output_data_frame("preview_table"),
            ),
            col_widths=(3, 9),
        ),
        title="5. Join spectra & export",
    ),

    # 6. Subsetting (shown only after a join)
    ui.output_ui("subsetting_panel"),
)


# --------------------------------------------------------
# Tab: Formatting
# --------------------------------------------------------
_SAMPLE_FILES = {
    "sample_visnir_data.csv": (
        "https://raw.githubusercontent.com/soilspectroscopy/ossl-models"
        "/main/sample-data/sample_visnir_data.csv"
    ),
    "sample_mir_data.csv": (
        "https://raw.githubusercontent.com/soilspectroscopy/ossl-models"
        "/main/sample-data/sample_mir_data.csv"
    ),
    "sample_neospectra_data.csv": (
        "https://raw.githubusercontent.com/soilspectroscopy/ossl-models"
        "/main/sample-data/sample_neospectra_data.csv"
    ),
}

formatting_content = ui.div(
    _premium_card(
        ui.p(
            "Use this tab to resample your own spectral CSV to a standard "
            "OSSL-compatible interval (2 or 10 nm / cm⁻¹) before uploading it "
            "to one of the ", ui.tags.b("Chemometrics"), " tools. "
            "No preprocessing is applied — only the spectral axis is resampled "
            "via linear interpolation.",
        ),
        ui.tags.ul(
            ui.tags.li(
                "The ", ui.tags.b("first column"), " is always treated as the "
                "sample ID and kept as-is."
            ),
            ui.tags.li(
                "Spectral columns are detected automatically as any column "
                "whose name is a plain number (e.g. 400, 402.5, 4000)."
            ),
            ui.tags.li(
                "Columns may be in any order (increasing or decreasing). "
                "The exported file always uses ", ui.tags.b("increasing order"),
                " regardless of the input."
            ),
        ),
        ui.p("Download example files to test the workflow:",
             class_="mb-1 mt-3 fw-semibold"),
        ui.div(
            ui.download_button("dl_sample_visnir",    "sample_visnir_data.csv",
                               class_="btn btn-sample me-2 mb-1"),
            ui.download_button("dl_sample_mir",       "sample_mir_data.csv",
                               class_="btn btn-sample me-2 mb-1"),
            ui.download_button("dl_sample_neospectra","sample_neospectra_data.csv",
                               class_="btn btn-sample me-2 mb-1"),
        ),
        title="Resample your own spectral data",
    ),

    _premium_card(
        ui.layout_columns(
            ui.div(
                ui.input_file("fmt_file", "Upload CSV file",
                              accept=[".csv"], width="100%"),
                ui.input_select(
                    "fmt_interval", "Target resolution",
                    choices={"2": "Every 2 units (nm or cm⁻¹)",
                             "10": "Every 10 units (nm or cm⁻¹)"},
                    selected="2", width="100%",
                ),
                ui.layout_columns(
                    ui.input_numeric("fmt_min", "Min (nm or cm⁻¹)",
                                     value=None),
                    ui.input_numeric("fmt_max", "Max (nm or cm⁻¹)",
                                     value=None),
                    col_widths=(6, 6),
                ),
                ui.p("Leave min/max blank to use the full range of your file.",
                     class_="text-muted small"),
                ui.input_action_button("fmt_run", "Resample",
                                       class_="btn-premium w-100 mt-2"),
            ),
            ui.div(
                ui.output_ui("fmt_preview"),
            ),
            col_widths=(4, 8),
        ),
        ui.div(
            ui.output_ui("fmt_download_ui"),
            class_="mt-3",
        ),
        title="Settings",
    ),
)


# --------------------------------------------------------
# Tab: Chemometrics
# --------------------------------------------------------
chemometrics_content = ui.div(
    _premium_card(
        ui.h3("mdatools: make chemometrics easy"),
        ui.p("Development and credits: ",
             ui.tags.a("Sergey Kucheryavskiy",
                        href="https://github.com/svkucheryavski", target="_blank"), "."),
        ui.p("Try most common chemometric methods directly in your browser. "
             "All calculations run on your local computer."),
        ui.p("Check video tutorials at ",
             ui.tags.a("youtube.com/@mdatools",
                        href="https://www.youtube.com/@mdatools", target="_blank"),
             ". For more information visit the ",
             ui.tags.a("mdatools website", href="https://mdatools.com", target="_blank"), "."),
        ui.hr(),
        ui.p("Download your processed dataset from the 'Prepare' tab, then open "
             "one of the tools below in a new tab to begin your analysis."),
        ui.layout_columns(
            ui.a("Spectral visualization and preprocessing",
                 href="https://mdatools.com/prep/", target="_blank",
                 class_="btn-success-premium w-100 text-center",
                 style="text-decoration:none;padding:15px;"),
            ui.a("Principal components analysis (PCA)",
                 href="https://mdatools.com/pca/", target="_blank",
                 class_="btn-success-premium w-100 text-center",
                 style="text-decoration:none;padding:15px;"),
            ui.a("Partial least squares regression (PLSR)",
                 href="https://mdatools.com/pls/", target="_blank",
                 class_="btn-success-premium w-100 text-center",
                 style="text-decoration:none;padding:15px;"),
            col_widths=(4, 4, 4),
            class_="mt-3 mb-4",
        ),
        ui.p("Features:"),
        ui.tags.ul(
            ui.tags.li("Spectral visualization and preprocessing"),
            ui.tags.ul(
                ui.tags.li("Original spectra visualization."),
                ui.tags.li("Savitzky-Golay filter and derivatives."),
                ui.tags.li("Normalization via SNV, area, length, and variable."),
                ui.tags.li("Scaling, baseline correction, spike removal."),
            ),
            ui.tags.li("Principal components analysis (PCA)"),
            ui.tags.li("Partial least squares regression (PLSR)"),
        ),
        ui.p("For the ", ui.tags.b("PLSR tool"), " use these CSV settings:"),
        ui.tags.ul(
            ui.tags.li(ui.tags.b("Delimiter: "), "`,`"),
            ui.tags.li(ui.tags.b("Row labels: "), "`yes`"),
            ui.tags.li(ui.tags.b("Header: "), "`values`"),
            ui.tags.li(ui.tags.b("Header name (predictors): "), "`wavelength` or `wavenumber`"),
            ui.tags.li(ui.tags.b("Header units (predictors): "), "`nm` or `cm`"),
        ),
    ),
)


# --------------------------------------------------------
# Tab: About
# --------------------------------------------------------
about_content = ui.div(
    _premium_card(
        ui.h3("About this app"),
        ui.p("This application was developed to streamline the processing of the "
             "Open Soil Spectral Library (OSSL) for chemometric modeling."),
        ui.p(ui.a("Soil Spectroscopy for Global Good",
                  href="https://soilspectroscopy.org/", target="_blank"),
             " was founded by Woodwell Climate Research Center, University of "
             "Florida and OpenGeoHub in 2020."),
        ui.p("Originally funded by the USDA National Institute of Food and "
             "Agriculture Award #2020-67021-32467."),
        ui.p("Currently maintained by Fund for Climate Solutions from Woodwell Climate."),
        ui.p("For questions, suggestions, bug reports, and inquiries:   "),
        ui.p(ui.a("soilspec4gg@woodwellclimate.org", href="mailto:soilspec4gg@woodwellclimate.org", target="_blank")),
        ui.hr(),
        ui.markdown("""
### Credits & data sources
* **Data:** [Open Soil Spectral Library (OSSL)](https://docs.soilspectroscopy.org/)
* **Engine:** Powered by [DuckDB](https://duckdb.org/) and [Shiny for Python](https://shiny.posit.co/py/)
* **Chemometrics:** Integration with [mdatools](https://mdatools.com/)
        """),
        ui.hr(),
        ui.h3("Citation"),
        ui.markdown("""
If you use data from this platform in your work, please cite the OSSL:

Safanelli, J. L., Hengl, T., Parente, L. L., Minarik, R., Bloom, D. E.,
Todd-Brown, K., Gholizadeh, A., Mendes, W. de S., & Sanderman, J. (2025).
Open Soil Spectral Library (OSSL): Building reproducible soil calibration
models through open development and community engagement.
*PLOS ONE*, 20(1), e0296545. https://doi.org/10.1371/journal.pone.0296545
        """),
        ui.p("Version: 1.4 | Updated: May 2026", class_="text-muted mt-4"),
    ),
)


# --------------------------------------------------------
# App UI
# --------------------------------------------------------
app_ui = ui.page_fluid(

    ui.tags.head(ui.tags.style("""
        body { font-family: 'Inter', sans-serif; background-color: #f8f9fa; }
        .premium-card {
            background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px; border: 1px solid #e9ecef;
        }
        .btn-premium {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; font-weight: 600;
            padding: 10px 20px; border-radius: 8px; transition: transform 0.2s;
        }
        .btn-premium:hover { transform: translateY(-2px); color: white; opacity: 0.9; }
        .btn-success-premium {
            background: linear-gradient(135deg, #20bf55 0%, #01baef 100%);
            color: white; border: none; font-weight: 600;
            padding: 10px 20px; border-radius: 8px;
        }
        .section-title {
            font-size: 1.1rem; font-weight: 600; color: #4a5568;
            margin-bottom: 1rem; border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }
        .subset-panel {
            border: 2px dashed #667eea; border-radius: 12px;
            padding: 20px; margin-bottom: 20px; background: #f5f4ff;
        }
        /* Workflow banner */
        .workflow-banner {
            background: linear-gradient(135deg, #f0f4ff 0%, #f5f0ff 100%);
            border: 1px solid #d6d0f5; border-radius: 12px;
            padding: 14px 24px; margin-bottom: 18px;
            display: flex; align-items: center; justify-content: center;
            flex-wrap: wrap; gap: 0; font-size: 0.88rem; color: #4a5568;
        }
        .workflow-step {
            display: flex; align-items: center; gap: 6px;
        }
        .workflow-step .step-num {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 50%; width: 22px; height: 22px;
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
        }
        .workflow-step .step-label { font-weight: 600; color: #2c3e50; }
        .workflow-arrow {
            margin: 0 10px; color: #a0aec0; font-size: 1rem;
        }
        /* Advanced filtering disclosure */
        .adv-filter-toggle {
            background: none; border: none; color: #667eea; font-size: 0.88rem;
            font-weight: 600; cursor: pointer; padding: 4px 0;
            display: flex; align-items: center; gap: 4px;
        }
        .adv-filter-toggle:hover { color: #764ba2; }
        /* Sample download buttons — teal outline */
        .btn-sample {
            border: 1.5px solid #1d9e75 !important; color: #0f6e56 !important;
            background: white !important; border-radius: 8px !important;
            font-size: 12px; font-weight: 600; padding: 5px 12px;
            transition: background 0.15s;
        }
        .btn-sample:hover {
            background: #e1f5ee !important; color: #085041 !important;
        }
    """)),

    ui.div(
        ui.img(src="https://raw.githubusercontent.com/soilspectroscopy/ossl-imports"
               "/main/img/soilspec4gg-logo_square.png",
               style="height:60px;margin-right:15px;vertical-align:middle;"),
        ui.h2("Chemometrics with the OSSL",
              style="display:inline-block;vertical-align:middle;"
                    "margin-top:15px;font-weight:700;color:#2c3e50;"),
        class_="text-center mb-4 mt-3",
    ),

    ui.div(
        ui.div(
            ui.div(
                ui.span("1", class_="step-num"),
                ui.span("Overview", class_="step-label"),
                ui.span("listed libraries", style="color:#718096"),
                class_="workflow-step",
            ),
            ui.span("→", class_="workflow-arrow"),
            ui.div(
                ui.span("2", class_="step-num"),
                ui.span("Prepare", class_="step-label"),
                ui.span("filter, join & export your extract", style="color:#718096"),
                class_="workflow-step",
            ),
            ui.span("→", class_="workflow-arrow"),
            ui.div(
                ui.span("3", class_="step-num"),
                ui.span("My Data", class_="step-label"),
                ui.span("resample your own spectra (optional)", style="color:#718096"),
                class_="workflow-step",
            ),
            ui.span("→", class_="workflow-arrow"),
            ui.div(
                ui.span("4", class_="step-num"),
                ui.span("Analyse", class_="step-label"),
                ui.span("open mdatools in a new tab", style="color:#718096"),
                class_="workflow-step",
            ),
            class_="workflow-banner",
        ),
        style="padding: 0 0 4px 0;",
    ),

    ui.navset_tab(
        ui.nav_panel("Overview",  data_overview_content),
        ui.nav_panel("Prepare",   data_selection_content),
        ui.nav_panel("My Data",   formatting_content),
        ui.nav_panel("Analyse",   chemometrics_content),
        ui.nav_panel("About",     about_content),
    ),

    ui.div(
        ui.hr(),
        ui.p("© 2026 Soil Spectroscopy for Global Good. Distributed under MIT License."),
        ui.p("Data provided by the OSSL may be subject to individual dataset licenses. "
             "Please cite the original authors."),
        ui.p(
            ui.a("OSSL Manual", href="https://docs.soilspectroscopy.org/",
                 target="_blank", style="color:#667eea;margin-right:16px;"),
            ui.a("GitHub", href="https://github.com/soilspectroscopy",
                 target="_blank", style="color:#667eea;margin-right:16px;"),
            ui.a("mdatools", href="https://mdatools.com",
                 target="_blank", style="color:#667eea;"),
            style="margin-top:4px;font-size:0.88rem;",
        ),
        class_="footer",
        style="text-align:center;padding:16px 0 24px;color:#718096;font-size:0.85rem;",
    ),
)


# --------------------------------------------------------
# Server
# --------------------------------------------------------
def server(input, output, session):

    # ---- Reactive state ------------------------------------------------
    site_cols_rv   = reactive.Value([])
    lab_cols_rv    = reactive.Value([])
    spec_cols_rv   = reactive.Value([])   # spectral *metadata* cols only

    unique_site_levels = reactive.Value({})
    unique_spec_levels = reactive.Value({})
    joined_data        = reactive.Value(None)
    fmt_result         = reactive.Value(None)   # for Formatting tab

    # Cached Arrow tables (avoid re-downloading within a session)
    site_arrow = reactive.Value(None)
    lab_arrow  = reactive.Value(None)
    spec_arrow = reactive.Value(None)

    # DuckDB connection — created on first "Load data" click
    con_rv = reactive.Value(None)

    # ---- Explore tab ---------------------------------------------------
    @render.data_frame
    def datasets_overview_table():
        if overview_df.empty:
            return render.DataGrid(pd.DataFrame({"Status": ["Could not load dataset list."]}))
        cols = ["new_code", "description", "extent", "sample_size", "spectral_range"]
        available = [c for c in cols if c in overview_df.columns]
        if not available:
            return render.DataGrid(pd.DataFrame({"Status": ["Expected columns not found."]}))
        display = (
            overview_df[available]
            .drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={
                "new_code":       "Dataset code",
                "description":    "Description",
                "extent":         "Extent",
                "sample_size":    "Sample size",
                "spectral_range": "Spectral range",
            })
        )
        return render.DataGrid(display, width="100%", height="400px", filters=True)

    # ---- Spectra type selector -----------------------------------------
    @render.ui
    def spectra_selector():
        d_code = input.dataset_code()
        if not d_code:
            return ui.p("Select a dataset first.")

        # Fast path: use metadata JSON
        if METADATA and d_code in METADATA.get("datasets", {}):
            types = METADATA["datasets"][d_code].get("spectra_types", [])
        else:
            if urls_df.empty:
                return ui.p("No dataset information available.")
            subset = urls_df[urls_df["dataset_code"] == d_code]
            types = []
            for fname in subset["ossl_file"]:
                if "_mir_"    in fname.lower(): types.append("mir")
                if "_visnir_" in fname.lower(): types.append("visnir")
                elif "_nir_"  in fname.lower(): types.append("nir")
            types = list(set(types))

        return ui.input_select("spectra_type", "Spectra type",
                                choices=sorted(types), width="100%")

    # ---- Load metadata (deferred heavy imports) -------------------------
    @reactive.Effect
    @reactive.event(input.load_metadata)
    def _load_metadata():
        # Heavy imports deferred until here — Pyodide fetches the wheels
        # only on first click, after the UI is already interactive.
        import duckdb  # pyrefly: ignore [missing-import]

        d_code = input.dataset_code()
        s_type = input.spectra_type()

        if not d_code or not s_type:
            ui.notification_show("Select dataset and spectra type.", type="warning")
            return

        # Resolve URLs
        if METADATA and d_code in METADATA.get("datasets", {}):
            ds = METADATA["datasets"][d_code]
            site_url = ds["files"].get("soilsite", {}).get("url")
            lab_url  = ds["files"].get("soillab",  {}).get("url")
            spec_url = ds["files"].get(s_type,     {}).get("url")
        else:
            # Fallback: derive URLs from the CSV
            subset = urls_df[urls_df["dataset_code"] == d_code]
            def _first_url(mask): 
                rows = subset[mask & subset["ossl_file"].str.contains(".parquet")]
                return rows["public_url"].values[0] if len(rows) else None

            site_url = _first_url(subset["ossl_file"].str.contains("soilsite"))
            lab_url  = _first_url(subset["ossl_file"].str.contains("soillab"))
            spec_url = _first_url(subset["ossl_file"].str.contains(f"_{s_type}_"))

        if not all([site_url, lab_url, spec_url]):
            ui.notification_show("Could not resolve URLs for site, lab, or spectra.",
                                  type="error")
            return

        con = duckdb.connect(":memory:")

        with ui.Progress(min=1, max=4) as p:
            try:
                p.set(1, message="Downloading site data…")
                s_table = fetch_parquet_as_arrow(site_url)
                con.register("site_view", s_table)
                site_arrow.set(s_table)

                p.set(2, message="Downloading lab data…")
                l_table = fetch_parquet_as_arrow(lab_url)
                con.register("lab_view", l_table)
                lab_arrow.set(l_table)

                p.set(3, message="Downloading spectra data…")
                sp_table = fetch_parquet_as_arrow(spec_url)
                con.register("spec_view", sp_table)
                spec_arrow.set(sp_table)

                # Populate column lists
                # Fast path: read from metadata JSON (no parquet schema parse needed)
                if METADATA and d_code in METADATA.get("datasets", {}):
                    ds = METADATA["datasets"][d_code]
                    s_cols  = sorted(ds["files"].get("soilsite", {}).get("columns", []))
                    l_cols  = sorted(ds["files"].get("soillab",  {}).get("columns", []))
                    sp_meta = sorted(ds["files"].get(s_type,     {}).get("meta_columns", []))
                    sp_info = ds["files"].get(s_type, {})
                    scan_min = sp_info.get("scan_min")
                    scan_max = sp_info.get("scan_max")
                else:
                    # Fallback: derive from Arrow schema
                    s_cols  = sorted(s_table.schema.names)
                    l_cols  = sorted(l_table.schema.names)
                    sp_names = sp_table.schema.names
                    scan_vals = []
                    for c in sp_names:
                        if str(c).startswith("scan_"):
                            num = re.sub(r"^scan_.*?\.", "", str(c))
                            num = re.sub(r"_(abs|ref|bc\.abs)$", "", num)
                            try: scan_vals.append(float(num))
                            except: pass
                    scan_min = min(scan_vals) if scan_vals else None
                    scan_max = max(scan_vals) if scan_vals else None
                    skip = {"id.layer_local_c","id.scan_local_c",
                            "id.layer_uuid_c","id.layer_uuid"}
                    sp_meta = sorted(
                        c for c in sp_names
                        if not str(c).startswith("scan_") and c not in skip
                    )

                if scan_min is not None:
                    ui.update_numeric("spec_min", value=scan_min)
                if scan_max is not None:
                    ui.update_numeric("spec_max", value=scan_max)

                site_cols_rv.set(s_cols)
                lab_cols_rv.set(l_cols)
                spec_cols_rv.set(sp_meta)

                # Store connection in a reactive value so other effects can read it
                con_rv.set(con)

                p.set(4, message="Done!")
                ui.notification_show("Metadata loaded successfully!", type="message")

            except Exception as e:
                import traceback
                ui.notification_show(f"Error loading metadata: {traceback.format_exc()}",
                                      type="error", duration=20)

    def _get_con():
        """Return the DuckDB connection, or None if not yet loaded."""
        return con_rv()

    # ---- Column selector UIs -------------------------------------------
    @render.ui
    def site_column_selector():
        cols = site_cols_rv()
        if not cols:
            return ui.p("Load metadata first.", class_="text-muted")
        return ui.input_selectize("site_cols", "Site columns",
                                   choices=cols, multiple=True, width="100%")

    @render.ui
    def lab_column_selector():
        cols = lab_cols_rv()
        if not cols:
            return ui.p("Load dataset contents first.", class_="text-muted")
        return ui.input_selectize("lab_cols", "Columns",
                                   choices=cols, selected=cols[:1],
                                   multiple=True, width="100%")

    @render.ui
    def spec_column_selector():
        cols = spec_cols_rv()
        if not cols:
            return ui.p("No metadata available.", class_="text-muted")
        return ui.input_selectize("spec_cols", "Columns",
                                   choices=cols, multiple=True, width="100%")

    # ---- Site level filtering ------------------------------------------
    @reactive.Effect
    @reactive.event(input.site_cols)
    def _fetch_site_levels():
        selected = input.site_cols()
        if not selected:
            unique_site_levels.set({})
            return
        con = _get_con()
        if con is None:
            return
        levels = {}
        cat_cols = [c for c in selected
                    if c.endswith(("_c","_txt","_uint16","_id","_logical","_code"))]
        if cat_cols:
            with ui.Progress(min=1, max=len(cat_cols)) as p:
                p.set(message="Extracting site levels…")
                for col in cat_cols:
                    try:
                        res = con.execute(
                            f'SELECT DISTINCT "{col}" FROM site_view '
                            f'WHERE "{col}" IS NOT NULL LIMIT 500'
                        ).fetchall()
                        levels[col] = [str(r[0]) for r in res]
                    except Exception as e:
                        print(f"Skipping unique values for {col}: {e}")
        unique_site_levels.set(levels)

    @render.ui
    def site_level_filters():
        selected = input.site_cols()
        levels_dict = unique_site_levels()
        if not selected:
            return ui.p("Select site columns to see filters.", class_="text-muted")
        filters = []
        for col in selected:
            if col.endswith(("_c","_txt","_uint16","_id","_logical","_code")):
                if col in levels_dict and levels_dict[col]:
                    safe_id = "site_filter_" + re.sub(r"\W+", "_", col)
                    filters.append(
                        ui.input_selectize(safe_id, f"Filter: {col}",
                                           choices=levels_dict[col],
                                           multiple=True, width="100%")
                    )
            elif col.endswith(("_cm","_dd","_m")):
                safe_id_min = "site_filter_min_" + re.sub(r"\W+", "_", col)
                safe_id_max = "site_filter_max_" + re.sub(r"\W+", "_", col)
                filters.append(
                    ui.div(
                        ui.markdown(f"Range: {col}"),
                        ui.layout_columns(
                            ui.input_numeric(safe_id_min, "Min", value=None),
                            ui.input_numeric(safe_id_max, "Max", value=None),
                            col_widths=(6, 6),
                        ),
                        class_="mb-3 border-bottom pb-2",
                    )
                )
        if not filters:
            return ui.p("Select other columns to filter.", class_="text-muted")
        return ui.div(*filters)

    # ---- Spectra metadata level filtering ------------------------------
    @reactive.Effect
    @reactive.event(input.spec_cols)
    def _fetch_spec_levels():
        selected = input.spec_cols()
        if not selected:
            unique_spec_levels.set({})
            return
        con = _get_con()
        if con is None:
            return
        levels = {}
        with ui.Progress(min=1, max=10) as p:
            p.set(message="Extracting spectra levels…")
            for col in selected:
                try:
                    res = con.execute(
                        f'SELECT DISTINCT "{col}" FROM spec_view '
                        f'WHERE "{col}" IS NOT NULL LIMIT 100'
                    ).fetchall()
                    levels[col] = [str(r[0]) for r in res]
                except Exception as e:
                    print(f"Skipping unique values for {col}: {e}")
        unique_spec_levels.set(levels)

    @render.ui
    def spec_level_filters():
        selected = input.spec_cols()
        levels_dict = unique_spec_levels()
        if not selected or not levels_dict:
            return ui.p("Select spectra columns to see unique levels.",
                         class_="text-muted")
        filters = []
        for col in selected:
            if col in levels_dict and levels_dict[col]:
                safe_id = "spec_filter_" + re.sub(r"\W+", "_", col)
                filters.append(
                    ui.input_selectize(safe_id, f"Filter: {col}",
                                       choices=levels_dict[col],
                                       multiple=True, width="100%")
                )
        if not filters:
            return ui.p("No discrete levels found.", class_="text-muted")
        return ui.div(*filters)

    # ---- SQL query builders --------------------------------------------
    def _build_site_where(site_selected):
        site_filters = []
        for col in site_selected:
            suffix = re.sub(r"\W+", "_", col)
            if col.endswith(("_c","_txt","_uint16","_id","_logical","_code")):
                try:
                    lvls = input["site_filter_" + suffix]()
                    if lvls:
                        fmtd = ", ".join(f"'{x}'" for x in lvls)
                        site_filters.append(f'"{col}"::VARCHAR IN ({fmtd})')
                except: pass
            elif col.endswith(("_cm","_dd","_m")):
                try:
                    lo = input["site_filter_min_" + suffix]()
                    hi = input["site_filter_max_" + suffix]()
                    if lo is not None: site_filters.append(f'"{col}"::DOUBLE >= {lo}')
                    if hi is not None: site_filters.append(f'"{col}"::DOUBLE <= {hi}')
                except: pass
        return ("WHERE " + " AND ".join(site_filters)) if site_filters else ""

    def _build_lab_selects_and_where(lab_selected, id_col, transform):
        selects = [f'"{id_col}"']
        where_clauses = []
        for c in lab_selected:
            if c == id_col:
                continue
            where_clauses.append(f'"{c}" IS NOT NULL')
            if transform == "log1p":
                selects.append(f'LN(1 + "{c}"::DOUBLE) AS "{c}_{transform}"')
            elif transform == "sqrt":
                selects.append(f'SQRT("{c}"::DOUBLE) AS "{c}_{transform}"')
            else:
                selects.append(f'"{c}"')
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        return selects, where

    def _build_site_select(site_selected, id_col):
        """Build SELECT for site including chosen metadata columns."""
        cols = [f'"{id_col}"']
        for c in site_selected:
            if c != id_col:
                cols.append(f'"{c}"')
        return ", ".join(cols)

    def _build_spec_where(spec_selected):
        filters = []
        for col in spec_selected:
            safe_id = "spec_filter_" + re.sub(r"\W+", "_", col)
            try:
                lvls = input[safe_id]()
                if lvls:
                    fmtd = ", ".join(f"'{x}'" for x in lvls)
                    filters.append(f'"{col}"::VARCHAR IN ({fmtd})')
            except: pass
        return ("WHERE " + " AND ".join(filters)) if filters else ""

    def _validate_filters():
        site_selected = list(input.site_cols() or [])
        for col in site_selected:
            if col.endswith(("_cm","_dd","_m")):
                suffix = re.sub(r"\W+", "_", col)
                try:
                    lo = input["site_filter_min_" + suffix]()
                    hi = input["site_filter_max_" + suffix]()
                    if lo is None and hi is None:
                        return False, (f"Please provide at least one bound (min or max) "
                                       f"for '{col}', or deselect it.")
                except:
                    return False, f"Could not read filter values for '{col}'"
        return True, ""

    # ---- Soil-only preview ---------------------------------------------
    @reactive.Effect
    @reactive.event(input.run_soil_only, input.run_soil_site)
    def _run_soil_only():
        con = _get_con()
        lab_selected = list(input.lab_cols() or [])
        if not lab_selected:
            ui.notification_show("Select at least one soil property.", type="error")
            return
        ok, msg = _validate_filters()
        if not ok:
            ui.notification_show(msg, type="error", duration=10)
            return
        if con is None or site_arrow() is None or lab_arrow() is None:
            ui.notification_show("Ensure metadata is loaded.", type="warning")
            return

        id_col        = input.id_col()
        site_selected = list(input.site_cols() or [])
        transform     = input.transform_type()

        with ui.Progress(min=1, max=3) as p:
            p.set(1, message="Processing soil data…")
            try:
                site_select = _build_site_select(site_selected, id_col)
                site_where  = _build_site_where(site_selected)
                site_query  = f"SELECT {site_select} FROM site_view {site_where}"

                lab_selects, lab_where = _build_lab_selects_and_where(
                    lab_selected, id_col, transform)
                lab_query = (f"SELECT {', '.join(lab_selects)} "
                             f"FROM lab_view {lab_where}")

                final_query = f"""
                    SELECT l.*, s_extra.*
                    FROM ({lab_query}) l
                    INNER JOIN (
                        SELECT {site_select} FROM site_view {site_where}
                    ) s_extra ON l."{id_col}" = s_extra."{id_col}"
                """
                # Simpler join: site cols first, then lab result
                final_query = f"""
                    SELECT l.*
                    FROM ({
                        f'SELECT "{id_col}" FROM site_view {site_where}'
                    }) s
                    INNER JOIN ({lab_query}) l ON s."{id_col}" = l."{id_col}"
                """

                # If site cols requested, fetch them separately and merge
                p.set(2, message="Collecting…")
                result = con.execute(final_query).df()

                if site_selected:
                    site_df = con.execute(
                        f"SELECT {site_select} FROM site_view {site_where}"
                    ).df()
                    result = result.merge(site_df, on=id_col, how="left",
                                          suffixes=("", "_site_dup"))
                    result = result[[c for c in result.columns
                                     if not c.endswith("_site_dup")]]

                joined_data.set(result)
                p.set(3, message="Done!")
                ui.notification_show(f"Soil data processed! {result.shape[0]} rows.",
                                      type="message")
            except Exception as e:
                ui.notification_show(f"Error: {e}", type="error")

    # ---- Full join (soil + spectra) ------------------------------------
    @reactive.Effect
    @reactive.event(input.run_join)
    def _run_join():
        con = _get_con()
        ok, msg = _validate_filters()
        if not ok:
            ui.notification_show(msg, type="error", duration=10)
            return
        if con is None or site_arrow() is None or lab_arrow() is None or spec_arrow() is None:
            ui.notification_show("Ensure metadata is loaded.", type="warning")
            return

        id_col = input.id_col()
        def _safe(name):
            try: return list(input[name]() or [])
            except: return []

        site_selected = _safe("site_cols")
        lab_selected  = _safe("lab_cols")
        spec_selected = _safe("spec_cols")
        j_type    = input.join_type()
        transform = input.transform_type()
        interval  = int(input.spec_interval())
        s_min     = input.spec_min()
        s_max     = input.spec_max()

        with ui.Progress(min=1, max=4) as p:
            try:
                p.set(1, message="Building queries…")
                site_select = _build_site_select(site_selected, id_col)
                site_where  = _build_site_where(site_selected)
                lab_selects, lab_where = _build_lab_selects_and_where(
                    lab_selected, id_col, transform)

                p.set(2, message="Filtering soil data…")
                final_lab_query = f"""
                    SELECT l.*
                    FROM (SELECT "{id_col}" FROM site_view {site_where}) s
                    INNER JOIN (
                        SELECT {', '.join(lab_selects)} FROM lab_view {lab_where}
                    ) l ON s."{id_col}" = l."{id_col}"
                """
                lab_result = con.execute(final_lab_query).df()

                # Attach site columns
                if site_selected:
                    site_df = con.execute(
                        f"SELECT {site_select} FROM site_view {site_where}"
                    ).df()
                    lab_result = lab_result.merge(site_df, on=id_col, how="left",
                                                   suffixes=("", "_site_dup"))
                    lab_result = lab_result[[c for c in lab_result.columns
                                             if not c.endswith("_site_dup")]]

                ui.notification_show(f"Lab rows: {len(lab_result)}", type="message")

                p.set(3, message="Fetching spectra…")
                sp_table     = spec_arrow()
                sp_all_names = sp_table.schema.names
                scan_cols    = [c for c in sp_all_names if str(c).startswith("scan_")]

                # Build rename dict and filter by range + interval
                rename_dict = {}
                for c in scan_cols:
                    num = re.sub(r"^scan_.*?\.", "", str(c))
                    num = re.sub(r"_(abs|ref|bc\.abs)$", "", num)
                    rename_dict[str(c)] = num

                filtered_scan = []
                for c in scan_cols:
                    try:
                        val = float(rename_dict[c])
                    except ValueError:
                        continue
                    if s_min <= val <= s_max:
                        if (val % interval) < 0.001 or (interval - val % interval) < 0.001:
                            filtered_scan.append(c)

                new_scan_names = [rename_dict[c] for c in filtered_scan]

                spec_selects = [f'"{id_col}"']
                for c in spec_selected:
                    if c != id_col:
                        spec_selects.append(f'"{c}"')
                for c in filtered_scan:
                    spec_selects.append(f'"{c}" AS "{rename_dict[c]}"')

                spec_where   = _build_spec_where(spec_selected)
                spec_result  = con.execute(
                    f"SELECT {', '.join(spec_selects)} FROM spec_view {spec_where}"
                ).df()
                ui.notification_show(f"Spec rows: {len(spec_result)}", type="message")

                p.set(4, message="Merging…")
                how_map = {"inner": "inner", "full": "outer", "left": "left"}
                result = lab_result.merge(spec_result, on=id_col,
                                           how=how_map[j_type],
                                           suffixes=("", "_spec_dup"))
                result = result[[c for c in result.columns
                                  if not c.endswith("_spec_dup")]]

                try:
                    sorted_scan = sorted(new_scan_names, key=float)
                except ValueError:
                    sorted_scan = sorted(new_scan_names)

                non_scan = [c for c in result.columns if c not in sorted_scan]
                result   = result[non_scan + sorted_scan]

                if result.empty:
                    ui.notification_show(
                        "Join produced 0 rows — check ID column and filters.",
                        type="warning")
                    return

                joined_data.set(result)
                ui.notification_show(
                    f"Join complete! {result.shape[0]} rows, {result.shape[1]} cols.",
                    type="message")

            except Exception as e:
                import traceback
                ui.notification_show(f"Join error: {traceback.format_exc()}",
                                      type="error", duration=20)

    # ---- Summary stats (excludes non-numeric / site cols) --------------
    @render.table
    def summary_stats_table():
        import numpy as np
        df = joined_data()
        if df is None:
            return pd.DataFrame({"Info": ["Click 'Preview' to see summary statistics."]})

        lab_selected = list(input.lab_cols() or [])
        id_col    = input.id_col()
        transform = input.transform_type()

        target_cols = []
        for c in lab_selected:
            if c == id_col:
                continue
            col_name = f"{c}_{transform}" if transform != "none" else c
            if col_name in df.columns:
                target_cols.append(col_name)

        if not target_cols:
            return pd.DataFrame({"Info": ["No soil properties selected for summary."]})

        stats_df = df[target_cols].select_dtypes(include=[np.number])
        if stats_df.empty:
            return pd.DataFrame({"Info": ["Selected properties are not numeric."]})

        stats = stats_df.describe().T
        stats["median"]   = stats_df.median()
        stats["iqr"]      = stats["75%"] - stats["25%"]
        stats["skewness"] = stats_df.skew()
        stats["kurtosis"] = stats_df.kurtosis()

        return (
            stats[["count","min","mean","std","median","iqr","max","skewness","kurtosis"]]
            .reset_index()
            .rename(columns={"index": "Property"})
            .round(4)
        )

    # ---- Preview table -------------------------------------------------
    @render.data_frame
    def preview_table():
        df = joined_data()
        if df is None:
            return render.DataGrid(pd.DataFrame(
                {"Status": ["Waiting for data join…"]}))
        preview = df.head(10).iloc[:, :10].copy()
        num_cols = preview.select_dtypes(include="number").columns
        preview[num_cols] = preview[num_cols].round(5)
        r, c = df.shape
        if c > 10:
            preview["…"] = "…"
        if r > 10:
            preview.loc[len(preview)] = ["…"] * preview.shape[1]
        return render.DataGrid(preview)

    # ---- Inline spectrum SVG plot --------------------------------------
    @render.ui
    def spectrum_preview():
        df = joined_data()
        if df is None:
            return None
        svg = _make_spectrum_svg(df)
        if svg is None:
            return None
        return ui.HTML(svg)

    # ---- Export column selector ----------------------------------------
    @render.ui
    def export_column_selector():
        df = joined_data()
        if df is None:
            return None
        metadata_cols = [c for c in df.columns if not _is_spectral(c)]
        return ui.div(
            ui.input_selectize(
                "export_cols", "Columns to export",
                choices=metadata_cols, selected=metadata_cols,
                multiple=True, width="100%",
            ),
            ui.p("All spectral bands are included automatically.",
                 class_="text-muted small mb-3"),
        )

    # ---- Subsetting panel (shown after join) ---------------------------
    @render.ui
    def subsetting_panel():
        df = joined_data()
        if df is None:
            return None

        site_selected = list(input.site_cols() or [])
        group_choices = {"": "— none —"}
        # Only offer categorical site cols as stratification options
        for c in site_selected:
            if c.endswith(("_c","_txt","_uint16","_id","_logical","_code")):
                group_choices[c] = c

        return ui.div(
            ui.div(
                ui.div("6. Subsetting (optional)", class_="section-title"),
                ui.p(
                    f"Dataset has {df.shape[0]:,} rows × {df.shape[1]:,} cols. "
                    "Large datasets (>50 k rows) may be slow to load into mdatools. "
                    "Use random or stratified sampling to reduce size.",
                    class_="text-muted small",
                ),
                ui.layout_columns(
                    ui.div(
                        ui.input_radio_buttons(
                            "subset_type", "Sampling method",
                            choices={"random": "Random",
                                     "stratified": "Stratified by group"},
                            selected="random", inline=True,
                        ),
                        ui.input_select(
                            "subset_group_col", "Group column (stratified only)",
                            choices=group_choices, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric("subset_n", "Number of rows (0 = keep all)",
                                         value=0, min=0),
                        ui.input_numeric("subset_pct", "Or: percentage of rows (0 = use N above)",
                                         value=0, min=0, max=100),
                        ui.input_action_button("run_subset", "Apply subsetting",
                                               class_="btn-premium w-100 mt-2"),
                    ),
                    col_widths=(6, 6),
                ),
                class_="premium-card subset-panel",
            )
        )

    @reactive.Effect
    @reactive.event(input.run_subset)
    def _run_subset():
        df = joined_data()
        if df is None:
            ui.notification_show("No data to subset — run a join first.", type="warning")
            return

        n   = input.subset_n()
        pct = input.subset_pct()

        if pct and pct > 0:
            n_target = max(1, int(len(df) * pct / 100))
        elif n and n > 0:
            n_target = int(n)
        else:
            ui.notification_show("Set N or percentage > 0.", type="warning")
            return

        if n_target >= len(df):
            ui.notification_show("Requested N ≥ dataset size — nothing to do.", type="info")
            return

        method     = input.subset_type()
        group_col  = input.subset_group_col()

        try:
            if method == "stratified" and group_col:
                sampled = (
                    df.groupby(group_col, group_keys=False)
                    .apply(lambda g: g.sample(
                        n=max(1, round(n_target * len(g) / len(df))),
                        random_state=42,
                    ))
                )
            else:
                sampled = df.sample(n=n_target, random_state=42)

            joined_data.set(sampled.reset_index(drop=True))
            ui.notification_show(
                f"Subset applied: {sampled.shape[0]:,} rows retained.", type="message")
        except Exception as e:
            ui.notification_show(f"Subsetting error: {e}", type="error")

    # ---- Download CSV --------------------------------------------------
    @render.download(
        filename=lambda: (
            input.dl_filename()
            if input.dl_filename().endswith(".csv")
            else input.dl_filename() + ".csv"
        )
    )
    def download_csv():
        df = joined_data()
        if df is None:
            yield "No data available. Please run a join first."
            return

        all_cols      = df.columns.tolist()
        spectral_cols = [c for c in all_cols if _is_spectral(c)]
        keep_meta     = list(input.export_cols() or [])
        final_cols    = keep_meta + spectral_cols
        yield df[final_cols].to_csv(index=False)

    # ---- Sample file downloads ----------------------------------------
    # Must be defined as explicit named functions — Shiny matches output
    # handlers by the function name, not by variable assignment, so a
    # factory returning an inner _handler is never wired to the button.

    @render.download(filename="sample_visnir_data.csv")
    def dl_sample_visnir():
        try:
            raw = _http_get(_SAMPLE_FILES["sample_visnir_data.csv"])
        except Exception as e:
            yield f"# Error fetching sample_visnir_data.csv: {e}\n"
            return
        yield raw.decode("utf-8", errors="replace")

    @render.download(filename="sample_mir_data.csv")
    def dl_sample_mir():
        try:
            raw = _http_get(_SAMPLE_FILES["sample_mir_data.csv"])
        except Exception as e:
            yield f"# Error fetching sample_mir_data.csv: {e}\n"
            return
        yield raw.decode("utf-8", errors="replace")

    @render.download(filename="sample_neospectra_data.csv")
    def dl_sample_neospectra():
        try:
            raw = _http_get(_SAMPLE_FILES["sample_neospectra_data.csv"])
        except Exception as e:
            yield f"# Error fetching sample_neospectra_data.csv: {e}\n"
            return
        yield raw.decode("utf-8", errors="replace")

    # ---- Formatting tab logic ------------------------------------------
    @reactive.Effect
    @reactive.event(input.fmt_run)
    def _fmt_run():
        import numpy as np

        file_info = input.fmt_file()
        if not file_info:
            ui.notification_show("Upload a CSV file first.", type="warning")
            return

        try:
            file_path = file_info[0]["datapath"]
            src = pd.read_csv(file_path)
        except Exception as e:
            ui.notification_show(f"Could not read file: {e}", type="error")
            return

        all_cols = src.columns.tolist()

        # First column is always the sample ID — never treat it as spectral
        # even if its name happens to be numeric.
        id_col   = all_cols[0]
        rest     = all_cols[1:]

        spectral_cols = [c for c in rest if _is_spectral(c)]
        meta_cols     = [id_col] + [c for c in rest if not _is_spectral(c)]

        if not spectral_cols:
            ui.notification_show(
                "No spectral columns detected after the first (ID) column. "
                "Column names must be plain numbers (e.g. 400, 402, 4000).",
                type="error", duration=10)
            return

        # Source axis — sort ascending regardless of original column order.
        # MIR data is commonly stored high→low (4000→600); np.interp requires
        # xp to be increasing, so we always sort here.
        xs_src_unsorted = np.array([float(c) for c in spectral_cols])
        sort_idx        = np.argsort(xs_src_unsorted)
        xs_src          = xs_src_unsorted[sort_idx]
        spectral_cols_sorted = [spectral_cols[i] for i in sort_idx]

        interval = int(input.fmt_interval())

        # Determine output range from the (now sorted) source axis
        fmt_min = input.fmt_min()
        fmt_max = input.fmt_max()
        x_min = float(fmt_min) if fmt_min is not None else float(xs_src[0])
        x_max = float(fmt_max) if fmt_max is not None else float(xs_src[-1])

        if x_min >= x_max:
            ui.notification_show(
                "Min wavelength must be less than max wavelength.", type="error")
            return

        # Build target axis (always increasing)
        xs_target = np.arange(x_min, x_max + interval * 0.01, interval)
        xs_target = xs_target[xs_target <= x_max]

        # Resample each row via linear interpolation on the sorted source axis
        src_vals = src[spectral_cols_sorted].to_numpy(dtype=float)
        out_vals = np.vstack([
            np.interp(xs_target, xs_src, row) for row in src_vals
        ])

        # Format output column names: integers where possible, otherwise floats
        def _fmt_col(v):
            return str(int(v)) if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")

        meta_df  = src[meta_cols].reset_index(drop=True)
        spec_out = pd.DataFrame(out_vals, columns=[_fmt_col(v) for v in xs_target])
        result   = pd.concat([meta_df, spec_out], axis=1)

        fmt_result.set(result)
        ui.notification_show(
            f"Resampled: {result.shape[0]:,} rows, "
            f"{len(xs_target)} spectral bands "
            f"({x_min:.0f}–{x_max:.0f}, every {interval}).",
            type="message",
        )

    @render.ui
    def fmt_preview():
        df = fmt_result()
        if df is None:
            return ui.p("Upload a file and click 'Resample' to preview.",
                         class_="text-muted")

        svg = _make_spectrum_svg(df)
        preview = df.head(5).iloc[:, :8]
        num_cols = preview.select_dtypes(include="number").columns
        preview[num_cols] = preview[num_cols].round(4)
        if df.shape[1] > 8:
            preview["…"] = "…"

        rows_html = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
            for row in preview.values
        )
        header_html = "".join(f"<th>{c}</th>" for c in preview.columns)
        table_html = (
            f'<div style="overflow-x:auto;margin-top:12px">'
            f'<table class="table table-sm table-bordered" style="font-size:12px">'
            f"<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )

        return ui.HTML((svg or "") + table_html)

    @render.ui
    def fmt_download_ui():
        if fmt_result() is None:
            return None
        # Use the original uploaded filename for the download
        file_info = input.fmt_file()
        fname = file_info[0]["name"] if file_info else "resampled_spectra.csv"
        return ui.download_button(
            "download_fmt_csv",
            f"Download {fname}",
            class_="btn-success-premium",
        )

    @render.download(
        filename=lambda: (
            input.fmt_file()[0]["name"]
            if input.fmt_file()
            else "resampled_spectra.csv"
        )
    )
    def download_fmt_csv():
        df = fmt_result()
        if df is None:
            yield "No data."
            return
        yield df.to_csv(index=False)


app = App(app_ui, server)