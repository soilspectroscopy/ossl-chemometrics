# pyrefly: ignore [missing-import]
from shiny import App, render, ui, reactive
# pyrefly: ignore [missing-import]
import duckdb
import pandas as pd
# pyrefly: ignore [missing-import]
import pyarrow.parquet as pq
import io
import re
import sys
from pathlib import Path
import ssl

# --------------------------------------------------------
# Environment setup
# --------------------------------------------------------
IS_WASM = sys.platform == "emscripten"

if not IS_WASM:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

# --------------------------------------------------------
# DuckDB connection (in-memory, per session)
# --------------------------------------------------------
con = duckdb.connect(":memory:")

# --------------------------------------------------------
# Parquet fetching helper — works in both WASM and native
# --------------------------------------------------------
def fetch_parquet_as_arrow(url: str):
    if IS_WASM:
        # pyrefly: ignore [missing-import]
        import js
        # pyrefly: ignore [missing-import]
        from pyodide.ffi import to_js

        # Use synchronous XHR via js bridge — binary-safe
        xhr = js.XMLHttpRequest.new()
        xhr.open("GET", url, False)  # False = synchronous
        xhr.responseType = "arraybuffer"
        xhr.send(None)
        if xhr.status != 200:
            raise RuntimeError(f"HTTP {xhr.status} fetching {url}")
        buf = io.BytesIO(bytes(xhr.response.to_py()))
    else:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            buf = io.BytesIO(resp.read())
    return pq.read_table(buf)

def get_columns(url: str) -> list[str]:
    if IS_WASM:
        # pyrefly: ignore [missing-import]
        import js
        xhr = js.XMLHttpRequest.new()
        xhr.open("GET", url, False)
        xhr.responseType = "arraybuffer"
        xhr.send(None)
        if xhr.status != 200:
            raise RuntimeError(f"HTTP {xhr.status} fetching {url}")
        buf = io.BytesIO(bytes(xhr.response.to_py()))
    else:
        import urllib.request
        with urllib.request.urlopen(url) as resp:
            buf = io.BytesIO(resp.read())
    schema = pq.read_schema(buf)
    return schema.names

# --------------------------------------------------------
# Load local CSV once at startup
# --------------------------------------------------------
try:
    csv_path = Path(__file__).parent / "input" / "ossl_individual_datasets_urls_v1.3.csv"
    urls_df = pd.read_csv(csv_path)
    dataset_codes = urls_df["dataset_code"].unique().tolist()
    dataset_codes.sort()
except Exception as e:
    print(f"Error reading URL list: {e}")
    dataset_codes = []
    urls_df = pd.DataFrame()

try:
    overview_path = Path(__file__).parent / "input" / "ossl_listed_libraries_overview.csv"
    overview_df = pd.read_csv(overview_path)
except Exception as e:
    print(f"Error reading overview CSV: {e}")
    overview_df = pd.DataFrame()

# --------------------------------------------------------
# UI
# --------------------------------------------------------

data_overview_content = ui.div(
    ui.div(
        ui.p("The ", ui.tags.a("Open Soil Spectral Library (OSSL)", href="https://docs.soilspectroscopy.org/", target="_blank"), " is a global compilation of various spectral datasets and reference soil data."),
        ui.p("Standardized soil spectral libraries were created by reformatting to a common data structure and making them accessible via `csv.gz` and `parquet` files."),
        ui.p("Please use this application to download data (", ui.tags.b("Prepare tab"), ") for your research or to prepare it for online chemometric analysis (", ui.tags.b("Chemometrics tab"), ")."),
        ui.p("The OSSL database schema is organized in three distinct tables:"),
        ui.tags.ul(
            ui.tags.li(ui.tags.b("Soilsite: "), "metadata of the sampling sites, locations (if available) and date of measurements (if available)."),
            ui.tags.li(ui.tags.b("Soillab: "), "measured soil properties."),
            ui.tags.li(ui.tags.b("Spectra (nir, visnir, mir): "), "raw spectral data formatted to evenly spaced intervals, across different spectral ranges."),
        ),
        ui.p("A list of currently available datasets is provided below."),
        class_="premium-card"
    ),
    ui.div(
        ui.div("Available datasets", class_="section-title"),
        ui.output_data_frame("datasets_overview_table"),
        class_="premium-card"
    )
)

data_selection_content = ui.div(
    # 1. Data Selection
    ui.div(
        ui.div("1. Select dataset and spectral region", class_="section-title"),
        ui.layout_columns(
            ui.div(
                ui.input_select("dataset_code", "Dataset code", choices=dataset_codes, width="100%"),
            ),
            ui.div(
                ui.output_ui("spectra_selector")
            ),
            ui.div(
                ui.input_action_button("load_metadata", "Load data", class_="btn-premium w-100 mt-4")
            ),
            col_widths=(4, 4, 4)
        ),
        class_="premium-card"
    ),

    # 2. Lab Selection
    ui.div(
        ui.div("2. Soil properties", class_="section-title"),
        ui.p("Select a soil property (or many). Rows with no valid values for the selected property (or combination of properties) will be automatically removed.", class_="text-muted small"),
        ui.output_ui("lab_column_selector"),
        ui.layout_columns(
            ui.div(
                ui.input_radio_buttons("transform_type", "Apply transformation (optional)", choices={"none": "No transformation", "sqrt": "Square root (sqrt)", "log1p": "Log(1+x)"}, selected="none", inline=True),
                ui.input_action_button("run_soil_only", "Preview", class_="btn-premium w-100 mb-2")
            ),
            ui.div(
                ui.p("Summary statistics of selected properties"),
                ui.output_table("summary_stats_table")
            ),
            col_widths=(3, 6)
        ),
        class_="premium-card"
    ),

    # 3. Site Filtering
    ui.div(
        ui.div("3. Site filtering (optional)", class_="section-title"),
        ui.layout_columns(
            ui.div(
                ui.p("Select site columns to filter. Leave empty to keep all rows.", class_="text-muted small"),
                ui.output_ui("site_column_selector"),
            ),
            ui.div(
                ui.p("Highlight the values to keep", class_="text-muted small"),
                ui.output_ui("site_level_filters")
            ),
            col_widths=(6, 6)
        ),
        ui.input_action_button("run_soil_site", "Recalculate statistics", class_="btn-premium w-25 mt-2"),
        class_="premium-card"
    ),

    # 4. Spectral Metadata Filtering
    ui.div(
        ui.div("4. Spectral metadata filtering (when available)", class_="section-title"),
        ui.layout_columns(
            ui.div(
                ui.p("Select columns to filter. Leave empty to keep all rows.", class_="text-muted small"),
                ui.output_ui("spec_column_selector"),
            ),
            ui.div(
                ui.p("Highlight the unique values to keep.", class_="text-muted small"),
                ui.output_ui("spec_level_filters"),
            ),
            col_widths=(6, 6)
        ),
        class_="premium-card"
    ),

    # 5. Join & Export
    ui.div(
        ui.div("5. Join spectra & export", class_="section-title"),
        ui.layout_columns(
            ui.div(
                ui.input_text("id_col", "Common ID Column", value="id.layer_local_c", width="100%"),
                ui.input_select("spec_interval", "Spectral resolution", choices={"2": "Every 2 units", "10": "Every 10 units"}, selected="2", width="100%"),
                ui.layout_columns(
                    ui.input_numeric("spec_min", "Min range", value=0),
                    ui.input_numeric("spec_max", "Max range", value=10000),
                    col_widths=(6, 6)
                ),
                ui.input_select("join_type", "Join type", choices={"inner": "Only data with spectra", "full": "Keep all data, add spectra where available"}, selected="inner", width="100%"),
                ui.input_action_button("run_join", "Join with available spectra", class_="btn-premium w-100 mb-3"),
                ui.output_ui("export_column_selector"),
                ui.input_text("dl_filename", "Filename to save", value="ossl_filtered_data.csv", width="100%"),
                ui.download_button("download_csv", "Download processed dataset", class_="btn-success-premium w-100")
            ),
            ui.div(
                ui.output_data_frame("preview_table")
            ),
            col_widths=(3, 9)
        ),
        class_="premium-card"
    )
)

chemometrics_content = ui.div(
    ui.div(
        ui.h3("mdatools: make chemometrics easy"),
        ui.p("Development and credits: ", ui.tags.a("Sergey Kucheryavskiy", href="https://github.com/svkucheryavski", target="_blank"), "."),
        ui.p("Try most common chemometric methods directly in your browser. All calculations will run on your local computer without sending data or any other information anywhere."),
        ui.p("Check video tutorials at ", ui.tags.a("youtube.com/@mdatools", href="https://www.youtube.com/@mdatools", target="_blank"), ". For more information, please visit the ", ui.tags.a("mdatools website", href="https://mdatools.com", target="_blank"), "."),
        ui.hr(),
        ui.p("Please download your processed dataset from the 'Prepare' tab, and open one of the tools below in a new tab to begin your analysis."),
        ui.p(),
        # Grid for the three buttons
        ui.layout_columns(
            ui.a(
                "Spectral visualization and preprocessing", 
                href="https://mdatools.com/prep/", 
                target="_blank", 
                class_="btn-success-premium w-100 text-center", 
                style="text-decoration: none; padding: 15px;"
            ),
            ui.a(
                "Principal components analysis (PCA)", 
                href="https://mdatools.com/pca/", 
                target="_blank", 
                class_="btn-success-premium w-100 text-center", 
                style="text-decoration: none; padding: 15px;"
            ),
            ui.a(
                "Partial least squares regression (PLSR)", 
                href="https://mdatools.com/pls/", 
                target="_blank", 
                class_="btn-success-premium w-100 text-center", 
                style="text-decoration: none; padding: 15px;"
            ),
            col_widths=(4, 4, 4),
            class_="mt-3 mb-4"
        ),
        ui.p("Features:"),
        ui.tags.ul(
            ui.tags.li("Spectral visualization and preprocessing"),
            ui.tags.ul(
                ui.tags.li("Original spectra visualizazation."),
                ui.tags.li("Savitzky-Golay filter and derivatives."),
                ui.tags.li("Normalization via SNV, area, length, and variable."),
                ui.tags.li("Scaling."),
                ui.tags.li("Baseline correction via EMSC and ALS."),
                ui.tags.li("Trim tails (edges)."),
                ui.tags.li("Spike removal."),
                ui.tags.li("Data and model download."),
            ),
            ui.tags.li("Principal components analysis (PCA)"),
            ui.tags.ul(
                ui.tags.li("Different visualization options."),
                ui.tags.li("Outlier detection and distance metrics."),
                ui.tags.li("Model training, retraining, saving to disk, and uploading for future use."),
            ),
            ui.tags.li("Partial least squares regression (PLSR)"),
            ui.tags.ul(
                ui.tags.li("Model evaluation via cross validation and test set."),
                ui.tags.li("Variable importance metrics."),
                ui.tags.li("Outlier detection and distance metrics."),
                ui.tags.li("Model training, retraining, saving to disk, and uploading for future use."),
            ),
        ),
        ui.p("You can use different subsets of OSSL datasets for training, testing, and applying models."),
        ui.p("Please ensure your `.csv` follows the format required for each tool."),
        ui.p("For example, when using the ", ui.tags.b("PLSR tool"), ", you may want to indicate:"),
        ui.tags.ul(
            ui.tags.li(ui.tags.b("Delimiter: "), "`,`, default"),
            ui.tags.li(ui.tags.b("Row labels: "), "`yes`, this is the id column.",),
            ui.tags.li(ui.tags.b("Header: "), "`values`, indicating the spectral predictors."),
            ui.tags.li(ui.tags.b("Header name (predictors): "), "`wavelength` or `wavenumber`"),
            ui.tags.li(ui.tags.b("Header units (predictors): "), "`nm` or `cm`"),
        ),
        class_="premium-card py-5"
    )   
)

about_content = ui.div(
    ui.div(
        ui.h3("About this app"),
        ui.p("This application was developed to streamline the processing of the Open Soil Spectral Library (OSSL) for chemometric modeling."),
        ui.p(ui.a("Soil Spectroscopy for Global Good", href="https://soilspectroscopy.org/", target="_blank"), " was founded by Woodwell Climate Research Center, University of Florida and OpenGeoHub in 2020."),
        ui.p("Originally funded by the USDA National Institute of Food and Agriculture Award #2020-67021-32467."),
        ui.p("Currently maintained by Fund for Climate Solutions from Woodwell Climate."),
        ui.hr(),
        ui.markdown("""
        ### Credits & data Sources
        * **Data:** [Open Soil Spectral Library (OSSL)](https://docs.soilspectroscopy.org/)
        * **Engine:** Powered by [DuckDB](https://duckdb.org/) and [Shiny for Python](https://shiny.posit.co/py/)
        * **Chemometrics:** Integration with [mdatools](https://mdatools.com/)
        """),
        ui.hr(),
        ui.h3("Citation"),
        ui.markdown("""
        If you use data from this platform in your work, please cite the OSSL in your manuscript as follows:

        Safanelli, J. L., Hengl, T., Parente, L. L., Minarik, R., Bloom, D. E., Todd-Brown, K., Gholizadeh, A., Mendes, W. de S., & Sanderman, J. (2025). Open Soil Spectral Library (OSSL): Building reproducible soil calibration models through open development and community engagement. PLOS ONE, 20(1), e0296545. https://doi.org/10.1371/journal.pone.0296545

        In BibTeX format:

                @article{Safanelli2025,
                title    = "Open Soil Spectral Library ({OSSL)}: Building reproducible soil
                            calibration models through open development and community
                            engagement",
                author   = "Safanelli, Jos{\'e} L and Hengl, Tomislav and Parente, Leandro L
                            and Minarik, Robert and Bloom, Dellena E and Todd-Brown,
                            Katherine and Gholizadeh, Asa and Mendes, Wanderson de Sousa and
                            Sanderman, Jonathan",
                journal  = "PLoS One",
                volume   =  20,
                number   =  1,
                pages    = "e0296545",
                month    =  jan,
                year     =  2025,
                language = "en",
                doi = {10.1371/journal.pone.0296545}
                }
        """),
        ui.p("Version: 1.3 | Updated: May 2026", class_="text-muted mt-4"),
        class_="premium-card"
    )
)

app_ui = ui.page_fluid(
    
    ui.tags.head(
        ui.tags.style("""
            body { font-family: 'Inter', sans-serif; background-color: #f8f9fa; }
            .premium-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e9ecef; }
            .header-title { font-weight: 700; color: #2c3e50; margin-bottom: 1.5rem; }
            .btn-premium { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; font-weight: 600; padding: 10px 20px; border-radius: 8px; transition: transform 0.2s; }
            .btn-premium:hover { transform: translateY(-2px); color: white; opacity: 0.9; }
            .btn-success-premium { background: linear-gradient(135deg, #20bf55 0%, #01baef 100%); color: white; border: none; font-weight: 600; padding: 10px 20px; border-radius: 8px; }
            .section-title { font-size: 1.1rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
        """)
    ),

    ui.div(
        ui.img(src="https://raw.githubusercontent.com/soilspectroscopy/ossl-imports/main/img/soilspec4gg-logo_square.png", style="height: 60px; margin-right: 15px; vertical-align: middle;"),
        ui.h2("Chemometrics with the OSSL", style="display: inline-block; vertical-align: middle; margin-top: 15px; font-weight: 700; color: #2c3e50;"),
        class_="text-center mb-4 mt-3"
    ),

    ui.navset_tab(
        ui.nav_panel("Explore", data_overview_content),
        ui.nav_panel("Prepare", data_selection_content),
        ui.nav_panel("Chemometrics", chemometrics_content),
        ui.nav_panel("About", about_content)
    ),

    ui.div(
        ui.hr(),
        ui.p("© 2026 Soil Spectroscopy for Global Good. Distributed under MIT License."),
        ui.p("Data provided by the OSSL may be subject to individual dataset licenses. Please cite the original authors."),
        class_="footer"
    )
)   

# --------------------------------------------------------
# Server
# --------------------------------------------------------
def server(input, output, session):
    # Reactive state
    site_cols_rv = reactive.Value([])
    lab_cols_rv = reactive.Value([])
    spec_cols_rv = reactive.Value([])

    unique_site_levels = reactive.Value({})
    unique_spec_levels = reactive.Value({})
    joined_data = reactive.Value(None)

    # Store loaded Arrow tables (avoid re-downloading)
    site_arrow = reactive.Value(None)
    lab_arrow = reactive.Value(None)
    spec_arrow = reactive.Value(None)

    # Store raw URLs for reference
    site_url_rv = reactive.Value(None)
    lab_url_rv = reactive.Value(None)
    spec_url_rv = reactive.Value(None)

    # ----------------
    # Listed datasets
    # ----------------
    @render.data_frame
    def datasets_overview_table():
        if overview_df.empty:
            return render.DataGrid(pd.DataFrame({"Status": ["Could not load dataset list."]}))

        cols = ["new_code", "description", "extent", "sample_size", "spectral_range"]
        available = [c for c in cols if c in overview_df.columns]

        if not available:
            return render.DataGrid(pd.DataFrame({"Status": [f"Expected columns not found. Available: {overview_df.columns.tolist()}"]}))

        display = (
            overview_df[available]
            .drop_duplicates()
            .reset_index(drop=True)
            .rename(columns={
                "new_code": "Dataset code",
                "description": "Description",
                "extent": "Extent",
                "sample_size": "Sample size",
                "spectral_range": "Spectral range"
            })
        )

        return render.DataGrid(
            display,
            width="100%",
            height="400px",
            filters=True,
        )
    
    @render.table
    def summary_stats_table():
        import numpy as np
        import pandas as pd
        
        df = joined_data()
        
        if df is None: 
            return pd.DataFrame({"Info": ["Select a dataset, load, select a property (or many), and click 'Preview ' to see summary statistics."]})
        
        # 2. Identify the lab columns user wants to summarize
        # We include the transformation suffix if applicable
        lab_selected = list(input.lab_cols() or [])
        id_col = input.id_col()
        transform = input.transform_type()
        
        # Create the list of expected column names in the final dataframe
        target_cols = []
        for c in lab_selected:
            if c == id_col:
                continue
            # Check for the transformed version of the column name
            col_name = f"{c}_{transform}" if transform != "none" else c
            if col_name in df.columns:
                target_cols.append(col_name)

        # 3. If no lab columns are found (only spectra joined), return info
        if not target_cols:
            return pd.DataFrame({"Info": ["No soil properties selected for summary"]})

        # 4. Filter the dataframe to ONLY these lab columns
        stats_df = df[target_cols].select_dtypes(include=[np.number])
        
        if stats_df.empty:
            return pd.DataFrame({"Info": ["Selected properties are not numeric"]})

        # 5. Calculate statistics only for filtered columns
        stats = stats_df.describe().T
        stats['median'] = stats_df.median()
        stats['iqr'] = stats['75%'] - stats['25%']
        stats['skewness'] = stats_df.skew()
        stats['kurtosis'] = stats_df.kurtosis()
        
        display_cols = [
            "count",
            "min", "mean", "std",
            "median", "iqr", "max",
            "skewness", "kurtosis"
        ]
        
        return (
            stats[display_cols]
            .reset_index()
            .rename(columns={"index": "Property"})
            .round(4)
        )

    # --------------------------------------------------
    # Spectra type selector (populated by dataset choice)
    # --------------------------------------------------
    @render.ui
    def spectra_selector():
        d_code = input.dataset_code()
        if not d_code or urls_df.empty:
            return ui.p("Select a dataset first.")

        subset = urls_df[urls_df["dataset_code"] == d_code]
        spectra_types = []
        for file in subset["ossl_file"]:
            if "_mir_" in file.lower():
                spectra_types.append("mir")
            if "_visnir_" in file.lower():
                spectra_types.append("visnir")
            elif "_nir_" in file.lower():
                spectra_types.append("nir")

        spectra_types = list(set(spectra_types))
        return ui.input_select("spectra_type", "Spectra type", choices=spectra_types, width="100%")

    # --------------------------------------------------
    # Load Metadata — fetch parquet files into PyArrow
    # --------------------------------------------------
    @reactive.Effect
    @reactive.event(input.load_metadata)
    def _load_metadata():
        d_code = input.dataset_code()
        s_type = input.spectra_type()

        if not d_code or not s_type:
            ui.notification_show("Select dataset and spectra type", type="warning")
            return

        subset = urls_df[urls_df["dataset_code"] == d_code]

        site_urls = subset[
            subset["ossl_file"].str.contains("soilsite") &
            subset["ossl_file"].str.contains(".parquet")
        ]["public_url"].values

        lab_urls = subset[
            subset["ossl_file"].str.contains("soillab") &
            subset["ossl_file"].str.contains(".parquet")
        ]["public_url"].values

        spec_urls = subset[
            subset["ossl_file"].str.contains(f"_{s_type}_") &
            subset["ossl_file"].str.contains(".parquet")
        ]["public_url"].values

        if len(site_urls) == 0 or len(lab_urls) == 0 or len(spec_urls) == 0:
            ui.notification_show("Could not find Parquet URLs for site, lab, or spectra", type="error")
            return

        site_url = site_urls[0]
        lab_url = lab_urls[0]
        spec_url = spec_urls[0]

        with ui.Progress(min=1, max=4) as p:
            
            try:
                p.set(1, message="Downloading site data…")
                s_table = fetch_parquet_as_arrow(site_url)
                con.register("site_view", s_table)
                site_arrow.set(s_table)
                site_url_rv.set(site_url)

                p.set(2, message="Downloading lab data…")
                l_table = fetch_parquet_as_arrow(lab_url)
                con.register("lab_view", l_table)
                lab_arrow.set(l_table)
                lab_url_rv.set(lab_url)

                p.set(3, message="Downloading spectra data…")
                sp_table = fetch_parquet_as_arrow(spec_url)
                con.register("spec_view", sp_table)
                spec_arrow.set(sp_table)
                spec_url_rv.set(spec_url)

                # 1. Get raw column names
                s_cols = sorted(s_table.schema.names)
                l_cols = sorted(l_table.schema.names)
                sp_cols_raw = sp_table.schema.names
                
                scan_vals = []
                for c in sp_cols_raw:
                        if str(c).startswith("scan_"):
                            # Extract numeric part of the column name
                            num_str = re.sub(r'^scan_.*?\.', '', str(c))
                            num_str = re.sub(r'_(abs|ref|bc\.abs)$', '', num_str)
                            try: scan_vals.append(float(num_str))
                            except: pass
                
                if scan_vals:
                    ui.update_numeric("spec_min", value=min(scan_vals))
                    ui.update_numeric("spec_max", value=max(scan_vals))

                # 3. FILTER the spectra metadata columns
                # We remove anything starting with 'scan_' AND the join IDs
                clean_spec_metadata = [
                    c for c in sp_cols_raw
                    if not str(c).startswith("scan_")
                    and c not in {"id.layer_local_c", "id.scan_local_c", "id.layer_uuid_c", "id.layer_uuid"}
                ]

                site_cols_rv.set(s_cols)
                lab_cols_rv.set(l_cols)
                # spec_cols_rv.set(sorted(sp_cols_raw))
                spec_cols_rv.set(sorted(clean_spec_metadata))

                # # Reset previously derived reactive values
                # unique_site_levels.set({})
                # unique_spec_levels.set({})
                # joined_data.set(None)

                p.set(4, message="Done!")
                ui.notification_show("Metadata loaded successfully!", type="message")

            except Exception as e:
                ui.notification_show(f"Error loading metadata: {e}", type="error")

    # --------------------------------------------------
    # Column selector UIs
    # --------------------------------------------------
    @render.ui
    def site_column_selector():
        cols = site_cols_rv()
        if not cols:
            return ui.p("Load metadata first.", class_="text-muted")
        return ui.input_selectize("site_cols", "Site columns", choices=cols, multiple=True, width="100%")

    @render.ui
    def lab_column_selector():
        cols = lab_cols_rv()
        if not cols:
            return ui.p("Load dataset contents first.", class_="text-muted")

        # Select first column by default
        defaults = cols[:1]

        return ui.input_selectize("lab_cols", "Columns", choices=cols, selected=defaults, multiple=True, width="100%")

    @render.ui
    def spec_column_selector():
        cols = spec_cols_rv()
        if not cols:
            return ui.p("No metadata available.", class_="text-muted")
        return ui.input_selectize("spec_cols", "Columns", choices=cols, multiple=True, width="100%")

    # --------------------------------------------------
    # Site level filtering
    # --------------------------------------------------
    @reactive.Effect
    @reactive.event(input.site_cols)
    def _fetch_site_levels():
        selected = input.site_cols()
        if not selected:
            unique_site_levels.set({})
            return

        s_table = site_arrow()
        if s_table is None:
            return

        levels = {}
        # Only fetch levels for columns that need a dropdown
        cat_cols = [c for c in selected if c.endswith(("_c", "_txt", "_uint16", "_id", "_logical"))]
        
        if cat_cols:
            with ui.Progress(min=1, max=len(cat_cols)) as p:
                p.set(message="Extracting site levels…")
                for col in cat_cols:
                    try:
                        res = con.execute(
                            f'SELECT DISTINCT "{col}" FROM site_view WHERE "{col}" IS NOT NULL LIMIT 500'
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
            # Categorical logic
            if col.endswith(("_c", "_txt", "_uint16", "_id", "_logical")):
                if col in levels_dict and levels_dict[col]:
                    safe_id = "site_filter_" + re.sub(r'\W+', '_', col)
                    filters.append(
                        ui.input_selectize(safe_id, f"Filter: {col}", 
                                        choices=levels_dict[col], multiple=True, width="100%")
                    )
            
            # Numeric range logic
            elif col.endswith(("_cm", "_dd")):
                safe_id_min = "site_filter_min_" + re.sub(r'\W+', '_', col)
                safe_id_max = "site_filter_max_" + re.sub(r'\W+', '_', col)
                filters.append(
                    ui.div(
                        ui.markdown(f"Range: {col}"),
                        ui.layout_columns(
                            ui.input_numeric(safe_id_min, "Min", value=None),
                            ui.input_numeric(safe_id_max, "Max", value=None),
                            col_widths=(6, 6)
                        ),
                        class_="mb-3 border-bottom pb-2"
                    )
                )

        if not filters:
            return ui.p("Select other columns to filter.", class_="text-muted")

        return ui.div(*filters)

    # --------------------------------------------------
    # Spectra metadata level filtering
    # --------------------------------------------------
    @reactive.Effect
    @reactive.event(input.spec_cols)
    def _fetch_spec_levels():
        selected = input.spec_cols()
        if not selected:
            unique_spec_levels.set({})
            return

        sp_table = spec_arrow()
        if sp_table is None:
            return

        levels = {}
        with ui.Progress(min=1, max=10) as p:
            p.set(message="Extracting spectra levels…")
            for col in selected:
                try:
                    res = con.execute(
                        f'SELECT DISTINCT "{col}" FROM spec_view WHERE "{col}" IS NOT NULL LIMIT 100'
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
            return ui.p("Select spectra columns to see unique levels.", class_="text-muted")

        filters = []
        for col in selected:
            if col in levels_dict and levels_dict[col]:
                safe_id = "spec_filter_" + re.sub(r'\W+', '_', col)
                filters.append(
                    ui.input_selectize(safe_id, f"Filter: {col}", choices=levels_dict[col], multiple=True, width="100%")
                )

        if not filters:
            return ui.p("No discrete levels found.", class_="text-muted")

        return ui.div(*filters)

    def _build_site_where(site_selected):
        site_filters = []
        for col in site_selected:
            safe_suffix = re.sub(r'\W+', '_', col)
            
            # Logic for categorical columns
            if col.endswith(("_c", "_txt", "_uint16", "_id", "_logical")):
                safe_id = "site_filter_" + safe_suffix
                try:
                    selected_levels = input[safe_id]()
                    if selected_levels:
                        formatted_levels = ", ".join([f"'{x}'" for x in selected_levels])
                        site_filters.append(f'"{col}"::VARCHAR IN ({formatted_levels})')
                except:
                    pass
            
            # Logic for numeric range columns
            elif col.endswith(("_cm", "_dd")):
                try:
                    l_min = input["site_filter_min_" + safe_suffix]()
                    l_max = input["site_filter_max_" + safe_suffix]()
                    
                    # Validate that at least one bound is set (not default)
                    has_min = l_min is not None and l_min != float("-inf")
                    has_max = l_max is not None and l_max != float("inf")
                    
                    if has_min:
                        site_filters.append(f'"{col}"::DOUBLE >= {l_min}')
                    if has_max:
                        site_filters.append(f'"{col}"::DOUBLE <= {l_max}')
                        
                except:
                    pass
                    
        return ("WHERE " + " AND ".join(site_filters)) if site_filters else ""

    def _build_lab_selects_and_where(lab_selected, id_col, transform):
        lab_selects = [f'"{id_col}"']
        lab_where_clauses = []
        for c in lab_selected:
            if c == id_col:
                continue
            lab_where_clauses.append(f'"{c}" IS NOT NULL')
            if transform == "log1p":
                lab_selects.append(f'LN(1 + "{c}"::DOUBLE) AS "{c}_{transform}"')
            elif transform == "sqrt":
                lab_selects.append(f'SQRT("{c}"::DOUBLE) AS "{c}_{transform}"')
            else:
                lab_selects.append(f'"{c}"')
        lab_where = ("WHERE " + " AND ".join(lab_where_clauses)) if lab_where_clauses else ""
        return lab_selects, lab_where

    def _build_spec_where(spec_selected):
        spec_filters = []
        for col in spec_selected:
            safe_id = "spec_filter_" + re.sub(r'\W+', '_', col)
            try:
                selected_levels = input[safe_id]()
                if selected_levels:
                    formatted_levels = ", ".join([f"'{x}'" for x in selected_levels])
                    spec_filters.append(f'"{col}"::VARCHAR IN ({formatted_levels})')
            except:
                pass
        return ("WHERE " + " AND ".join(spec_filters)) if spec_filters else ""

    # --------------------------------------------------
    # Soil-only preview (no spectra join)
    # --------------------------------------------------
    def _validate_filters():
        """
        Validate that all selected filter columns have values set.
        Returns (is_valid, error_message)
        """
        site_selected = list(input.site_cols() or [])
        
        # Check numeric site filters
        for col in site_selected:
            if col.endswith(("_cm", "_dd")):
                safe_suffix = re.sub(r'\W+', '_', col)
                try:
                    l_min = input["site_filter_min_" + safe_suffix]()
                    l_max = input["site_filter_max_" + safe_suffix]()
                    
                    has_min = l_min is not None and l_min != float("-inf")
                    has_max = l_max is not None and l_max != float("inf")
                    
                    if not has_min and not has_max:
                        return False, f"Please provide at least one value (min or max) for '{col}', or deselect it."
                except:
                    return False, f"Could not read filter values for '{col}'"
        
        # Check categorical site filters (if they have levels but nothing selected, warn)
        levels_dict = unique_site_levels()
        for col in site_selected:
            if col.endswith(("_c", "_txt", "_uint16", "_id", "_logical")):
                if col in levels_dict and levels_dict[col]:  # Has available levels
                    safe_id = "site_filter_" + re.sub(r'\W+', '_', col)
                    try:
                        selected_levels = input[safe_id]()
                        if not selected_levels:  # Nothing selected
                            # This is OK — it means keep all levels (no filter)
                            pass
                    except:
                        pass
        
        return True, ""

    @reactive.Effect
    @reactive.event(input.run_soil_only, input.run_soil_site)
    def _run_soil_only():
        
        lab_selected = list(input.lab_cols() or [])
        
        if not lab_selected:
            ui.notification_show("Mandatory: Please select at least one soil property.", type="error")
            return
        
        # Validate filters before processing
        is_valid, error_msg = _validate_filters()
        if not is_valid:
            ui.notification_show(error_msg, type="error", duration=10)
            return

        if site_arrow() is None or lab_arrow() is None:
            ui.notification_show("Ensure metadata is loaded.", type="warning")
            return

        id_col = input.id_col()
        site_selected = list(input.site_cols() or [])
        lab_selected = list(input.lab_cols() or [])
        transform = input.transform_type()

        with ui.Progress(min=1, max=3) as p:
            p.set(1, message="Processing soil data…", detail="Applying filters")
            try:
                site_where = _build_site_where(site_selected)
                site_query = f'SELECT "{id_col}" FROM site_view {site_where}'

                lab_selects, lab_where = _build_lab_selects_and_where(lab_selected, id_col, transform)
                lab_query = f'SELECT {", ".join(lab_selects)} FROM lab_view {lab_where}'

                final_query = f"""
                SELECT l.*
                FROM ({site_query}) s
                INNER JOIN ({lab_query}) l ON s."{id_col}" = l."{id_col}"
                """

                p.set(2, message="Collecting…", detail="Fetching results")
                result = con.execute(final_query).df()

                joined_data.set(result)
                p.set(3, message="Done!")
                ui.notification_show(f"Soil data processed! {result.shape[0]} rows.", type="message")

            except Exception as e:
                ui.notification_show(f"Error: {str(e)}", type="error")

    # --------------------------------------------------
    # Full join (soil + spectra)
    # --------------------------------------------------
    @reactive.Effect
    @reactive.event(input.run_join)
    def _run_join():
        try:
            # Validate filters first
            is_valid, error_msg = _validate_filters()
            if not is_valid:
                ui.notification_show(error_msg, type="error", duration=10)
                return
            
            if site_arrow() is None or lab_arrow() is None or spec_arrow() is None:
                ui.notification_show("Ensure metadata is loaded.", type="warning")
                return

            id_col = input.id_col()
            def safe_input(name):
                try:
                    val = input[name]()
                    return list(val) if val else []
                except:
                    return []

            site_selected = safe_input("site_cols")
            lab_selected = safe_input("lab_cols")
            spec_selected = safe_input("spec_cols")

            j_type = input.join_type()
            # include_outcome = input.include_outcome()
            transform = input.transform_type()

            with ui.Progress(min=1, max=4) as p:
                p.set(1, message="Building queries…")

                site_where = _build_site_where(site_selected)
                site_query = f'SELECT "{id_col}" FROM site_view {site_where}'

                lab_selects, lab_where = _build_lab_selects_and_where(lab_selected, id_col, transform)
                lab_query = f'SELECT {", ".join(lab_selects)} FROM lab_view {lab_where}'

                p.set(2, message="Filtering soil data…")
                final_lab_query = f"""
                SELECT l.*
                FROM ({site_query}) s
                INNER JOIN ({lab_query}) l ON s."{id_col}" = l."{id_col}"
                """
                lab_result = con.execute(final_lab_query).df()
                ui.notification_show(f"Lab rows: {len(lab_result)}", type="message")

                p.set(3, message="Fetching spectra…")
                sp_table = spec_arrow()
                sp_schema_names = sp_table.schema.names
                scan_cols = [c for c in sp_schema_names if str(c).startswith("scan_")]

                rename_dict = {}
                for c in scan_cols:
                    num_str = re.sub(r'^scan_.*?\.', '', str(c))
                    num_str = re.sub(r'_(abs|ref|bc\.abs)$', '', num_str)
                    rename_dict[str(c)] = num_str

                # Filter scan columns based on user interval selection
                interval = int(input.spec_interval())
                s_min = input.spec_min()
                s_max = input.spec_max()

                filtered_scan_cols = []
                for c in scan_cols:
                    val = float(rename_dict[c])
                    # Check if the column is within the selected range
                    if s_min <= val <= s_max:
                        # Check if the column matches the sampling interval
                        if (val % interval) < 0.001 or (interval - (val % interval)) < 0.001:
                            filtered_scan_cols.append(c)

                # Overwrite scan_cols with our trimmed/thinned list
                scan_cols = filtered_scan_cols

                # Update rename_dict to only include the thinned columns
                new_scan_cols = [rename_dict[c] for c in scan_cols]

                spec_selects = [f'"{id_col}"']
                for c in spec_selected:
                    if c != id_col:
                        spec_selects.append(f'"{c}"')
                for c in scan_cols:
                    spec_selects.append(f'"{c}" AS "{rename_dict[c]}"')

                spec_where = _build_spec_where(spec_selected)
                spec_query = f'SELECT {", ".join(spec_selects)} FROM spec_view {spec_where}'
                spec_result = con.execute(spec_query).df()
                ui.notification_show(f"Spec rows: {len(spec_result)}", type="message")

                p.set(4, message="Merging…")
                how_map = {"inner": "inner", "full": "outer", "left": "left"}
                result = lab_result.merge(spec_result, on=id_col, how=how_map[j_type], suffixes=("", "_spec_dup"))
                dup_cols = [c for c in result.columns if c.endswith("_spec_dup")]
                result = result.drop(columns=dup_cols)

                # # Reorder scan columns
                # new_scan_cols = list(rename_dict.values())
                try:
                    sorted_scan_cols = sorted(new_scan_cols, key=lambda x: float(x))
                except ValueError:
                    sorted_scan_cols = sorted(new_scan_cols)

                final_cols = result.columns.tolist()
                non_scan = [c for c in final_cols if c not in sorted_scan_cols]
                result = result[non_scan + sorted_scan_cols]

                if result.empty:
                    ui.notification_show("Join produced 0 rows — check ID column and filters.", type="warning")
                    return

                joined_data.set(result)
                ui.notification_show(f"Join complete! {result.shape[0]} rows, {result.shape[1]} cols.", type="message")

        except Exception as e:
            import traceback
            ui.notification_show(f"Join error: {traceback.format_exc()}", type="error", duration=20)

    # --------------------------------------------------
    # Preview table
    # --------------------------------------------------
    @render.data_frame
    def preview_table():
        df = joined_data()
        if df is None:
            return render.DataGrid(pd.DataFrame({"Status": ["Waiting for data join..."]}))

        preview_df = df.head(10).iloc[:, :10].copy()
        
        # Round numeric columns to 5 decimal places
        num_cols = preview_df.select_dtypes(include="number").columns
        preview_df[num_cols] = preview_df[num_cols].round(5)

        r_count, c_count = df.shape
        if c_count > 10:
            preview_df["... "] = "..."
        if r_count > 10:
            preview_df.loc[len(preview_df)] = ["..."] * preview_df.shape[1]

        return render.DataGrid(preview_df)
    
    @render.ui
    def export_column_selector():
        df = joined_data()
        if df is None:
            return None
        
        # Identify non-spectral columns (Metadata & Lab properties)
        # Spectral bands in your app are strings like "600", "602.5", etc.
        all_cols = df.columns.tolist()
        
        def is_spectral(col_name):
            try:
                float(col_name)
                return True
            except ValueError:
                return False

        metadata_cols = [c for c in all_cols if not is_spectral(c)]
        
        return ui.div(
            ui.input_selectize(
                "export_cols", 
                "Columns to export", 
                choices=metadata_cols, 
                selected=metadata_cols, 
                multiple=True, 
                width="100%"
            ),
            ui.p("Note: All spectral bands will be included automatically.", class_="text-muted small mb-3")
        )

    # --------------------------------------------------
    # Download CSV
    # --------------------------------------------------
    @render.download(
        filename=lambda: input.dl_filename() if input.dl_filename().endswith(".csv") else input.dl_filename() + ".csv"
    )
    def download_csv():
        df = joined_data()
        if df is None:
            yield "No data available. Please run a join process first."
            return

        # 1. Helper to identify spectral columns (numeric strings like "600")
        def is_spectral(col_name):
            try:
                float(col_name)
                return True
            except ValueError:
                return False

        # 2. Get all available columns
        all_cols = df.columns.tolist()
        
        # 3. Separate spectral columns from the rest
        spectral_cols = [c for c in all_cols if is_spectral(c)]
        
        # 4. Get the metadata/soil columns the user checked in the UI
        # (These are the columns that are NOT spectral)
        keep_metadata = list(input.export_cols() or [])
        
        # 5. Combine: User's chosen metadata + All spectral bands
        final_export_cols = keep_metadata + spectral_cols
        
        # 6. Filter the dataframe and export
        export_df = df[final_export_cols]
        yield export_df.to_csv(index=False)


app = App(app_ui, server)