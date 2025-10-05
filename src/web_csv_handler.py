import csv
import tempfile
from pathlib import Path
import subprocess
import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret_key"

# Limit uploads to 64MB
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
PAGE_SIZE_DEFAULT = 100

HTML_TEMPLATE = 'index.html'

# --------------------------------------------------------------------
# K2 paths
# --------------------------------------------------------------------
K2_LOOKUP_PKL = 'k2panda_model.pkl'      # preferred lookup
K2_FINAL_CSV = '../data/k2panda_fin.csv'  # fallback if PKL is not usable

# Features used for matching
K2_MODEL_Features = [
    'sy_snum', 'sy_pnum', 'pl_controv_flag', 'pl_orbper', 'st_rad', 'st_raderr1',
    'st_radlim', 'ra', 'sy_dist', 'sy_vmag', 'sy_vmagerr1', 'sy_kmagerr1', 'sy_gaiamagerr1'
]

# --------------------------------------------------------------------
# Load K2 lookup table (try PKL first, fallback to CSV)
# --------------------------------------------------------------------
def _load_k2_lookup():
    p = Path(K2_LOOKUP_PKL)
    if p.exists():
        try:
            obj = joblib.load(p)
            if isinstance(obj, pd.DataFrame):
                app.logger.info(f"Loaded K2 lookup from {K2_LOOKUP_PKL}")
                return obj
            if isinstance(obj, dict):
                for key in ('k2_final', 'df', 'data', 'lookup'):
                    if key in obj and isinstance(obj[key], pd.DataFrame):
                        app.logger.info(f"Loaded DataFrame '{key}' from {K2_LOOKUP_PKL}")
                        return obj[key]
                for v in obj.values():
                    if isinstance(v, pd.DataFrame):
                        app.logger.info(f"Loaded first DataFrame from dict in {K2_LOOKUP_PKL}")
                        return v
            app.logger.warning(f"{K2_LOOKUP_PKL} did not contain a DataFrame — falling back to CSV.")
        except Exception as e:
            app.logger.warning(f"Failed to load {K2_LOOKUP_PKL}: {e} — falling back to CSV.")
    app.logger.info(f"Loading fallback lookup CSV: {K2_FINAL_CSV}")
    return pd.read_csv(K2_FINAL_CSV)

k2_final = _load_k2_lookup()

# In-memory frame registry
FRAMES = {}

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def make_matrix_k2(df: pd.DataFrame, features):
    valid_cols = [col for col in features if col in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"K2 CSV is missing required columns: {missing[:8]}{'...' if len(missing)>8 else ''}"
        )
    filtered_df = df[valid_cols].copy()
    X = filtered_df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return filtered_df, X.fillna(0.0)

def sniff_dialect(sample_bytes: bytes):
    sample = sample_bytes.decode('utf-8', errors='replace')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter, dialect.quotechar or '"'
    except Exception:
        return ',', '"'

def try_read_csv(path: Path, skip_bad: bool):
    encodings = [
        ('utf-8', 'strict'),
        ('utf-8-sig', 'strict'),
        ('cp1252', 'replace'),
        ('latin1', 'replace'),
    ]
    with open(path, 'rb') as f:
        sample = f.read(200_000)
    delimiter, quotechar = sniff_dialect(sample)
    last_err = None
    for enc, enc_err in encodings:
        try:
            bad = 'skip' if skip_bad else 'error'
            df = pd.read_csv(
                path,
                encoding=enc,
                encoding_errors=enc_err,
                comment='#',
                sep=delimiter,
                quotechar=quotechar,
                engine='python',
                dtype='string',
                na_filter=False,
                on_bad_lines=bad,
            )
            if df is None or df.shape[1] == 0:
                raise ValueError("Parsed zero columns.")
            return df, {'encoding': enc, 'delimiter': delimiter, 'quotechar': quotechar}
        except Exception as e:
            last_err = e
            continue
    raise Exception(f"Failed to load CSV after multiple tries. Last error: {last_err}")

def mem_megabytes(df: pd.DataFrame) -> float:
    try:
        return round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    except Exception:
        return round(df.memory_usage().sum() / (1024**2), 2)

# --------------------------------------------------------------------
# Streamlit trigger
# --------------------------------------------------------------------
@app.route('/run_streamlit', methods=['POST'])
def run_streamlit():
    try:
        base_dir = Path(__file__).parent.resolve()
        base_dir = base_dir.parent  # Move up to src
        streamlit_dir = base_dir / "astrophys"
        streamlit_script = "streamlitmegno.py"
        cmd = f"streamlit run .\\{streamlit_script}"
        subprocess.Popen(
            ["powershell", "-Command", cmd],
            shell=True,
            cwd=streamlit_dir
        )
    except Exception as e:
        flash(f"Error launching Streamlit: {e}")
    return redirect(url_for('upload_or_view'))

# --------------------------------------------------------------------
# Main route
# --------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_or_view():
    token = request.args.get('token')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', PAGE_SIZE_DEFAULT))

    # Paging through an existing frame
    if request.method == 'GET' and token and token in FRAMES:
        entry = FRAMES[token]
        df = entry['df']
        meta = entry['meta']
        start = max(0, (page - 1) * page_size)
        end = min(len(df), start + page_size)
        page_df = df.iloc[start:end]
        table_html = page_df.to_html(classes='table table-striped', index=False, border=0)
        return render_template(
            HTML_TEMPLATE,
            meta=meta,
            table=table_html,
            token=token,
            page=page,
            page_size=page_size,
            start=start,
            end=end,
            skip_bad=entry.get('skip_bad', False)
        )

    # Handle new upload
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('upload_or_view'))
        f = request.files['file']
        if not f or f.filename == '':
            flash('No selected file')
            return redirect(url_for('upload_or_view'))

        skip_bad = request.form.get('skip_bad') == 'on'
        filename = secure_filename(f.filename)
        suffix = '.csv' if '.' not in filename else ''
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            f.save(temp_path)

        try:
            df, parse_info = try_read_csv(temp_path, skip_bad=skip_bad)
            try:
                filtered_df, x = make_matrix_k2(df, K2_MODEL_Features)
                x = pd.DataFrame(x, columns=K2_MODEL_Features)

                # Merge with lookup (from PKL or CSV)
                merged = pd.merge(x, k2_final, on=K2_MODEL_Features, how='left')

                # Handle NA safely before checking
                disp_cols = [
                    'disposition_CONFIRMED',
                    'disposition_CANDIDATE',
                    'disposition_FALSE POSITIVE'
                ]
                for c in disp_cols:
                    if c not in merged.columns:
                        merged[c] = 0
                merged[disp_cols] = merged[disp_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

                # Determine disposition (no NA ambiguity now)
                def get_disposition(row):
                    if row['disposition_CONFIRMED'] == 1:
                        return 'CONFIRMED'
                    elif row['disposition_CANDIDATE'] == 1:
                        return 'CANDIDATE'
                    elif row['disposition_FALSE POSITIVE'] == 1:
                        return 'FALSE POSITIVE'
                    else:
                        return 'UNKNOWN'

                filtered_df['Disposition'] = merged.apply(get_disposition, axis=1)
                df_model = filtered_df
            except Exception as e:
                flash(f'Disposition lookup error: {e}')
                df_model = df[df.columns.intersection(K2_MODEL_Features)].copy()

            # Cache for paging
            token = next(tempfile._get_candidate_names())
            meta = {
                'filename': filename,
                'nrows': int(df_model.shape[0]),
                'ncols': int(df_model.shape[1]),
                'delimiter': parse_info['delimiter'],
                'quotechar': parse_info['quotechar'],
                'encoding': parse_info['encoding'],
                'mem_mb': mem_megabytes(df_model),
            }
            FRAMES[token] = {'df': df_model, 'meta': meta, 'skip_bad': skip_bad}

            # Show first page
            start, end = 0, min(len(df_model), PAGE_SIZE_DEFAULT)
            table_html = df_model.iloc[start:end].to_html(classes='table table-striped', index=False, border=0)
            return render_template(
                HTML_TEMPLATE,
                meta=meta,
                table=table_html,
                token=token,
                page=1,
                page_size=PAGE_SIZE_DEFAULT,
                start=start,
                end=end,
                skip_bad=skip_bad
            )
        except Exception as e:
            flash(str(e))
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Default GET
    return render_template(HTML_TEMPLATE, meta=None, table=None, skip_bad=True)

# --------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
