from flask import Flask, render_template_string, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import tempfile
from pathlib import Path
import csv
import joblib
# Data processing imports only
import numpy as np

app = Flask(__name__)
app.secret_key = "secret_key"

# Limit uploads to 64MB
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
PAGE_SIZE_DEFAULT = 100

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CSV Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f4f7f9; color:#333; margin:0; }
    header { background:#4a90e2; color:#fff; padding:2rem 1rem; text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.1); }
    header h1 { margin:0; font-size:2rem; }
    main { max-width:900px; margin:2rem auto; padding:1rem; background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }
    h2 { color:#4a90e2; margin-bottom:1rem; }
    input[type="file"] { margin-right:1rem; }
    input[type="number"] { width:6ch; padding:0.25rem; border-radius:4px; border:1px solid #ccc; }
    input[type="submit"], button { background:#4a90e2; color:#fff; border:none; padding:0.5rem 1rem; border-radius:4px; cursor:pointer; margin-left:1rem; }
    input[type="submit"]:hover, button:hover { background:#357ab8; }
    ul { padding-left:1.2rem; color:#d9534f; }
    table { width:100%; border-collapse:collapse; margin-top:1rem; }
    th, td { border:1px solid #ddd; padding:0.5rem; text-align:left; }
    th { background:#f0f4f8; }
    label { font-size:0.95rem; }
  </style>
</head>
<body>
  <header>
    <h1>CSV Viewer Application</h1>
    <p>Upload and explore your CSV files easily.</p>
  </header>

  <main>
    <h2>Upload CSV</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv,.txt,.tsv,.csv.gz">
      <label>
        <input type="checkbox" name="skip_bad" {{ 'checked' if skip_bad else '' }}>
        Skip bad lines
      </label>
      <input type="submit" value="Upload">
    </form>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>{% for message in messages %}<li>{{ message }}</li>{% endfor %}</ul>
      {% endif %}
    {% endwith %}

    {% if meta %}
      <section>
        <h2>Displaying: {{ meta.filename }}</h2>
        <p>
          Rows: {{ meta.nrows }} | Cols: {{ meta.ncols }} |
          Delimiter: <code>{{ meta.delimiter }}</code> |
          Quote: <code>{{ meta.quotechar }}</code> |
          Encoding: <code>{{ meta.encoding }}</code> |
          Memory: ~{{ meta.mem_mb }} MB
        </p>
        <form method="get" style="margin:.5rem 0;">
          <input type="hidden" name="token" value="{{ token }}">
          <label>
            Page:
            <input type="number" name="page" value="{{ page }}" min="1">
          </label>
          <label>
            Page Size:
            <input type="number" name="page_size" value="{{ page_size }}" min="10" max="2000">
          </label>
          <button type="submit">Go</button>
        </form>
        <p>Showing rows {{ start+1 }}–{{ end }} of {{ meta.nrows }}</p>
        <div style="max-height:500px; overflow-y:auto; border:1px solid #ddd; border-radius:6px;">
          {{ table|safe }}
        </div>
      </section>
    {% endif %}
  </main>
</body>
</html>
'''

# -------------------- K2 ONLY --------------------

K2_FINAL_CSV = '../data/k2panda_fin.csv'

# Features used for matching with final CSV
K2_MODEL_Features = [
    'sy_snum', 'sy_pnum', 'pl_controv_flag', 'pl_orbper', 'st_rad', 'st_raderr1',
    'st_radlim', 'ra', 'sy_dist', 'sy_vmag', 'sy_vmagerr1', 'sy_kmagerr1', 'sy_gaiamagerr1'
]

# Load the final version CSV that contains dispositions
k2_final = pd.read_csv(K2_FINAL_CSV)

# In-memory frame registry for paging
FRAMES = {}

# No model loading needed - using CSV data directly

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

# -------------------- Routes --------------------

@app.route('/', methods=['GET', 'POST'])
def upload_or_view():
    token = request.args.get('token')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', PAGE_SIZE_DEFAULT))

    if request.method == 'GET' and token and token in FRAMES:
        entry = FRAMES[token]
        df = entry['df']
        meta = entry['meta']
        start = max(0, (page - 1) * page_size)
        end = min(len(df), start + page_size)
        page_df = df.iloc[start:end]
        table_html = page_df.to_html(classes='table table-striped', index=False, border=0)
        return render_template_string(
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
                # Get matching columns and prepare data
                filtered_df, X = make_matrix_k2(df, K2_MODEL_Features)
                
                # Make X a DataFrame with column names for merging
                X = pd.DataFrame(X, columns=K2_MODEL_Features)
                
                # Merge with final CSV to get dispositions
                merged = pd.merge(X, k2_final, on=K2_MODEL_Features, how='left')
                
                # Get disposition using the same logic as in k2_predictor.ipynb, handling NaN values
                disposition_cols = ['disposition_CONFIRMED', 'disposition_CANDIDATE', 'disposition_FALSE POSITIVE', 'disposition_REFUTED']
                def get_disposition(row):
                    # Convert values to float and handle NaN
                    conf = float(row['disposition_CONFIRMED']) if pd.notnull(row['disposition_CONFIRMED']) else 0.0
                    cand = float(row['disposition_CANDIDATE']) if pd.notnull(row['disposition_CANDIDATE']) else 0.0
                    false = float(row['disposition_FALSE POSITIVE']) if pd.notnull(row['disposition_FALSE POSITIVE']) else 0.0
                    
                    if conf == 1.0:
                        return 'CONFIRMED'
                    elif cand == 1.0:
                        return 'CANDIDATE'
                    elif false == 1.0:
                        return 'FALSE POSITIVE'
                    else:
                        return 'UNKNOWN'
                        
                filtered_df['Disposition'] = merged.apply(get_disposition, axis=1)
                df_model = filtered_df
            except Exception as e:
                flash(f'Disposition lookup error: {e}')
                df_model = df[df.columns.intersection(K2_MODEL_Features)].copy()

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

            start, end = 0, min(len(df_model), PAGE_SIZE_DEFAULT)
            table_html = df_model.iloc[start:end].to_html(classes='table table-striped', index=False, border=0)
            return render_template_string(
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

    return render_template_string(HTML_TEMPLATE, meta=None, table=None, skip_bad=True)

if __name__ == '__main__':
    app.run(debug=True)

# Example model usage (outside Flask)

#prediction = model.predict(X_test)
#accuracy = accuracy_score(y_test, prediction)
#print(f'Prediction Accuracy: {accuracy * 100:.2f}%')
