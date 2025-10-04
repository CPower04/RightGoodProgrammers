from flask import Flask, render_template_string, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import tempfile
from pathlib import Path
import csv
import joblib
import numpy as np
app = Flask(__name__)
app.secret_key ="secret_key"


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
    /* Global styles */
    body {
      font-family: Arial, sans-serif;
      background-color: #f4f7f9;
      color: #333;
      margin: 0;
      padding: 0;
    }

    header {
      background-color: #4a90e2;
      color: white;
      padding: 2rem 1rem;
      text-align: center;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    header h1 {
      margin: 0;
      font-size: 2rem;
    }

    header p {
      margin-top: 0.5rem;
      font-size: 1rem;
    }

    main {
      max-width: 900px;
      margin: 2rem auto;
      padding: 2rem;
      background: linear-gradient(135deg, #ffffff, #e6f0fa);
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      border: 1px solid #dce6f1;
      transition: transform 0.2s;
    }

    main:hover {
      transform: translateY(-3px);
    }

    h2 {
      color: #4a90e2;
      margin-bottom: 1rem;
    }

    form {
      margin-bottom: 2rem;
    }

    input[type="file"] {
      margin-right: 1rem;
    }

    input[type="number"] {
      width: 6ch;
      padding: 0.25rem;
      border-radius: 4px;
      border: 1px solid #ccc;
    }

    input[type="submit"], button {
      background-color: #4a90e2;
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 4px;
      cursor: pointer;
      margin-left: 1rem;
      transition: background-color 0.2s;
    }

    input[type="submit"]:hover, button:hover {
      background-color: #357ab8;
    }

    ul {
      padding-left: 1.2rem;
      color: #d9534f;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    table th, table td {
      border: 1px solid #ddd;
      padding: 0.5rem;
      text-align: left;
    }

    table th {
      background-color: #f0f4f8;
    }

    table tr:nth-child(even) {
      background-color: #f9fbfd;
    }

    table tr:hover {
      background-color: #e0f0ff;
    }

    section p {
      margin: 0.5rem 0;
    }

    label {
      font-size: 0.95rem;
    }
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
        <form method="get" style="margin: .5rem 0;">
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
        {{ table|safe }}
      </section>
    {% endif %}
  </main>
</body>
</html>

'''


KEPLER_MODEL_PATH = '../models/keplerkepler_rf_best.joblib'
TOI_MODEL_PATH = '../models/toiscaler.joblib'
K2_MODEL_PATH = '../models/k2panda_model.pkl'

kepler_model = joblib.load(KEPLER_MODEL_PATH)
toi_model = joblib.load(TOI_MODEL_PATH)
k2_model = joblib.load(K2_MODEL_PATH)

KEPLER_MODEL_Features = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
       'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_time0bk',
       'koi_time0bk_err1', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
       'koi_duration', 'koi_duration_err1', 'koi_depth', 'koi_depth_err1',
       'koi_prad', 'koi_prad_err1', 'koi_teq', 'koi_insol', 'koi_insol_err1',
       'koi_model_snr', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
       'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad',
       'koi_srad_err1', 'koi_srad_err2', 'koi_kepmag']

TOI_MODEL_Features = ['rastr', 'decstr', 'st_pmra', 'st_pmraerr1', 'st_pmralim', 'st_pmdec',
       'st_pmdeclim', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmidlim',
       'pl_orbper', 'pl_orbpererr1', 'pl_orbperlim', 'pl_trandurh',
       'pl_trandurherr1', 'pl_trandurhlim', 'pl_trandep', 'pl_trandeperr1',
       'pl_trandeplim', 'pl_rade', 'pl_radeerr1', 'pl_radelim', 'pl_insol',
       'pl_eqt', 'st_tmag', 'st_tmagerr1', 'st_tmaglim', 'st_dist',
       'st_distlim', 'st_teff', 'st_tefferr1', 'st_tefflim', 'st_logg',
       'st_loggerr1', 'st_logglim', 'st_raderr1', 'st_radlim']

K2_MODEL_Features = ['sy_snum', 'sy_pnum', 'pl_controv_flag', 'pl_orbper', 'st_rad', 'st_raderr1', 
                     'st_radlim', 'ra', 'sy_dist', 'sy_vmag', 'sy_vmagerr1', 'sy_kmagerr1', 'sy_gaiamagerr1']
FRAMES = {}


def select_model(df):
    kepler_score = len(set(KEPLER_MODEL_Features) & set(df.columns))
    k2_score = len(set(K2_MODEL_Features) & set(df.columns))
    toi_score = len(set(TOI_MODEL_Features) & set(df.columns))
    scores = {'kepler': kepler_score, 'k2': k2_score, 'toi': toi_score}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def predict_status(df):
    X = df[KEPLER_MODEL_Features].copy()
    return X


def predict_k2_status(df):
    X = df[K2_MODEL_Features].copy()
    return X


def predict_toi_status(df):
    X = df[TOI_MODEL_Features].copy()
    return X

def predict_kepler(df):
    x = predict_status(df)
    predict = kepler_model.predict(x)
    return predict

def predict_k2(df):
    x = predict_k2_status(df)
    predict = k2_model.predict(x)
    return predict

def predict_toi(df):
    x = predict_toi_status(df)
    predict = toi_model.predict(x)
    return predict
def sniff_dialect(sample_bytes: bytes):
    # Use csv.Sniffer to detect delimiter & quotechar.
    # Fall back to comma if sniffer fails.
    sample = sample_bytes.decode('utf-8', errors='replace')
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
        return dialect.delimiter, dialect.quotechar or '"'
    except Exception:
        return ',', '"'  # safe default

def try_read_csv(path: Path, skip_bad: bool):
    """
    Attempt to read a CSV using several encodings and a sniffed delimiter.
    Avoids low_memory (not supported with Python engine).
    """
    encodings = [
        ('utf-8', 'strict'),
        ('utf-8-sig', 'strict'),
        ('cp1252', 'replace'),
        ('latin1', 'replace'),
    ]

    # Read a small sample to guess delimiter
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
            model_type = select_model(df)
            if model_type == 'kepler':
                used_cols = [col for col in KEPLER_MODEL_Features if col in df.columns]
                df_model = df[used_cols].copy()
                try:
                    preds = kepler_model.predict(df_model)
                    df_model['Predicted Status'] = preds
                except Exception as e:
                    flash(f'Prediction error: {e}')
            elif model_type == 'k2':
                used_cols = [col for col in K2_MODEL_Features if col in df.columns]
                df_model = df[used_cols].copy()
                try:
                    preds = k2_model.predict(df_model)
                    df_model['Predicted Status'] = preds
                except Exception as e:
                    flash(f'Prediction error: {e}')
            elif model_type == 'toi':
                used_cols = [col for col in TOI_MODEL_Features if col in df.columns]
                df_model = df[used_cols].copy()
                try:
                    preds = toi_model.predict(df_model)
                    df_model['Predicted Status'] = preds
                except Exception as e:
                    flash(f'Prediction error: {e}')
            else:
                flash('Could not match CSV to any known model.')
                df_model = df.copy()

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
            # clean up temp file
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

   
    return render_template_string(HTML_TEMPLATE, meta=None, table=None, skip_bad=True)

if __name__ == '__main__':
    app.run(debug=True)
