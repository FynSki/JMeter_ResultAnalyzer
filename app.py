from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

loaded_files = {}

JMETER_COLUMNS = {
    'timeStamp': ['timeStamp', 'timestamp', 'Timestamp'],
    'elapsed': ['elapsed', 'Elapsed', 'response_time'],
    'label': ['label', 'Label', 'sampler_label'],
    'responseCode': ['responseCode', 'response_code', 'ResponseCode'],
    'success': ['success', 'Success'],
    'bytes': ['bytes', 'Bytes'],
    'threadName': ['threadName', 'thread_name'],
    'grpThreads': ['grpThreads'],
    'allThreads': ['allThreads'],
}

def normalize_columns(df):
    col_map = {}
    for canonical, variants in JMETER_COLUMNS.items():
        for v in variants:
            if v in df.columns:
                col_map[v] = canonical
                break
    return df.rename(columns=col_map)

def load_csv(filepath):
    df = None
    for sep in [',', ';', '\t']:
        try:
            tmp = pd.read_csv(filepath, low_memory=False, sep=sep, encoding='utf-8')
            if len(tmp.columns) > 1:
                df = tmp
                break
        except Exception:
            pass
    if df is None:
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')
        except Exception as e:
            raise ValueError(f"Nie mozna odczytac pliku CSV: {e}")

    df.columns = [c.strip() for c in df.columns]
    df = normalize_columns(df)

    if 'timeStamp' in df.columns:
        ts = pd.to_numeric(df['timeStamp'], errors='coerce')
        df = df[ts.notna()].copy()
        ts = pd.to_numeric(df['timeStamp'], errors='coerce')
        try:
            if ts.median() > 1e12:
                df['datetime'] = pd.to_datetime(ts, unit='ms')
            else:
                df['datetime'] = pd.to_datetime(ts, unit='s')
            start = df['datetime'].min()
            df['rel_minute'] = ((df['datetime'] - start).dt.total_seconds() // 60).astype(int)
            df['minute'] = df['datetime'].dt.floor('min')
        except Exception:
            pass

    if 'success' in df.columns:
        df['success'] = df['success'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])

    if 'elapsed' in df.columns:
        df['elapsed'] = pd.to_numeric(df['elapsed'], errors='coerce')
        df = df[df['elapsed'].notna()].copy()

    return df

def classify_label(label):
    return 'request' if str(label).startswith('/') else 'scenario'

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/jmx')
def jmx_page():
    return send_from_directory('static', 'jmx.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No filename'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        df = load_csv(filepath)
        loaded_files[filename] = df
        labels_raw = sorted(df['label'].unique().tolist()) if 'label' in df.columns else []
        labels = [{'name': l, 'type': classify_label(l)} for l in labels_raw]
        time_range = None
        if 'datetime' in df.columns:
            time_range = {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat(),
                'duration_s': int((df['datetime'].max() - df['datetime'].min()).total_seconds())
            }
        return jsonify({
            'filename': filename,
            'rows': len(df),
            'labels': labels,
            'columns': list(df.columns),
            'time_range': time_range
        })
    except Exception as e:
        import traceback
        print("BLAD UPLOADU:", traceback.format_exc())
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    result = []
    for fname, df in loaded_files.items():
        labels_raw = sorted(df['label'].unique().tolist()) if 'label' in df.columns else []
        labels = [{'name': l, 'type': classify_label(l)} for l in labels_raw]
        result.append({'filename': fname, 'rows': len(df), 'labels': labels})
    return jsonify(result)

@app.route('/api/single_run', methods=['POST'])
def single_run():
    data = request.json
    filename = data.get('filename')
    if filename not in loaded_files:
        return jsonify({'error': 'File not loaded'}), 404
    df = loaded_files[filename]
    if 'rel_minute' not in df.columns:
        return jsonify({'error': 'No timestamp column found'}), 400

    result = []
    for rel_min, group in df.groupby('rel_minute'):
        result.append({
            'minute': int(rel_min),
            'request_count': len(group),
            'median_response_ms': round(float(group['elapsed'].median()), 2) if 'elapsed' in group.columns else None,
            'avg_response_ms': round(float(group['elapsed'].mean()), 2) if 'elapsed' in group.columns else None,
            'p95_response_ms': round(float(group['elapsed'].quantile(0.95)), 2) if 'elapsed' in group.columns else None,
            'error_rate': round(float((~group['success']).mean() * 100), 2) if 'success' in group.columns else None,
        })
    result.sort(key=lambda x: x['minute'])

    overall = {
        'total_requests': len(df),
        'median_response_ms': round(float(df['elapsed'].median()), 2) if 'elapsed' in df.columns else None,
        'avg_response_ms': round(float(df['elapsed'].mean()), 2) if 'elapsed' in df.columns else None,
        'p95_response_ms': round(float(df['elapsed'].quantile(0.95)), 2) if 'elapsed' in df.columns else None,
        'error_rate': round(float((~df['success']).mean() * 100), 2) if 'success' in df.columns else None,
        'throughput_rps': round(len(df) / max(1, (df['datetime'].max() - df['datetime'].min()).total_seconds()), 2) if 'datetime' in df.columns else None,
        'labels': [{'name': l, 'type': classify_label(l)} for l in sorted(df['label'].unique().tolist())] if 'label' in df.columns else []
    }

    return jsonify({'per_minute': result, 'overall': overall})

@app.route('/api/compare_request', methods=['POST'])
def compare_request():
    data = request.json
    baseline_file = data.get('baseline')
    compare_file = data.get('compare')
    label = data.get('label')

    if baseline_file not in loaded_files:
        return jsonify({'error': f'Baseline file not loaded: {baseline_file}'}), 404
    if compare_file not in loaded_files:
        return jsonify({'error': f'Compare file not loaded: {compare_file}'}), 404

    def stats_for(df, fname):
        sub = df[df['label'] == label].copy() if (label and 'label' in df.columns) else df.copy()
        if len(sub) == 0:
            return {'filename': fname, 'count': 0}
        s = {
            'filename': fname,
            'count': len(sub),
            'avg_response_ms': round(float(sub['elapsed'].mean()), 2) if 'elapsed' in sub.columns else None,
            'median_response_ms': round(float(sub['elapsed'].median()), 2) if 'elapsed' in sub.columns else None,
            'p90_response_ms': round(float(sub['elapsed'].quantile(0.90)), 2) if 'elapsed' in sub.columns else None,
            'p95_response_ms': round(float(sub['elapsed'].quantile(0.95)), 2) if 'elapsed' in sub.columns else None,
            'p99_response_ms': round(float(sub['elapsed'].quantile(0.99)), 2) if 'elapsed' in sub.columns else None,
            'min_response_ms': round(float(sub['elapsed'].min()), 2) if 'elapsed' in sub.columns else None,
            'max_response_ms': round(float(sub['elapsed'].max()), 2) if 'elapsed' in sub.columns else None,
            'error_rate': round(float((~sub['success']).mean() * 100), 2) if 'success' in sub.columns else None,
        }
        if 'rel_minute' in sub.columns:
            pm = sub.groupby('rel_minute').agg(
                count=('elapsed', 'count'),
                avg=('elapsed', 'mean'),
                median=('elapsed', 'median')
            ).reset_index()
            s['per_minute'] = [
                {'minute': int(r['rel_minute']), 'count': int(r['count']),
                 'avg_response_ms': round(float(r['avg']), 2),
                 'median_response_ms': round(float(r['median']), 2)}
                for _, r in pm.iterrows()
            ]
        return s

    baseline_stats = stats_for(loaded_files[baseline_file], baseline_file)
    compare_stats = stats_for(loaded_files[compare_file], compare_file)

    diff = {}
    if baseline_stats.get('avg_response_ms') and compare_stats.get('avg_response_ms'):
        b, c = baseline_stats['avg_response_ms'], compare_stats['avg_response_ms']
        diff['avg_response_ms_change_pct'] = round((c - b) / b * 100, 2)
    if baseline_stats.get('median_response_ms') and compare_stats.get('median_response_ms'):
        b, c = baseline_stats['median_response_ms'], compare_stats['median_response_ms']
        diff['median_response_ms_change_pct'] = round((c - b) / b * 100, 2)

    all_labels = set()
    for fname in [baseline_file, compare_file]:
        dff = loaded_files[fname]
        if 'label' in dff.columns:
            all_labels.update(dff['label'].unique().tolist())

    return jsonify({
        'label': label,
        'baseline': baseline_stats,
        'compare': compare_stats,
        'diff': diff,
        'available_labels': [{'name': l, 'type': classify_label(l)} for l in sorted(all_labels)]
    })

@app.route('/api/multi_compare', methods=['POST'])
def multi_compare():
    data = request.json
    baseline_file = data.get('baseline')
    compare_files = data.get('compare_files', [])
    label = data.get('label')

    all_files = [baseline_file] + compare_files
    for f in all_files:
        if f not in loaded_files:
            return jsonify({'error': f'File not loaded: {f}'}), 404

    def get_per_minute(fname, lbl):
        df = loaded_files[fname].copy()
        if lbl and 'label' in df.columns:
            df = df[df['label'] == lbl]
        if 'rel_minute' not in df.columns or len(df) == 0:
            return []
        has_success = 'success' in df.columns
        agg_dict = {
            'count': ('elapsed', 'count'),
            'avg': ('elapsed', 'mean'),
            'median': ('elapsed', 'median'),
            'p95': ('elapsed', lambda x: x.quantile(0.95)),
        }
        if has_success:
            agg_dict['errors'] = ('success', lambda x: (~x).mean() * 100)
        pm = df.groupby('rel_minute').agg(**agg_dict).reset_index()
        rows = []
        for _, r in pm.iterrows():
            rows.append({
                'minute': int(r['rel_minute']),
                'avg_response_ms': round(float(r['avg']), 2),
                'median_response_ms': round(float(r['median']), 2),
                'p95_response_ms': round(float(r['p95']), 2),
                'request_count': int(r['count']),
                'error_rate': round(float(r['errors']), 2) if has_success else 0,
            })
        return rows

    def overall_stats(fname, lbl):
        df = loaded_files[fname].copy()
        if lbl and 'label' in df.columns:
            df = df[df['label'] == lbl]
        if len(df) == 0:
            return {'count': 0}
        return {
            'count': len(df),
            'avg_response_ms': round(float(df['elapsed'].mean()), 2) if 'elapsed' in df.columns else None,
            'median_response_ms': round(float(df['elapsed'].median()), 2) if 'elapsed' in df.columns else None,
            'p95_response_ms': round(float(df['elapsed'].quantile(0.95)), 2) if 'elapsed' in df.columns else None,
            'error_rate': round(float((~df['success']).mean() * 100), 2) if 'success' in df.columns else None,
        }

    all_labels = set()
    for fname in all_files:
        dff = loaded_files[fname]
        if 'label' in dff.columns:
            all_labels.update(dff['label'].unique().tolist())

    return jsonify({
        'baseline': {
            'filename': baseline_file,
            'per_minute': get_per_minute(baseline_file, label),
            'overall': overall_stats(baseline_file, label)
        },
        'compares': [
            {
                'filename': f,
                'per_minute': get_per_minute(f, label),
                'overall': overall_stats(f, label)
            }
            for f in compare_files
        ],
        'available_labels': [{'name': l, 'type': classify_label(l)} for l in sorted(all_labels)]
    })


@app.route('/api/summary_table', methods=['POST'])
def summary_table():
    data = request.json
    baseline_file = data.get('baseline')
    compare_file  = data.get('compare')

    if baseline_file not in loaded_files:
        return jsonify({'error': f'File not loaded: {baseline_file}'}), 404
    if compare_file and compare_file not in loaded_files:
        return jsonify({'error': f'File not loaded: {compare_file}'}), 404

    def per_label_stats(df):
        if 'label' not in df.columns:
            return {}
        out = {}
        has_success = 'success' in df.columns
        has_elapsed = 'elapsed' in df.columns
        for lbl, grp in df.groupby('label'):
            out[lbl] = {
                'count':  int(len(grp)),
                'avg':    round(float(grp['elapsed'].mean()), 1)        if has_elapsed else None,
                'median': round(float(grp['elapsed'].median()), 1)      if has_elapsed else None,
                'p90':    round(float(grp['elapsed'].quantile(.90)), 1) if has_elapsed else None,
                'p95':    round(float(grp['elapsed'].quantile(.95)), 1) if has_elapsed else None,
                'p99':    round(float(grp['elapsed'].quantile(.99)), 1) if has_elapsed else None,
                'min':    round(float(grp['elapsed'].min()), 1)         if has_elapsed else None,
                'max':    round(float(grp['elapsed'].max()), 1)         if has_elapsed else None,
                'errors': round(float((~grp['success']).mean()*100), 2) if has_success else None,
            }
        return out

    def safe_diff(a, b):
        if a is None or b is None: return None
        return round(b - a, 1)

    def safe_pct(a, b):
        if a is None or b is None or a == 0: return None
        return round((b - a) / abs(a) * 100, 1)

    base_stats = per_label_stats(loaded_files[baseline_file])

    if compare_file:
        cmp_stats = per_label_stats(loaded_files[compare_file])
        all_labels = sorted(set(list(base_stats.keys()) + list(cmp_stats.keys())))
        rows = []
        for lbl in all_labels:
            b = base_stats.get(lbl)
            c = cmp_stats.get(lbl)
            bv = lambda k: b[k] if b else None
            cv = lambda k: c[k] if c else None
            rows.append({
                'label': lbl, 'type': classify_label(lbl),
                'b_count': bv('count'), 'c_count': cv('count'),
                'd_count': safe_diff(bv('count'), cv('count')),
                'd_count_pct': safe_pct(bv('count'), cv('count')),
                'b_avg': bv('avg'), 'c_avg': cv('avg'),
                'd_avg': safe_diff(bv('avg'), cv('avg')),
                'd_avg_pct': safe_pct(bv('avg'), cv('avg')),
                'b_median': bv('median'), 'c_median': cv('median'),
                'd_median': safe_diff(bv('median'), cv('median')),
                'd_median_pct': safe_pct(bv('median'), cv('median')),
                'b_p90': bv('p90'), 'c_p90': cv('p90'),
                'd_p90': safe_diff(bv('p90'), cv('p90')),
                'd_p90_pct': safe_pct(bv('p90'), cv('p90')),
                'b_p95': bv('p95'), 'c_p95': cv('p95'),
                'd_p95': safe_diff(bv('p95'), cv('p95')),
                'd_p95_pct': safe_pct(bv('p95'), cv('p95')),
                'b_errors': bv('errors'), 'c_errors': cv('errors'),
                'd_errors': safe_diff(bv('errors'), cv('errors')),
                'b_min': bv('min'), 'c_min': cv('min'),
                'b_max': bv('max'), 'c_max': cv('max'),
            })
        return jsonify({'mode': 'compare', 'rows': rows,
                        'baseline': baseline_file, 'compare': compare_file})
    else:
        rows = []
        for lbl in sorted(base_stats.keys()):
            s = base_stats[lbl]
            rows.append({'label': lbl, 'type': classify_label(lbl), **s})
        return jsonify({'mode': 'single', 'rows': rows, 'baseline': baseline_file})


HTTP_CODES = {
    100:'Continue', 101:'Switching Protocols', 110:'Connection Timed Out',
    111:'Connection Refused', 200:'OK', 201:'Created', 202:'Accepted',
    204:'No Content', 206:'Partial Content', 301:'Moved Permanently',
    302:'Found', 303:'See Other', 304:'Not Modified', 307:'Temporary Redirect',
    310:'Too Many Redirects', 400:'Bad Request', 401:'Unauthorized',
    403:'Forbidden', 404:'Not Found', 405:'Method Not Allowed',
    408:'Request Timeout', 409:'Conflict', 410:'Gone', 413:'Request Entity Too Large',
    415:'Unsupported Media Type', 422:'Unprocessable Entity',
    429:'Too Many Requests', 500:'Internal Server Error', 502:'Bad Gateway',
    503:'Service Unavailable', 504:'Gateway Timeout',
}

def get_code_class(code):
    s = str(code)
    if not s: return 'unknown'
    if s[0] == '2': return 'success'
    if s[0] == '3': return 'redirect'
    if s[0] == '4': return 'client-error'
    if s[0] == '5': return 'server-error'
    return 'other'

@app.route('/api/errors', methods=['POST'])
def errors():
    data = request.json
    filename = data.get('filename')
    if filename not in loaded_files:
        return jsonify({'error': f'File not loaded: {filename}'}), 404

    df = loaded_files[filename].copy()

    if 'success' in df.columns:
        failed = df[~df['success']].copy()
    else:
        if 'responseCode' in df.columns:
            df['_rc'] = pd.to_numeric(df['responseCode'], errors='coerce')
            failed = df[~df['_rc'].between(200, 299, inclusive='both') | df['_rc'].isna()].copy()
        else:
            failed = pd.DataFrame()

    total_requests = len(df)
    total_errors   = len(failed)

    code_summary = []
    if 'responseCode' in df.columns:
        code_counts = df['responseCode'].value_counts().reset_index()
        code_counts.columns = ['code', 'total_count']
        if len(failed):
            err_counts = failed['responseCode'].value_counts().reset_index()
            err_counts.columns = ['code', 'err_count']
        else:
            err_counts = pd.DataFrame(columns=['code','err_count'])
        merged = code_counts.merge(err_counts, on='code', how='left')
        merged['err_count'] = merged['err_count'].fillna(0).astype(int)
        for _, row in merged.sort_values('err_count', ascending=False).iterrows():
            code = str(row['code'])
            try: code_int = int(float(code))
            except: code_int = None
            code_summary.append({
                'code': code,
                'code_int': code_int,
                'description': HTTP_CODES.get(code_int, ''),
                'code_class': get_code_class(code),
                'total_count': int(row['total_count']),
                'err_count':   int(row['err_count']),
                'err_pct': round(int(row['err_count']) / max(1, int(row['total_count'])) * 100, 1),
            })

    label_errors = []
    if len(failed) > 0 and 'label' in failed.columns:
        for lbl, grp in failed.groupby('label'):
            total_for_label = len(df[df['label'] == lbl]) if 'label' in df.columns else 0
            code_breakdown = grp['responseCode'].value_counts().to_dict() if 'responseCode' in grp.columns else {}
            failure_msgs = {}
            if 'failureMessage' in grp.columns:
                msgs = grp['failureMessage'].dropna().astype(str)
                failure_msgs = msgs.value_counts().head(3).to_dict()
            label_errors.append({
                'label': lbl,
                'type': classify_label(lbl),
                'error_count': len(grp),
                'total_count': total_for_label,
                'error_pct': round(len(grp) / max(1, total_for_label) * 100, 1),
                'codes': {str(k): int(v) for k, v in code_breakdown.items()},
                'top_messages': {str(k): int(v) for k, v in failure_msgs.items()},
            })
        label_errors.sort(key=lambda x: x['error_count'], reverse=True)

    timeline = []
    if 'rel_minute' in df.columns:
        for m, grp in df.groupby('rel_minute'):
            err_in_min = grp[~grp['success']] if 'success' in grp.columns else pd.DataFrame()
            timeline.append({
                'minute': int(m),
                'total': len(grp),
                'errors': len(err_in_min),
                'error_pct': round(len(err_in_min) / max(1, len(grp)) * 100, 1),
            })
        timeline.sort(key=lambda x: x['minute'])

    all_codes = sorted(df['responseCode'].dropna().astype(str).unique().tolist()) if 'responseCode' in df.columns else []

    return jsonify({
        'filename': filename,
        'total_requests': total_requests,
        'total_errors': total_errors,
        'error_rate': round(total_errors / max(1, total_requests) * 100, 2),
        'code_summary': code_summary,
        'label_errors': label_errors,
        'timeline': timeline,
        'all_codes': all_codes,
    })


@app.route('/api/errors_timeline', methods=['POST'])
def errors_timeline():
    data = request.json
    filename = data.get('filename')
    label    = data.get('label')

    if filename not in loaded_files:
        return jsonify({'error': f'File not loaded: {filename}'}), 404

    df = loaded_files[filename].copy()
    if label and 'label' in df.columns:
        df = df[df['label'] == label]

    if 'rel_minute' not in df.columns or len(df) == 0:
        return jsonify({'timeline': []})

    has_success = 'success' in df.columns
    rows = []
    for m, grp in df.groupby('rel_minute'):
        errs = int((~grp['success']).sum()) if has_success else 0
        rows.append({
            'minute':    int(m),
            'total':     len(grp),
            'errors':    errs,
            'error_pct': round(errs / max(1, len(grp)) * 100, 1),
        })
    rows.sort(key=lambda x: x['minute'])
    return jsonify({'timeline': rows, 'label': label})


# ── JMX comparison ────────────────────────────────────────────────────────────
import xml.etree.ElementTree as ET

def parse_jmx(filepath):
    """Parse JMX file into a tree structure for comparison."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    def node_to_dict(el, path=""):
        tag = el.tag
        name = el.get('testname') or el.get('name') or ''
        enabled = el.get('enabled', 'true')
        node_path = f"{path}/{tag}[{name}]" if name else f"{path}/{tag}"

        # Gather all properties (stringProp, intProp, boolProp, etc.)
        props = {}
        for child in el:
            if child.tag in ('stringProp','intProp','longProp','boolProp','floatProp','elementProp'):
                pname = child.get('name','')
                pval  = (child.text or '').strip()
                if pname:
                    props[pname] = pval
            # nested elementProp props
            if child.tag == 'elementProp':
                for gc in child:
                    pname = gc.get('name','')
                    pval  = (gc.text or '').strip()
                    if pname:
                        props[f"{child.get('name','')}.{pname}"] = pval

        # Children that are "structural" (not props)
        structural_tags = {
            'ThreadGroup','SetupThreadGroup','TearDownThreadGroup',
            'HTTPSamplerProxy','JDBCSampler','JSR223Sampler','BeanShellSampler',
            'GenericSampler','TCPSampler','FTPSampler','SMTPSampler',
            'TransactionController','LoopController','WhileController',
            'IfController','ForeachController','RunTime','CriticalSectionController',
            'InterleaveControl','RandomController','RandomOrderController',
            'ThroughputController','SwitchController',
            'HeaderManager','CookieManager','AuthManager','CacheManager',
            'DNSCacheManager','KeystoreConfig',
            'ResponseAssertion','JSONPathAssertion','XPathAssertion','DurationAssertion',
            'SizeAssertion','MD5HexAssertion','HTMLAssertion','XPathExtractor',
            'RegexExtractor','JSONPathExtractor','BoundaryExtractor','XPath2Extractor',
            'ConstantTimer','GaussianRandomTimer','UniformRandomTimer',
            'PoissonRandomTimer','SynchronizingTimer','ConstantThroughputTimer',
            'ResultCollector','BackendListener','DebugSampler',
            'CSVDataSet','CounterConfig','RandomVariableConfig',
            'Arguments','UserDefinedVariables',
            'hashTree','HTTPSamplerProxy',
        }

        children = []
        child_list = list(el)
        i = 0
        while i < len(child_list):
            c = child_list[i]
            if c.tag in structural_tags or (c.tag not in ('stringProp','intProp','longProp','boolProp','floatProp','elementProp','collectionProp')):
                children.append(node_to_dict(c, node_path))
            i += 1

        return {
            'tag': tag,
            'name': name,
            'enabled': enabled,
            'path': node_path,
            'props': props,
            'children': children,
        }

    return node_to_dict(root)


def flatten_tree(node, result=None, depth=0):
    """Flatten tree into list of nodes with depth for display."""
    if result is None:
        result = []
    result.append({
        'tag': node['tag'],
        'name': node['name'],
        'enabled': node['enabled'],
        'path': node['path'],
        'props': node['props'],
        'depth': depth,
    })
    for child in node.get('children', []):
        flatten_tree(child, result, depth + 1)
    return result


def compare_trees(flat_a, flat_b):
    """
    Compare two flattened trees. Match nodes by path.
    Returns annotated lists with diff status: 'same'|'modified'|'only_left'|'only_right'
    """
    map_a = {n['path']: n for n in flat_a}
    map_b = {n['path']: n for n in flat_b}

    all_paths_ordered = []
    seen = set()
    for n in flat_a:
        if n['path'] not in seen:
            all_paths_ordered.append(n['path'])
            seen.add(n['path'])
    for n in flat_b:
        if n['path'] not in seen:
            all_paths_ordered.append(n['path'])
            seen.add(n['path'])

    result_a, result_b = [], []
    diff_paths = []

    for path in all_paths_ordered:
        na = map_a.get(path)
        nb = map_b.get(path)

        if na and nb:
            # Both exist — compare props
            changed_props = {}
            all_prop_keys = set(na['props']) | set(nb['props'])
            for k in all_prop_keys:
                va = na['props'].get(k, '')
                vb = nb['props'].get(k, '')
                if va != vb:
                    changed_props[k] = {'left': va, 'right': vb}
            status = 'modified' if (changed_props or na['enabled'] != nb['enabled']) else 'same'
            ra = {**na, 'status': status, 'changed_props': changed_props}
            rb = {**nb, 'status': status, 'changed_props': changed_props}
        elif na:
            ra = {**na, 'status': 'only_left', 'changed_props': {}}
            rb = {'tag': '', 'name': '', 'enabled': 'true', 'path': path,
                  'props': {}, 'depth': na['depth'], 'status': 'only_left', 'changed_props': {}}
        else:
            ra = {'tag': '', 'name': '', 'enabled': 'true', 'path': path,
                  'props': {}, 'depth': nb['depth'], 'status': 'only_right', 'changed_props': {}}
            rb = {**nb, 'status': 'only_right', 'changed_props': {}}

        result_a.append(ra)
        result_b.append(rb)
        if status if na and nb else True:
            if (na and nb and status != 'same') or not (na and nb):
                diff_paths.append(path)

    return result_a, result_b, diff_paths


@app.route('/api/jmx/upload', methods=['POST'])
def jmx_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    if not filename.endswith('.jmx'):
        return jsonify({'error': 'File must be .jmx'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        tree = parse_jmx(filepath)
        flat = flatten_tree(tree)
        return jsonify({
            'filename': filename,
            'node_count': len(flat),
            'flat': flat,
        })
    except Exception as e:
        import traceback
        print("JMX PARSE ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/jmx/compare', methods=['POST'])
def jmx_compare():
    data = request.json
    file_a = data.get('file_a')
    file_b = data.get('file_b')

    path_a = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_a))
    path_b = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_b))

    if not os.path.exists(path_a):
        return jsonify({'error': f'File not found: {file_a}'}), 404
    if not os.path.exists(path_b):
        return jsonify({'error': f'File not found: {file_b}'}), 404

    try:
        tree_a = parse_jmx(path_a)
        tree_b = parse_jmx(path_b)
        flat_a = flatten_tree(tree_a)
        flat_b = flatten_tree(tree_b)
        result_a, result_b, diff_paths = compare_trees(flat_a, flat_b)

        stats = {
            'total': len(result_a),
            'same': sum(1 for n in result_a if n['status'] == 'same'),
            'modified': sum(1 for n in result_a if n['status'] == 'modified'),
            'only_left': sum(1 for n in result_a if n['status'] == 'only_left'),
            'only_right': sum(1 for n in result_b if n['status'] == 'only_right'),
        }

        return jsonify({
            'left': result_a,
            'right': result_b,
            'diff_paths': diff_paths,
            'stats': stats,
            'file_a': file_a,
            'file_b': file_b,
        })
    except Exception as e:
        import traceback
        print("JMX COMPARE ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
