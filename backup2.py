# jmeter_analyzer_full.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import hashlib
import re

# ---------------------------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------------------------
st.set_page_config(page_title="JMeter Results Analyzer (full)", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Pomocnicze funkcje
# ---------------------------------------------------------------------------

def deterministic_rgb_from_label(label: str):
    """Zwraca deterministyczne (r,g,b) z napisu (użyte do kolorów)."""
    if label is None:
        return (128, 128, 128)
    h = hashlib.md5(str(label).encode('utf-8')).digest()
    return (h[0], h[1], h[2])

def compute_missing_dates_for_rangebreaks(df, date_col='timestamp_dt'):
    """
    Zwraca listę dat YYYY-MM-DD, które są pomiędzy min a max, ale nie występują w df.
    Przydatne dla Plotly rangebreaks, żeby pominąć puste dni.
    """
    if df is None or df.empty or date_col not in df.columns:
        return []
    series = pd.to_datetime(df[date_col], errors='coerce').dropna()
    if series.empty:
        return []
    min_d = series.min().normalize()
    max_d = series.max().normalize()
    if (max_d - min_d).days < 1:
        return []
    present_days = set(series.dt.normalize().dt.strftime("%Y-%m-%d").unique())
    missing = []
    d = min_d
    while d <= max_d:
        ds = d.strftime("%Y-%m-%d")
        if ds not in present_days:
            missing.append(ds)
        d += pd.Timedelta(days=1)
    return missing

def safe_get_unique_values(series, sort_func=None):
    try:
        if series is None:
            return []
        unique_vals = pd.Series(series).dropna().unique()
        if len(unique_vals) == 0:
            return []
        unique_vals_str = [str(val) for val in unique_vals]
        if sort_func:
            return sort_func(unique_vals_str)
        else:
            return sorted(unique_vals_str)
    except Exception as e:
        st.warning(f"Błąd podczas pobierania unikalnych wartości: {str(e)}")
        return []

# ---------------------------------------------------------------------------
# Parsowanie plików (raw i aggregate/summary)
# ---------------------------------------------------------------------------

def load_jmeter_csv_from_bytes(content_bytes, filename="uploaded_file"):
    """
    Spróbuj wczytać raw JMeter CSV (timeStamp, elapsed, label, responseCode, success, ...).
    Zwraca (df, info_str) lub (None, None).
    """
    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1250']
    expected_cols = ['timestamp', 'timestamp_dt', 'timestamp_ms', 'timestamp_ms', 'elapsed', 'label', 'responsecode', 'success', 'timeStamp']

    for enc in encodings:
        try:
            text = content_bytes.decode(enc)
        except Exception:
            continue

        first_line = text.splitlines()[0] if text else ''
        sep = ';' if ';' in first_line else (',' if ',' in first_line else None)

        # 1) standard
        try:
            buf = io.StringIO(text)
            if sep:
                df = pd.read_csv(buf, sep=sep, engine='c', on_bad_lines='skip')
            else:
                df = pd.read_csv(buf, engine='python', on_bad_lines='skip')
            lc = {c.lower() for c in df.columns}
            if any(x in lc for x in ['timestamp', 'timestamp_dt', 'timestamp_ms', 'timeStamp'.lower(), 'elapsed', 'label']):
                return df, f"{enc} (sep: {sep})"
        except Exception:
            pass

        # 2) tolerancyjne
        try:
            buf = io.StringIO(text)
            df = pd.read_csv(buf, sep=sep or ',', quotechar='"', skipinitialspace=True,
                             on_bad_lines='skip', engine='python')
            lc = {c.lower() for c in df.columns}
            if any(x in lc for x in ['timestamp', 'timeStamp'.lower(), 'elapsed', 'label']):
                return df, f"{enc} (sep: {sep}, tolerancyjne)"
        except Exception:
            pass

        # 3) auto sep
        try:
            buf = io.StringIO(text)
            df = pd.read_csv(buf, sep=None, engine='python', on_bad_lines='skip', skipinitialspace=True)
            lc = {c.lower() for c in df.columns}
            if any(x in lc for x in ['timestamp', 'timeStamp'.lower(), 'elapsed', 'label']):
                return df, f"{enc} (auto-sep)"
        except Exception:
            pass

    return None, None

def normalize_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje kolumny agregatów na standaryzowane nazwy (lower-case).
    Zwraca zmodyfikowany df.
    """
    col_map = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc == 'label':
            col_map[c] = 'label'
        elif lc in ['# samples', '#samples', 'samples', 'sample#', 'sample count']:
            col_map[c] = 'samples'
        elif 'average' in lc and 'line' not in lc:
            col_map[c] = 'average'
        elif 'median' in lc:
            col_map[c] = 'median'
        elif '90' in lc and 'line' in lc or '90%' in lc or lc == '90% line':
            col_map[c] = 'p90'
        elif '95' in lc and 'line' in lc or '95%' in lc or lc == '95% line':
            col_map[c] = 'p95'
        elif '99' in lc and 'line' in lc or '99%' in lc or lc == '99% line':
            col_map[c] = 'p99'
        elif lc == 'min' or lc.startswith('min'):
            col_map[c] = 'min'
        elif lc == 'max' or lc.startswith('max'):
            col_map[c] = 'max'
        elif 'error' in lc and '%' in lc or lc == 'error %' or lc == 'error%':
            col_map[c] = 'error'
        elif 'throughput' in lc:
            col_map[c] = 'throughput'
        elif 'received' in lc and 'kb' in lc:
            col_map[c] = 'received_kb_sec'
        elif 'std' in lc:
            col_map[c] = 'std_dev'
        else:
            # niechodzące kolumny mogą pozostać oryginalne - ale wyżej spróbujemy dopasować
            pass

    df = df.rename(columns=col_map)

    # Konwersja pól liczbowych
    def to_num(series):
        s = series.astype(str).str.replace('%', '', regex=False)
        s = s.str.replace('\xa0', '', regex=False)  # non-breakable space
        s = s.str.replace(' ', '', regex=False)
        s = s.str.replace(',', '.', regex=False)
        return pd.to_numeric(s.replace(['', 'nan', 'None'], np.nan), errors='coerce')

    for cand in ['samples', 'average', 'median', 'p90', 'p95', 'p99', 'min', 'max', 'throughput', 'received_kb_sec', 'std_dev']:
        if cand in df.columns:
            df[cand] = to_num(df[cand])

    if 'error' in df.columns:
        df['error'] = df['error'].astype(str).str.replace('%', '', regex=False)
        df['error'] = df['error'].str.replace(',', '.', regex=False)
        df['error'] = pd.to_numeric(df['error'].replace(['', 'nan', 'None'], np.nan), errors='coerce') / 100.0

    return df

def load_jmeter_summary_from_bytes(content_bytes, filename="uploaded_file"):
    """
    Wczytuje summary/aggregate-like report (CSV) i normalizuje kolumny.
    """
    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1250']
    for enc in encodings:
        try:
            text = content_bytes.decode(enc)
        except Exception:
            continue

        for sep in [',', ';']:
            try:
                buf = io.StringIO(text)
                df = pd.read_csv(buf, sep=sep, engine='python', on_bad_lines='skip')
                if df.empty:
                    continue
                df.columns = [str(c).strip() for c in df.columns]
                lc = {c.lower() for c in df.columns}
                if 'label' in lc and (('# samples' in lc) or ('average' in lc) or ('90% line' in lc) or ('95% line' in lc)):
                    df = normalize_aggregate_columns(df)
                    df['source_file'] = filename
                    return df, f"{enc} (sep: {sep})"
            except Exception:
                continue
    return None, None

# ---------------------------------------------------------------------------
# Preprocessing danych raw
# ---------------------------------------------------------------------------

def preprocess_data_raw(df):
    df = df.copy()
    # unify label column name
    for c in df.columns:
        if str(c).lower().strip() == 'label':
            df = df.rename(columns={c: 'label'})
            break
    # timestamp
    if 'timeStamp' in df.columns:
        try:
            # timeStamp is usually ms
            df['timestamp_dt'] = pd.to_datetime(df['timeStamp'], unit='ms', errors='coerce')
        except Exception:
            try:
                df['timestamp_dt'] = pd.to_datetime(df['timeStamp'], errors='coerce')
            except Exception:
                df['timestamp_dt'] = pd.NaT
    elif 'timestamp' in df.columns:
        try:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception:
            df['timestamp_dt'] = pd.NaT

    # responseCode normalization
    for c in df.columns:
        if str(c).lower().strip() == 'responsecode':
            df = df.rename(columns={c: 'responseCode'})
            break

    if 'responseCode' in df.columns:
        df['responseCode'] = df['responseCode'].astype(str).str.strip().replace(['nan', 'None', ''], 'Unknown')

    # success
    for c in df.columns:
        if str(c).lower().strip() == 'success':
            df = df.rename(columns={c: 'success'})
            break
    if 'success' not in df.columns and 'responseCode' in df.columns:
        df['success'] = df['responseCode'].astype(str).str.startswith('2')
    else:
        if 'success' in df.columns:
            df['success'] = df['success'].astype(str).str.lower().isin(['true', '1', 'yes'])

    # elapsed numeric
    for c in df.columns:
        if str(c).lower().strip() == 'elapsed':
            df = df.rename(columns={c: 'elapsed'})
            break
    if 'elapsed' in df.columns:
        df['elapsed'] = pd.to_numeric(df['elapsed'], errors='coerce')

    df['source_file'] = df.get('source_file', 'unknown')
    return df

# ---------------------------------------------------------------------------
# Statystyki
# ---------------------------------------------------------------------------

def calculate_statistics_raw(df):
    stats = {}
    if df is None or df.empty:
        return stats

    if 'elapsed' in df.columns:
        ec = df['elapsed'].dropna()
        if not ec.empty:
            stats['avg_response_time'] = float(ec.mean())
            stats['median_response_time'] = float(ec.median())
            stats['p90'] = float(ec.quantile(0.9))
            stats['p95'] = float(ec.quantile(0.95))
            stats['p99'] = float(ec.quantile(0.99))
            stats['min_response_time'] = float(ec.min())
            stats['max_response_time'] = float(ec.max())

    stats['total_requests'] = int(len(df))

    if 'success' in df.columns:
        succ = df['success'].fillna(False)
        stats['successful_requests'] = int(succ.sum())
        stats['failed_requests'] = int((~succ).sum())
        if stats['total_requests'] > 0:
            stats['success_rate'] = (stats['successful_requests'] / stats['total_requests']) * 100

    if 'timestamp_dt' in df.columns:
        ts = df['timestamp_dt'].dropna()
        if not ts.empty:
            duration = (ts.max() - ts.min()).total_seconds()
            stats['test_duration_sec'] = duration
            if duration > 0:
                stats['throughput'] = stats['total_requests'] / duration

    return stats

def calculate_statistics_aggregate(df):
    stats = {}
    if df is None or df.empty:
        return stats
    if 'samples' in df.columns:
        stats['total_samples'] = int(df['samples'].sum(skipna=True))
    if 'average' in df.columns:
        stats['avg_of_averages'] = float(df['average'].dropna().mean()) if df['average'].notna().any() else None
        # weighted average if samples present
        if 'samples' in df.columns and df['samples'].notna().sum() > 0:
            valid = df[df['samples'].notna() & df['average'].notna()]
            total_samples = valid['samples'].sum()
            if total_samples > 0:
                stats['weighted_avg_response_time'] = float((valid['average'] * valid['samples']).sum() / total_samples)
    if 'p95' in df.columns:
        stats['p95_agg_median'] = float(df['p95'].dropna().median()) if df['p95'].notna().any() else None
    if 'throughput' in df.columns:
        stats['total_throughput_sum'] = float(df['throughput'].sum(skipna=True))
    return stats

# ---------------------------------------------------------------------------
# Wykresy (raw)
# ---------------------------------------------------------------------------

def plot_response_times_over_time(df):
    """Wykres czasów odpowiedzi dla raw - pomija dni bez danych (rangebreaks)."""
    if df is None or df.empty or 'timestamp_dt' not in df.columns or 'elapsed' not in df.columns:
        return None

    df_clean = df.dropna(subset=['timestamp_dt', 'elapsed']).sort_values('timestamp_dt')
    if df_clean.empty:
        return None

    # jeśli dużo labeli, grupuj
    if 'label' in df_clean.columns and df_clean['label'].nunique() <= 10:
        fig = px.line(df_clean, x='timestamp_dt', y='elapsed', color='label', title='Czasy odpowiedzi w czasie')
    else:
        fig = px.line(df_clean, x='timestamp_dt', y='elapsed', title='Czasy odpowiedzi w czasie')

    # moving average (30s) - jeśli wystarczająco dużo danych
    if len(df_clean) >= 50:
        try:
            moving_avg = df_clean.set_index('timestamp_dt')['elapsed'].rolling(window='30S').mean()
            fig.add_scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', name='Średnia krocząca (30s)', line=dict(width=3, dash='dash'))
        except Exception:
            pass

    missing_dates = compute_missing_dates_for_rangebreaks(df_clean, 'timestamp_dt')
    if missing_dates:
        fig.update_xaxes(rangebreaks=[dict(values=missing_dates)])

    fig.update_layout(hovermode='x unified', xaxis_title='Czas', yaxis_title='Czas odpowiedzi (ms)',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_response_time_histogram(df):
    if df is None or df.empty or 'elapsed' not in df.columns:
        return None
    elapsed_clean = df['elapsed'].dropna()
    if elapsed_clean.empty:
        return None
    fig = px.histogram(elapsed_clean, x=elapsed_clean, nbins=50, title='Rozkład czasów odpowiedzi', labels={'x': 'Czas odpowiedzi (ms)', 'count': 'Liczba żądań'})
    p90 = elapsed_clean.quantile(0.9)
    p95 = elapsed_clean.quantile(0.95)
    p99 = elapsed_clean.quantile(0.99)
    fig.add_vline(x=p90, line_dash='dash', annotation_text=f'P90: {p90:.0f}ms')
    fig.add_vline(x=p95, line_dash='dash', annotation_text=f'P95: {p95:.0f}ms')
    fig.add_vline(x=p99, line_dash='dash', annotation_text=f'P99: {p99:.0f}ms')
    return fig

def plot_throughput_over_time(df):
    if df is None or df.empty or 'timestamp_dt' not in df.columns:
        return None
    df_clean = df.dropna(subset=['timestamp_dt'])
    if df_clean.empty:
        return None
    if 'label' in df_clean.columns:
        df_grouped = df_clean.groupby([df_clean['timestamp_dt'].dt.floor('S'), 'label']).size().reset_index()
        df_grouped.columns = ['timestamp', 'label', 'requests_per_second']
        top_labels = df_clean['label'].value_counts().head(5).index
        df_grouped_filtered = df_grouped[df_grouped['label'].isin(top_labels)]
        fig = px.line(df_grouped_filtered, x='timestamp', y='requests_per_second', color='label', title='Przepustowość według typu requestów (żądania na sekundę)')
        total_throughput = df_clean.groupby(df_clean['timestamp_dt'].dt.floor('S')).size().reset_index()
        total_throughput.columns = ['timestamp', 'total_requests_per_second']
        fig.add_scatter(x=total_throughput['timestamp'], y=total_throughput['total_requests_per_second'], mode='lines', name='CAŁKOWITA PRZEPUSTOWOŚĆ', line=dict(color='black', width=3, dash='dash'))
    else:
        df_grouped = df_clean.groupby(df_clean['timestamp_dt'].dt.floor('S')).size().reset_index()
        df_grouped.columns = ['timestamp', 'requests_per_second']
        fig = px.line(df_grouped, x='timestamp', y='requests_per_second', title='Przepustowość (żądania na sekundę)')
    # moving avg
    try:
        moving_avg = df_grouped.set_index('timestamp')['requests_per_second'].rolling(window=10).mean()
        fig.add_scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', name='Średnia krocząca (10s)', line=dict(width=2, dash='dot'))
    except Exception:
        pass
    missing_dates = compute_missing_dates_for_rangebreaks(df_clean, 'timestamp_dt')
    if missing_dates:
        fig.update_xaxes(rangebreaks=[dict(values=missing_dates)])
    fig.update_layout(hovermode='x unified', xaxis_title='Czas', yaxis_title='Żądania/s')
    return fig

def plot_error_analysis(df):
    if df is None or df.empty or 'responseCode' not in df.columns:
        return None
    response_codes_clean = df['responseCode'].fillna('Unknown')
    error_counts = response_codes_clean.value_counts().head(10)
    if error_counts.empty:
        return None
    fig = px.bar(x=error_counts.index, y=error_counts.values, title='Top 10 kodów odpowiedzi', labels={'x': 'Kod odpowiedzi', 'y': 'Liczba wystąpień'})
    return fig

def plot_label_performance(df):
    if df is None or df.empty or 'label' not in df.columns or 'elapsed' not in df.columns:
        return None
    df_clean = df.dropna(subset=['label', 'elapsed'])
    if df_clean.empty:
        return None
    label_stats = df_clean.groupby('label')['elapsed'].agg(['mean', 'median', 'count']).reset_index()
    label_stats = label_stats.sort_values('mean', ascending=False).head(15)
    if label_stats.empty:
        return None
    fig = px.bar(label_stats, x='label', y='mean', title='Średni czas odpowiedzi według typu żądania (Top 15)', labels={'label': 'Typ żądania', 'mean': 'Średni czas odpowiedzi (ms)'})
    fig.update_layout(xaxis_tickangle=45)
    return fig

# ---------------------------------------------------------------------------
# Wykresy (aggregate)
# ---------------------------------------------------------------------------

def plot_response_times_aggregated_by_label(df):
    if df is None or df.empty or 'label' not in df.columns:
        return None
    y_cols = []
    for cand in ['average', 'p90', 'p95', 'p99', 'median']:
        if cand in df.columns:
            y_cols.append(cand)
    if not y_cols:
        return None
    melted = df.melt(id_vars=['label'], value_vars=y_cols, var_name='metric', value_name='ms')
    fig = px.bar(melted, x='label', y='ms', color='metric', barmode='group', title='Średnie i percentyle czasów odpowiedzi (per label)')
    fig.update_layout(xaxis_tickangle=45)
    return fig

def plot_throughput_aggregate_by_label(df):
    if df is None or df.empty or 'throughput' not in df.columns or 'label' not in df.columns:
        return None
    fig = px.bar(df.sort_values('throughput', ascending=False), x='label', y='throughput', title='Throughput per Label')
    fig.update_layout(xaxis_tickangle=45)
    return fig

def plot_error_percent_aggregate(df):
    if df is None or df.empty or 'error' not in df.columns or 'label' not in df.columns:
        return None
    fig = px.bar(df.sort_values('error', ascending=False), x='label', y='error', title='Error % per Label', labels={'error': 'Error (fraction)'})
    fig.update_layout(xaxis_tickangle=45)
    return fig

# ---------------------------------------------------------------------------
# Główna aplikacja
# ---------------------------------------------------------------------------

def main():
    st.title("📊 JMeter Results Analyzer — full")

    st.markdown("Analiza **raw logs** oraz **agregowanych raportów** JMeter. Filtry działają dla wybranego źródła danych.")

    # Sidebar: upload + diagnostics
    st.sidebar.header("Wczytaj pliki CSV")
    uploaded_files = st.sidebar.file_uploader("Wybierz pliki CSV z JMetera (raw/summary/aggregate)", type=['csv'], accept_multiple_files=True)

    st.sidebar.subheader("🔍 Diagnostyka pliku")
    if st.sidebar.checkbox("Pokaż diagnostykę pliku"):
        diagnostic_file = st.sidebar.file_uploader("Wybierz plik do diagnostyki", type=['csv'], key="diagnostic")
        if diagnostic_file is not None:
            try:
                diagnostic_file.seek(0)
                raw = diagnostic_file.read()
                # spróbuj dekodować na utf-8, jeśli nie to pokaż surowe bajty
                try:
                    text = raw.decode('utf-8')
                except Exception:
                    try:
                        text = raw.decode('iso-8859-1')
                    except Exception:
                        text = str(raw[:1000])
                first_lines = "\n".join(text.splitlines()[:20])
            except Exception as e:
                first_lines = f"(Błąd czytania pliku diagnostycznego: {e})"
            st.sidebar.text_area("📄 Pierwsze linie pliku:", first_lines, height=200)

    if not uploaded_files:
        st.info("👈 Wgraj pliki CSV (raw lub aggregate), aby rozpocząć analizę.")
        return

    all_raw = []
    all_aggregate = []
    file_info = []

    # Wczytywanie i rozpoznawanie plików
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read()
            # najpierw spróbuj raw
            df_raw, enc_info = load_jmeter_csv_from_bytes(content, uploaded_file.name)
            if df_raw is not None:
                df_raw = preprocess_data_raw(df_raw)
                df_raw['source_file'] = uploaded_file.name
                all_raw.append(df_raw)
                file_info.append({'filename': uploaded_file.name, 'rows': len(df_raw), 'encoding': enc_info, 'columns': list(df_raw.columns)})
                continue
            # spróbuj aggregate/summary
            df_agg, enc_info = load_jmeter_summary_from_bytes(content, uploaded_file.name)
            if df_agg is not None:
                df_agg['source_file'] = uploaded_file.name
                all_aggregate.append(df_agg)
                file_info.append({'filename': uploaded_file.name, 'rows': len(df_agg), 'encoding': enc_info, 'columns': list(df_agg.columns)})
                continue
            st.warning(f"Plik {uploaded_file.name} nie został rozpoznany jako raw ani agregat/summary")
        except Exception as e:
            st.error(f"Błąd podczas wczytywania pliku {uploaded_file.name}: {e}")

    combined_raw = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    combined_aggregate = pd.concat(all_aggregate, ignore_index=True) if all_aggregate else pd.DataFrame()

    # Sidebar: wybór źródła danych i filtry
    st.sidebar.header("🔎 Źródło danych do analizy")
    source_choices = []
    if not combined_raw.empty:
        source_choices.append("Szczegółowe (raw)")
    if not combined_aggregate.empty:
        source_choices.append("Zagregowane (aggregate)")
    if not source_choices:
        st.error("Nie rozpoznano żadnych wczytanych plików. Sprawdź ich format.")
        return
    data_source = st.sidebar.radio("Wybierz źródło danych:", source_choices)

    # Unia label z obu źródeł (użyteczna dla globalnego filtrowania)
    labels_raw = safe_get_unique_values(combined_raw['label']) if not combined_raw.empty and 'label' in combined_raw.columns else []
    labels_agg = safe_get_unique_values(combined_aggregate['label']) if not combined_aggregate.empty and 'label' in combined_aggregate.columns else []
    unique_labels = sorted(set(labels_raw + labels_agg))

    st.sidebar.header("🔍 Filtry danych")
    selected_labels = []
    if unique_labels:
        filter_type = st.sidebar.radio("Sposób filtrowania requestów:", ["Wszystkie requesty", "Wybierz z listy", "Wyszukaj po nazwie"], key="filter_type")
        if filter_type == "Wybierz z listy":
            selected_labels = st.sidebar.multiselect("Wybierz requesty do analizy:", unique_labels, default=unique_labels[:5] if len(unique_labels) > 5 else unique_labels, key="label_multiselect")
        elif filter_type == "Wyszukaj po nazwie":
            search_term = st.sidebar.text_input("Wpisz część nazwy requestu:", placeholder="np. Initialize, Execute, GET", key="search_input")
            if search_term:
                matching_labels = [label for label in unique_labels if search_term.lower() in label.lower()]
                if matching_labels:
                    selected_labels = st.sidebar.multiselect(f"Znalezione requesty ({len(matching_labels)}):", matching_labels, default=matching_labels, key="search_results")
                else:
                    st.sidebar.warning("Nie znaleziono requestów pasujących do wyszukiwania")
        else:
            selected_labels = unique_labels

    selected_codes = None
    status_filter = "Wszystkie"
    if data_source == "Szczegółowe (raw)" and not combined_raw.empty and 'responseCode' in combined_raw.columns:
        st.sidebar.subheader("Kody odpowiedzi")
        unique_codes = safe_get_unique_values(combined_raw['responseCode'])
        selected_codes = st.sidebar.multiselect("Filtruj po kodach odpowiedzi:", unique_codes, default=unique_codes, key="response_codes")

    if data_source == "Szczegółowe (raw)" and not combined_raw.empty and 'success' in combined_raw.columns:
        st.sidebar.subheader("Status requestów")
        status_filter = st.sidebar.selectbox("Pokaż requesty:", ["Wszystkie", "Tylko udane", "Tylko nieudane"], key="status_filter")

    # Filtr wydajności (opcjonalny)
    st.sidebar.subheader("Filtr wydajności (opcjonalnie)")
    enable_perf = st.sidebar.checkbox("Włącz filtr wydajności", key='enable_perf_filter')
    perf_filter_type = None
    custom_time_range = None
    if enable_perf:
        perf_filter_type = st.sidebar.selectbox("Wybierz typ filtra:", ["Szybkie (< P95)", "Wolne (> P95)", "Niestandardowy zakres"], key='perf_filter_type')
        if perf_filter_type == "Niestandardowy zakres":
            if data_source == "Szczegółowe (raw)":
                min_time = st.sidebar.number_input("Min (ms)", value=0, key='custom_min_raw')
                max_time = st.sidebar.number_input("Max (ms)", value=1000, key='custom_max_raw')
            else:
                min_time = st.sidebar.number_input("Min average (ms)", value=0, key='custom_min_agg')
                max_time = st.sidebar.number_input("Max average (ms)", value=1000, key='custom_max_agg')
            custom_time_range = (min_time, max_time)
            st.session_state['custom_time_range'] = custom_time_range

    # Zastosowanie filtrów
    if data_source == "Szczegółowe (raw)":
        filtered_df = combined_raw.copy() if not combined_raw.empty else pd.DataFrame()
        orig_len = len(filtered_df)
        if not filtered_df.empty:
            if 'label' in filtered_df.columns and selected_labels and len(selected_labels) < len(safe_get_unique_values(filtered_df['label'])):
                filtered_df = filtered_df[filtered_df['label'].astype(str).isin(selected_labels)]
            if selected_codes and 'responseCode' in filtered_df.columns and len(selected_codes) < len(safe_get_unique_values(filtered_df['responseCode'])):
                filtered_df = filtered_df[filtered_df['responseCode'].astype(str).isin(selected_codes)]
            if status_filter == "Tylko udane" and 'success' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['success'] == True]
            elif status_filter == "Tylko nieudane" and 'success' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['success'] == False]
            if enable_perf and 'elapsed' in filtered_df.columns:
                if perf_filter_type == "Szybkie (< P95)":
                    p95_elapsed = combined_raw['elapsed'].quantile(0.95)
                    filtered_df = filtered_df[filtered_df['elapsed'] < p95_elapsed]
                elif perf_filter_type == "Wolne (> P95)":
                    p95_elapsed = combined_raw['elapsed'].quantile(0.95)
                    filtered_df = filtered_df[filtered_df['elapsed'] > p95_elapsed]
                elif perf_filter_type == "Niestandardowy zakres" and 'custom_time_range' in st.session_state:
                    min_time, max_time = st.session_state['custom_time_range']
                    filtered_df = filtered_df[(filtered_df['elapsed'] >= min_time) & (filtered_df['elapsed'] <= max_time)]
    else:
        filtered_df = combined_aggregate.copy() if not combined_aggregate.empty else pd.DataFrame()
        orig_len = len(filtered_df)
        if not filtered_df.empty:
            if 'label' in filtered_df.columns and selected_labels and len(selected_labels) < len(safe_get_unique_values(filtered_df['label'])):
                filtered_df = filtered_df[filtered_df['label'].astype(str).isin(selected_labels)]
            if enable_perf and 'average' in filtered_df.columns:
                if perf_filter_type == "Szybkie (< P95)":
                    p95_val = combined_aggregate['average'].quantile(0.95)
                    filtered_df = filtered_df[filtered_df['average'] < p95_val]
                elif perf_filter_type == "Wolne (> P95)":
                    p95_val = combined_aggregate['average'].quantile(0.95)
                    filtered_df = filtered_df[filtered_df['average'] > p95_val]
                elif perf_filter_type == "Niestandardowy zakres" and 'custom_time_range' in st.session_state:
                    min_time, max_time = st.session_state['custom_time_range']
                    filtered_df = filtered_df[(filtered_df['average'] >= min_time) & (filtered_df['average'] <= max_time)]

    # Sidebar info
    if orig_len > 0 and len(filtered_df) != orig_len:
        reduction_pct = ((orig_len - len(filtered_df)) / orig_len * 100) if orig_len > 0 else 0
        st.sidebar.success(f"📊 Przefiltrowano: {len(filtered_df):,} z {orig_len:,} rekordów (-{reduction_pct:.1f}%)")
    st.sidebar.success(f"✅ Wczytano {len(uploaded_files)} plików")
    st.sidebar.write(f"Łącznie rekordów (raw logs): {len(combined_raw)}")
    st.sidebar.write(f"Agregowanych raportów: {len(all_aggregate)}")

    # Główne widoki: zduplikowane zakładki dla raw i aggregate
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Przegląd", "⏱️ Czasy odpowiedzi", "🚀 Przepustowość", "❌ Błędy", "📋 Szczegóły"])

    # TAB: Przegląd
    with tab1:
        st.header("Przegląd")
        if data_source == "Szczegółowe (raw)":
            if combined_raw.empty:
                st.info("Brak raw logs.")
            else:
                stats = calculate_statistics_raw(filtered_df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Łączna liczba żądań", f"{stats.get('total_requests', 0):,}")
                    if 'success_rate' in stats:
                        st.metric("Wskaźnik sukcesu", f"{stats['success_rate']:.1f}%")
                with col2:
                    if 'avg_response_time' in stats:
                        st.metric("Średni czas odpowiedzi", f"{stats['avg_response_time']:.0f} ms")
                        st.metric("Mediana", f"{stats['median_response_time']:.0f} ms")
                with col3:
                    if 'p95' in stats:
                        st.metric("95 percentyl", f"{stats['p95']:.0f} ms")
                        st.metric("99 percentyl", f"{stats['p99']:.0f} ms")
                with col4:
                    if 'throughput' in stats:
                        st.metric("Przepustowość", f"{stats['throughput']:.1f} req/s")
                    if 'test_duration_sec' in stats:
                        st.metric("Czas trwania testu", f"{stats['test_duration_sec']:.0f} s")
                if 'label' in filtered_df.columns:
                    fig_labels = plot_label_performance(filtered_df)
                    if fig_labels:
                        st.plotly_chart(fig_labels, use_container_width=True)
        else:
            # aggregate
            if combined_aggregate.empty:
                st.info("Brak agregatów.")
            else:
                stats_agg = calculate_statistics_aggregate(filtered_df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Liczba etykiet", f"{len(filtered_df) if filtered_df is not None else 0}")
                    if 'total_samples' in stats_agg:
                        st.metric("Łącznie próbek (# samples)", f"{stats_agg['total_samples']:,}")
                with col2:
                    if 'weighted_avg_response_time' in stats_agg and stats_agg['weighted_avg_response_time'] is not None:
                        st.metric("Ważona średnia (average)", f"{stats_agg['weighted_avg_response_time']:.0f} ms")
                    elif 'avg_of_averages' in stats_agg and stats_agg['avg_of_averages'] is not None:
                        st.metric("Średnia z Average", f"{stats_agg['avg_of_averages']:.0f} ms")
                with col3:
                    if 'p95_agg_median' in stats_agg:
                        st.metric("95 percentile (przybliżenie)", f"{stats_agg['p95_agg_median']:.0f} ms")
                with col4:
                    if 'total_throughput_sum' in stats_agg:
                        st.metric("Suma throughput (sum per label)", f"{stats_agg['total_throughput_sum']:.2f}")

    # TAB: Czasy odpowiedzi
    with tab2:
        st.header("Czasy odpowiedzi")
        if data_source == "Szczegółowe (raw)":
            if not filtered_df.empty and 'timestamp_dt' in filtered_df.columns:
                fig_time = plot_response_times_over_time(filtered_df)
                if fig_time:
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.warning("Nie można wygenerować wykresu czasów odpowiedzi - brak odpowiednich danych")
            else:
                st.warning("Brak danych czasowych w raw logs (brak kolumny timestamp/timeStamp albo puste wartości).")
            fig_hist = plot_response_time_histogram(filtered_df)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            # aggregate -> bar chart dla Average/Median/P90/P95/P99
            if not filtered_df.empty:
                fig_a = plot_response_times_aggregated_by_label(filtered_df)
                if fig_a:
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.info("Brak kolumn Average/percentyli w agregatach.")
            else:
                st.info("Brak danych agregowanych.")

    # TAB: Przepustowość
    with tab3:
        st.header("Przepustowość")
        if data_source == "Szczegółowe (raw)":
            fig_tp = plot_throughput_over_time(filtered_df)
            if fig_tp:
                st.plotly_chart(fig_tp, use_container_width=True)
            else:
                st.warning("Nie można wygenerować wykresu przepustowości - brak danych czasowych")
        else:
            fig_tp_a = plot_throughput_aggregate_by_label(filtered_df)
            if fig_tp_a:
                st.plotly_chart(fig_tp_a, use_container_width=True)
            else:
                st.info("Brak danych throughput w agregatach")

    # TAB: Błędy
    with tab4:
        st.header("Analiza błędów")
        if data_source == "Szczegółowe (raw)":
            if 'responseCode' in filtered_df.columns:
                fig_err = plot_error_analysis(filtered_df)
                if fig_err:
                    st.plotly_chart(fig_err, use_container_width=True)
                else:
                    st.info("Brak danych o kodach odpowiedzi")
                if 'responseCode' in filtered_df.columns and 'success' in filtered_df.columns:
                    failed_requests = filtered_df[~filtered_df['success']]
                    if not failed_requests.empty:
                        st.subheader("Szczegóły błędów")
                        error_details = failed_requests.groupby(['responseCode', 'label']).size().reset_index(name='count')
                        error_details = error_details.sort_values('count', ascending=False)
                        st.dataframe(error_details)
                    else:
                        st.success("🎉 Brak błędów w przefiltrowanych danych!")
            else:
                st.info("Brak danych o kodach odpowiedzi w raw logs")
        else:
            fig_err_a = plot_error_percent_aggregate(filtered_df)
            if fig_err_a:
                st.plotly_chart(fig_err_a, use_container_width=True)
            else:
                st.info("Brak kolumn error% w agregatach")

    # TAB: Szczegóły
    with tab5:
        st.header("Szczegóły")
        if data_source == "Szczegółowe (raw)":
            if not filtered_df.empty:
                st.dataframe(filtered_df.head(200))
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(label="📥 Pobierz przefiltrowane CSV (raw)", data=filtered_df.to_csv(index=False, sep=';'), file_name=f"jmeter_filtered_results_raw_{len(filtered_df)}_records.csv", mime="text/csv")
                with col2:
                    st.download_button(label="📥 Pobierz wszystkie raw CSV", data=combined_raw.to_csv(index=False, sep=';'), file_name=f"jmeter_all_results_raw_{len(combined_raw)}_records.csv", mime="text/csv")
            else:
                st.info("Brak raw logs do wyświetlenia")
        else:
            if not filtered_df.empty:
                st.dataframe(filtered_df)
                st.download_button(label="📥 Pobierz agregowane raporty jako CSV", data=filtered_df.to_csv(index=False, sep=';'), file_name=f"jmeter_summary_aggregated_{len(filtered_df)}_rows.csv", mime="text/csv")
            else:
                st.info("Brak agregatów do wyświetlenia")

    # Sekcja ze szczegółami plików
    st.markdown("---")
    st.subheader("📁 Informacje o wczytanych plikach")
    for info in file_info:
        with st.expander(f"📄 {info['filename']}"):
            st.write(f"**Liczba rekordów:** {info['rows']:,}")
            st.write(f"**Kodowanie/uwagi:** {info['encoding']}")
            st.write(f"**Kolumny:** {', '.join(info['columns'])}")

if __name__ == "__main__":
    main()
