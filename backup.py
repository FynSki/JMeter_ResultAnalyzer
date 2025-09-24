import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
import re

# Konfiguracja strony
st.set_page_config(
    page_title="JMeter Results Analyzer",
    page_icon="📊",
    layout="wide"
)


def extract_date_from_filename(filename):
    """Wyciąganie daty z nazwy pliku w formacie YYYYMMDD_HHMM"""
    try:
        # Wzorzec dla daty w formacie YYYYMMDD_HHMM
        pattern = r'(\d{8})_(\d{4})'
        match = re.search(pattern, filename)

        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMM

            # Konwersja do formatu datetime
            datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}"
            return pd.to_datetime(datetime_str)
        else:
            # Jeśli nie znajdziemy wzorca, zwróć None
            return None
    except Exception as e:
        st.error(f"Błąd podczas parsowania daty z pliku {filename}: {e}")
        return None


def categorize_request_type(label):
    """Kategoryzowanie requestów na typy: API (zaczyna się od '/') vs Proces (bez '/')"""
    return "API Request" if label.startswith('/') else "Process"


def filter_requests_by_type(df, include_api=True, include_processes=True, label_column='label'):
    """Filtrowanie requestów według typu"""
    if not include_api and not include_processes:
        return pd.DataFrame()  # Zwróć pusty DataFrame jeśli oba typy są wyłączone

    df_filtered = df.copy()

    if include_api and include_processes:
        return df_filtered  # Zwróć wszystkie requesty
    elif include_api and not include_processes:
        return df_filtered[df_filtered[label_column].str.startswith('/')]
    elif not include_api and include_processes:
        return df_filtered[~df_filtered[label_column].str.startswith('/')]

    return df_filtered


def get_request_type_counts(df_list, label_column='label'):
    """Zliczanie requestów według typu"""
    if not df_list:
        return 0, 0

    combined_df = pd.concat(df_list, ignore_index=True)
    unique_labels = combined_df[label_column].unique()

    api_count = sum(1 for label in unique_labels if label.startswith('/'))
    process_count = sum(1 for label in unique_labels if not label.startswith('/'))

    return api_count, process_count


def detect_file_type(df):
    """Automatyczne wykrywanie typu pliku na podstawie kolumn"""
    df_columns = set(df.columns.str.strip().str.lower())

    # Sprawdzenie dla plików ogólnych
    general_indicators = {'timestamp', 'elapsed', 'label', 'responsecode'}
    if general_indicators.issubset(df_columns):
        return 'general'

    # Sprawdzenie dla plików aggregate
    aggregate_indicators = {'label', '# samples', 'average', 'median'}
    if aggregate_indicators.issubset(df_columns):
        return 'aggregate'

    return 'unknown'


def load_and_process_files(uploaded_files):
    """Ładowanie i przetwarzanie plików CSV"""
    general_files = []
    aggregate_files = []

    for uploaded_file in uploaded_files:
        try:
            # Próbujemy różne separatory
            separators = [';', ',', '\t']
            df = None

            for sep in separators:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if len(df.columns) > 5:  # Jeśli mamy więcej niż 5 kolumn, prawdopodobnie separator jest dobry
                        break
                except:
                    continue

            if df is None or len(df.columns) <= 1:
                st.error(f"Nie udało się wczytać pliku: {uploaded_file.name}")
                continue

            # Czyszczenie nazw kolumn
            df.columns = df.columns.str.strip()

            file_type = detect_file_type(df)

            if file_type == 'general':
                # Przetwarzanie pliku ogólnego
                if 'timeStamp' in df.columns:
                    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms', errors='coerce')
                df['file_name'] = uploaded_file.name
                # Dodaj kategorię typu requestu
                df['request_type'] = df['label'].apply(categorize_request_type)
                general_files.append(df)

            elif file_type == 'aggregate':
                # Przetwarzanie pliku aggregate
                df['file_name'] = uploaded_file.name

                # NOWA FUNKCJONALNOŚĆ: Wyciąganie daty z nazwy pliku
                extracted_date = extract_date_from_filename(uploaded_file.name)
                if extracted_date is not None:
                    df['test_date'] = extracted_date
                    df['test_date_str'] = extracted_date.strftime('%Y-%m-%d %H:%M')
                else:
                    # Fallback - użyj nazwy pliku lub aktualnej daty
                    df['test_date'] = pd.Timestamp.now()
                    df['test_date_str'] = uploaded_file.name
                    st.warning(f"Nie udało się wyciągnąć daty z nazwy pliku: {uploaded_file.name}. Używam nazwy pliku.")

                # Dodaj kategorię typu requestu dla plików aggregate
                df['request_type'] = df['Label'].apply(categorize_request_type)
                aggregate_files.append(df)

            else:
                st.warning(f"Nierozpoznany format pliku: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Błąd podczas przetwarzania pliku {uploaded_file.name}: {str(e)}")

    return general_files, aggregate_files


def plot_general_data(df_list, selected_labels=None, chart_type="Response Times", skip_empty_dates=False,
                      include_api=True, include_processes=True):
    """Tworzenie wykresów dla danych ogólnych"""
    if not df_list:
        return None

    # Łączenie wszystkich plików
    combined_df = pd.concat(df_list, ignore_index=True)

    # Filtrowanie według typu requestów
    combined_df = filter_requests_by_type(combined_df, include_api, include_processes, 'label')

    # Filtrowanie po labelach
    if selected_labels and len(selected_labels) > 0:
        combined_df = combined_df[combined_df['label'].isin(selected_labels)]

    # Usuwanie wierszy z pustymi timestampami
    combined_df = combined_df.dropna(subset=['timeStamp'])

    if combined_df.empty:
        st.warning("Brak danych do wyświetlenia po zastosowaniu filtrów.")
        return None

    if chart_type == "Response Times":
        if skip_empty_dates:
            # Grupowanie co minutę i liczenie średniej dla lepszej czytelności wykresu liniowego
            combined_df['minute'] = combined_df['timeStamp'].dt.floor('1min')
            response_df = combined_df.groupby(['minute', 'label'])['elapsed'].mean().reset_index()

            fig = px.line(
                response_df,
                x='minute',
                y='elapsed',
                color='label',
                title='Czasy odpowiedzi w czasie (średnia co minutę)',
                labels={'elapsed': 'Czas odpowiedzi (ms)', 'minute': 'Czas'},
                markers=True
            )
        else:
            # Wykres liniowy z wszystkimi punktami, ale z połączonymi liniami
            combined_df_sorted = combined_df.sort_values(['label', 'timeStamp'])
            fig = px.line(
                combined_df_sorted,
                x='timeStamp',
                y='elapsed',
                color='label',
                title='Czasy odpowiedzi w czasie',
                labels={'elapsed': 'Czas odpowiedzi (ms)', 'timeStamp': 'Czas'},
                markers=True
            )

    elif chart_type == "Throughput":
        # Grupowanie co minutę dla throughput
        combined_df['minute'] = combined_df['timeStamp'].dt.floor('1min')
        throughput_df = combined_df.groupby(['minute', 'label']).size().reset_index(name='count')

        if skip_empty_dates:
            # Usuwanie minut bez danych
            throughput_df = throughput_df[throughput_df['count'] > 0]

        fig = px.line(
            throughput_df,
            x='minute',
            y='count',
            color='label',
            title='Throughput (żądania/minutę)',
            labels={'count': 'Liczba żądań', 'minute': 'Czas'},
            markers=True
        )

    elif chart_type == "Success Rate":
        # Grupowanie co minutę dla success rate
        combined_df['minute'] = combined_df['timeStamp'].dt.floor('1min')
        success_df = combined_df.groupby(['minute', 'label']).agg({
            'success': ['count', 'sum']
        }).reset_index()
        success_df.columns = ['minute', 'label', 'total', 'successful']
        success_df['success_rate'] = (success_df['successful'] / success_df['total']) * 100

        if skip_empty_dates:
            # Usuwanie minut bez danych
            success_df = success_df[success_df['total'] > 0]

        fig = px.line(
            success_df,
            x='minute',
            y='success_rate',
            color='label',
            title='Wskaźnik sukcesu (%)',
            labels={'success_rate': 'Wskaźnik sukcesu (%)', 'minute': 'Czas'},
            markers=True
        )

    # Dostosowanie layoutu - legenda pod wykresem i większa szerokość
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=600,  # Zwiększona wysokość
        margin=dict(b=100)  # Dodatkowy margines na dole dla legendy
    )

    return fig


def plot_aggregate_data(df_list, selected_labels=None, metric="Average", chart_style="Po requestach",
                        include_api=True, include_processes=True):
    """Tworzenie wykresów dla danych aggregate - zmodyfikowana wersja"""
    if not df_list:
        return None

    # Łączenie wszystkich plików
    combined_df = pd.concat(df_list, ignore_index=True)

    # Filtrowanie według typu requestów
    combined_df = filter_requests_by_type(combined_df, include_api, include_processes, 'Label')

    # Filtrowanie po labelach
    if selected_labels and len(selected_labels) > 0:
        combined_df = combined_df[combined_df['Label'].isin(selected_labels)]

    if combined_df.empty:
        st.warning("Brak danych do wyświetlenia po zastosowaniu filtrów.")
        return None

    # Sortowanie po dacie dla lepszej czytelności
    combined_df = combined_df.sort_values('test_date')

    if chart_style == "Po requestach":
        # ZMODYFIKOWANY WYKRES: używamy test_date_str jako oś X zamiast Label
        fig = px.line(
            combined_df,
            x='test_date_str',  # Używamy daty zamiast Label
            y=metric,
            color='Label',  # Label staje się kolorami (różne requesty)
            title=f'{metric} w czasie dla różnych requestów',
            labels={metric: f'{metric} (ms)', 'test_date_str': 'Data testu'},
            markers=True
        )
    else:  # "Po plikach"
        # Grupowanie po dacie i pliku (średnia dla requestów w tym samym pliku)
        grouped_df = combined_df.groupby(['test_date_str', 'file_name'])[metric].mean().reset_index()

        fig = px.line(
            grouped_df,
            x='test_date_str',
            y=metric,
            color='file_name',  # Różne pliki jako różne linie
            title=f'Średnia {metric} w czasie (zagregowane po plikach)',
            labels={metric: f'Średnia {metric} (ms)', 'test_date_str': 'Data testu'},
            markers=True
        )

    # Dostosowanie layoutu
    fig.update_layout(
        xaxis_tickangle=45,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=600,
        margin=dict(b=120)  # Więcej miejsca na obrócone etykiety
    )

    return fig


def create_summary_stats(df_list, file_type, include_api=True, include_processes=True):
    """Tworzenie statystyk podsumowujących"""
    if not df_list:
        return None

    combined_df = pd.concat(df_list, ignore_index=True)

    if file_type == 'general':
        # Filtrowanie według typu requestów
        combined_df = filter_requests_by_type(combined_df, include_api, include_processes, 'label')

        if combined_df.empty:
            return None

        stats = combined_df.groupby(['label', 'request_type']).agg({
            'elapsed': ['mean', 'median', 'min', 'max', 'std'],
            'success': 'mean',
            'label': 'count'
        }).round(2)
        stats.columns = ['Średnia (ms)', 'Mediana (ms)', 'Min (ms)', 'Max (ms)', 'Odch. std', 'Success Rate',
                         'Liczba próbek']

    else:  # aggregate
        # Filtrowanie według typu requestów
        combined_df = filter_requests_by_type(combined_df, include_api, include_processes, 'Label')

        if combined_df.empty:
            return None

        stats = combined_df[
            ['Label', 'request_type', 'Average', 'Median', '90% Line', '95% Line', '99% Line', 'Error %', 'Throughput',
             'test_date_str']].round(2)

    return stats


# Główna aplikacja
st.title("📊 JMeter Results Analyzer")
st.markdown("Aplikacja do analizy wyników testów wydajnościowych JMeter")

# Sidebar do ładowania plików
st.sidebar.header("⚙️ Konfiguracja")

uploaded_files = st.sidebar.file_uploader(
    "Wybierz pliki CSV z wynikami JMeter",
    type=['csv'],
    accept_multiple_files=True,
    help="Możesz załadować zarówno pliki ogólne (z timestampami) jak i pliki aggregate"
)

if uploaded_files:
    # Przetwarzanie plików
    with st.spinner("Przetwarzanie plików..."):
        general_files, aggregate_files = load_and_process_files(uploaded_files)

    # Informacje o załadowanych plikach i typach requestów
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pliki ogólne", len(general_files))
    with col2:
        st.metric("Pliki aggregate", len(aggregate_files))

    # Statystyki typów requestów
    if general_files:
        api_count_gen, process_count_gen = get_request_type_counts(general_files, 'label')
        with col3:
            st.metric("API Requests (ogólne)", api_count_gen)
        with col4:
            st.metric("Procesy (ogólne)", process_count_gen)

    if aggregate_files:
        api_count_agg, process_count_agg = get_request_type_counts(aggregate_files, 'Label')
        col5, col6 = st.columns(2)
        with col5:
            st.metric("API Requests (aggregate)", api_count_agg)
        with col6:
            st.metric("Procesy (aggregate)", process_count_agg)

    # Globalne filtry typów requestów w sidebar
    st.sidebar.markdown("### 🎯 Filtry typów requestów")
    st.sidebar.markdown("**API Requests**: zaczynają się od '/'")
    st.sidebar.markdown("**Procesy**: bez prefiksu '/'")

    include_api = st.sidebar.checkbox("Pokaż API Requests", value=True, help="Requesty zaczynające się od '/'")
    include_processes = st.sidebar.checkbox("Pokaż Procesy", value=True, help="Requesty niezaczynające się od '/'")

    # Wyświetlenie informacji o znalezionych datach w plikach aggregate
    if aggregate_files:
        st.sidebar.markdown("### 📅 Znalezione daty w plikach:")
        for df in aggregate_files:
            file_name = df['file_name'].iloc[0]
            test_date = df['test_date_str'].iloc[0]
            st.sidebar.text(f"📄 {file_name}")
            st.sidebar.text(f"📅 {test_date}")
            st.sidebar.markdown("---")

    # Tabs dla różnych typów analiz
    tab1, tab2, tab3 = st.tabs(["📈 Analiza czasowa", "📊 Analiza aggregate", "📋 Statystyki"])

    with tab1:
        if general_files:
            st.header("Analiza danych czasowych")

            # Pobieranie wszystkich unikalnych labeli z uwzględnieniem filtrów
            all_labels = []
            for df in general_files:
                filtered_df = filter_requests_by_type(df, include_api, include_processes, 'label')
                all_labels.extend(filtered_df['label'].unique().tolist())
            unique_labels = sorted(list(set(all_labels)))

            if not unique_labels:
                st.warning("Brak requestów spełniających wybrane kryteria filtrowania.")
            else:
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Wyszukiwanie requestów
                    search_term = st.text_input("🔍 Szukaj requestów:", placeholder="Wpisz nazwę requestu...")
                    if search_term:
                        filtered_labels = [label for label in unique_labels if search_term.lower() in label.lower()]
                    else:
                        filtered_labels = unique_labels

                with col2:
                    # Dropdown do wyboru requestów
                    selected_labels = st.multiselect(
                        "Wybierz requesty:",
                        filtered_labels,
                        default=filtered_labels[:5] if len(filtered_labels) > 5 else filtered_labels
                    )

                # Wybór typu wykresu i opcji
                col_chart, col_options = st.columns([2, 1])

                with col_chart:
                    chart_type = st.selectbox(
                        "Typ wykresu:",
                        ["Response Times", "Throughput", "Success Rate"]
                    )

                with col_options:
                    skip_empty_dates = st.checkbox(
                        "Pomiń puste daty",
                        value=True,
                        help="Usuwa z wykresu okresy bez danych, co czyni wykres bardziej czytelnym"
                    )

                # Tworzenie wykresu
                fig = plot_general_data(general_files, selected_labels, chart_type, skip_empty_dates,
                                        include_api, include_processes)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak plików ogólnych do analizy czasowej")

    with tab2:
        if aggregate_files:
            st.header("Analiza danych zagregowanych")

            # Pobieranie wszystkich unikalnych labeli z plików aggregate z uwzględnieniem filtrów
            all_aggregate_labels = []
            for df in aggregate_files:
                filtered_df = filter_requests_by_type(df, include_api, include_processes, 'Label')
                all_aggregate_labels.extend(filtered_df['Label'].unique().tolist())
            unique_aggregate_labels = sorted(list(set(all_aggregate_labels)))

            if not unique_aggregate_labels:
                st.warning("Brak requestów spełniających wybrane kryteria filtrowania.")
            else:
                # Organizacja kontrolek w 4 kolumnach
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Wyszukiwanie requestów
                    search_term_agg = st.text_input("🔍 Szukaj requestów (aggregate):",
                                                    placeholder="Wpisz nazwę requestu...")
                    if search_term_agg:
                        filtered_agg_labels = [label for label in unique_aggregate_labels if
                                               search_term_agg.lower() in label.lower()]
                    else:
                        filtered_agg_labels = unique_aggregate_labels

                with col2:
                    # Dropdown do wyboru requestów
                    selected_agg_labels = st.multiselect(
                        "Wybierz requesty (aggregate):",
                        filtered_agg_labels,
                        default=filtered_agg_labels[:10] if len(filtered_agg_labels) > 10 else filtered_agg_labels
                    )

                with col3:
                    # Wybór metryki
                    metric = st.selectbox(
                        "Metryka:",
                        ["Average", "Median", "90% Line", "95% Line", "99% Line", "Min", "Max", "Throughput", "Error %"]
                    )

                with col4:
                    # Wybór stylu wykresu
                    chart_style = st.selectbox(
                        "Styl wykresu:",
                        ["Po requestach", "Po plikach"],
                        help="Po requestach: różne requesty jako różne linie\nPo plikach: różne pliki jako różne linie"
                    )

                # Tworzenie wykresu
                fig_agg = plot_aggregate_data(aggregate_files, selected_agg_labels, metric, chart_style,
                                              include_api, include_processes)
                if fig_agg:
                    st.plotly_chart(fig_agg, use_container_width=True)

                # Dodatkowa informacja o wykresie
                if chart_style == "Po requestach":
                    st.info("💡 Wykres pokazuje jak różne requesty zachowują się w czasie. Oś X to daty z nazw plików.")
                else:
                    st.info("💡 Wykres pokazuje średnią metrykę dla wszystkich requestów w każdym pliku w czasie.")

        else:
            st.info("Brak plików aggregate do analizy")

    with tab3:
        st.header("Statystyki podsumowujące")

        if general_files:
            st.subheader("📊 Statystyki plików ogólnych")
            general_stats = create_summary_stats(general_files, 'general', include_api, include_processes)
            if general_stats is not None:
                st.dataframe(general_stats, use_container_width=True)
            else:
                st.warning("Brak danych spełniających wybrane kryteria filtrowania.")

        if aggregate_files:
            st.subheader("📊 Statystyki plików aggregate")
            agg_stats = create_summary_stats(aggregate_files, 'aggregate', include_api, include_processes)
            if agg_stats is not None:
                st.dataframe(agg_stats, use_container_width=True)

                # Dodatkowe podsumowanie dat
                st.subheader("📅 Podsumowanie dat testów")
                date_summary = pd.concat(aggregate_files)[['file_name', 'test_date_str']].drop_duplicates().sort_values(
                    'test_date_str')
                st.dataframe(date_summary, use_container_width=True)
            else:
                st.warning("Brak danych spełniających wybrane kryteria filtrowania.")

    # Dodatkowa informacja o aktualnych filtrach
    if not include_api or not include_processes:
        filter_info = []
        if not include_api:
            filter_info.append("API Requests wyłączone")
        if not include_processes:
            filter_info.append("Procesy wyłączone")

        st.info(f"🎯 Aktywne filtry: {', '.join(filter_info)}")

else:
    st.info("👆 Załaduj pliki CSV w panelu bocznym, aby rozpocząć analizę")

    st.markdown("""
    ### Funkcje aplikacji:

    **🔄 Automatyczne rozpoznawanie formatów:**
    - Pliki ogólne JMeter (z timestampami)
    - Pliki aggregate JMeter (z podsumowaniami)
    - **NOWOŚĆ**: Automatyczne wyciąganie dat z nazw plików aggregate (format: YYYYMMDD_HHMM)
    - **NOWOŚĆ**: Automatyczna kategoryzacja requestów (API vs Procesy)

    **🎯 Filtrowanie typów requestów:**
    - **API Requests**: requesty zaczynające się od "/" (np. /api/users, /login)
    - **Procesy**: requesty bez prefiksu "/" (np. Login Process, Data Processing)
    - Checkbox'y w panelu bocznym do włączania/wyłączania typów
    - Automatyczne liczenie i wyświetlanie ilości każdego typu

    **📈 Wykresy interaktywne:**
    - Czasy odpowiedzi w czasie (wykres liniowy)
    - Throughput (żądania/minutę)
    - Wskaźnik sukcesu
    - Porównania metryk aggregate w czasie (wykres liniowy z datami)
    - Wszystkie wykresy respektują filtry typów requestów

    **🔍 Filtrowanie:**
    - Wyszukiwanie requestów po nazwie
    - Lista rozwijana z requestami (przefiltrowanymi według typu)
    - Wybór wielu requestów jednocześnie
    - Opcja pomijania pustych dat
    - Wybór stylu wykresu (po requestach lub po plikach)

    **📊 Statystyki:**
    - Automatyczne podsumowania z podziałem na typy requestów
    - Metryki wydajności
    - Porównania między plikami
    - Podsumowanie dat testów
    - Wszystkie statystyki respektują filtry typów

    **📅 Obsługiwane formaty nazw plików:**
    - `aggregate_2025_07_v1.jmx_20250701_1437` → 2025-07-01 14:37
    - `test_20241225_2359` → 2024-12-25 23:59
    - Każda nazwa zawierająca wzorzec `YYYYMMDD_HHMM`
    """)