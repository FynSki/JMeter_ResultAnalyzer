# JMeter Performance Analyzer

Aplikacja webowa do analizy i porównywania plików CSV z Apache JMeter.

## Uruchomienie

```bash
pip install -r requirements.txt
python run.py
```

Następnie otwórz przeglądarkę pod adresem: **http://localhost:5050**

## Funkcje

### 📊 Analiza pojedynczego przebiegu
- Łączna liczba żądań, throughput (req/s)
- Mediana, średnia, P95 czasu odpowiedzi
- Wskaźnik błędów
- **Wykres: liczba żądań per minuta**
- **Wykres: mediana czasu odpowiedzi per minuta**
- Tabela z pełnymi danymi per minuta

### ⚖️ Porównanie żądań między przebiegami
- Wybór pliku bazowego (Baseline) i porównywanego (Compare)
- Filtrowanie po konkretnym żądaniu (Label)
- Statystyki: count, avg, median, P90, P95, P99, min, max, error rate
- **Oznaczenie zmian: 🟢 lepsza / 🔴 gorsza wydajność**
- **Wykres porównania liczby żądań per minuta**
- **Wykres porównania średniego czasu per minuta**

## Obsługiwane kolumny CSV (JMeter default)

| Kolumna JMeter | Opis |
|---|---|
| `timeStamp` | Unix timestamp (ms lub s) |
| `elapsed` | Czas odpowiedzi (ms) |
| `label` | Nazwa samplera/żądania |
| `responseCode` | Kod HTTP |
| `success` | Czy żądanie się powiodło |
| `bytes` | Rozmiar odpowiedzi |
| `threadName` | Nazwa wątku |

## Struktura projektu

```
jmeter_analyzer/
├── app.py          # Backend Flask (API)
├── run.py          # Uruchamiacz
├── requirements.txt
├── static/
│   └── index.html  # Frontend (JS + Chart.js)
└── uploads/        # Wgrane pliki CSV (tworzone automatycznie)
```
