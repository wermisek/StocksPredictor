# StocksPredictor

Przewidywanie cen akcji z użyciem machine learningu i analizy technicznej.

## Co to robi

Aplikacja pobiera dane giełdowe z Polygon.io, liczy wskaźniki techniczne (RSI, średnie kroczące)
i próbuje przewidzieć przyszłe ceny używając modelu LSTM (sztuczna sieć neuronowa).

Wyniki pokazuje na wykresach w przeglądarce.

## Uruchomienie

```bash
# 1. zainstaluj zależności
pip install -r requirements.txt

# 2. dodaj klucz API do Polygon.io
echo "POLYGON_API_KEY=twoj_klucz" > .env

# 3. uruchom
streamlit run app.py
```

Darmowy klucz API dostaniesz na [polygon.io](https://polygon.io/).

## Uwaga

Model trenuje się od nowa przy każdej predykcji — to trwa kilka-kilkanaście sekund.
Nie zapisuje wytrenowanego modelu między uruchomieniami.

To nie jest porada inwestycyjna. Przewidywania są symulacją, nie faktem.

## Z czego korzysta

| Biblioteka | Do czego |
|---|---|
| streamlit | UI w przeglądarce |
| yfinance | dane giełdowe z Yahoo Finance |
| polygon-api-client | alternatywne źródło danych |
| tensorflow / keras | model LSTM |
| plotly | wykresy |
| pandas + numpy | obróbka danych |
| scikit-learn | skalowanie danych |
| python-dotenv | klucz API z pliku .env |

## Licencja

Brak — project publiczny, używasz na własne ryzyko.
