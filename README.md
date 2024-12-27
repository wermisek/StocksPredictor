# Stock Price Predictor

An AI-powered stock price prediction application using machine learning and technical analysis.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StocksPredictor.git
cd StocksPredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
   - Get a free API key from [Polygon.io](https://polygon.io/)
   - Create a `.env` file from the example:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API key
   - Create a Streamlit secrets file:
     ```bash
     mkdir -p .streamlit
     cp .streamlit/secrets.toml.example .streamlit/secrets.toml
     ```
   - Edit `.streamlit/secrets.toml` and add your API key

4. Run the application:
```bash
streamlit run app.py
```

## Features

- Real-time stock data from Polygon.io
- Advanced technical analysis
- AI-powered price predictions
- Interactive charts and visualizations
- Technical indicators (RSI, Moving Averages)

## Security Note

This project uses environment variables and Streamlit secrets for API key management. Never commit your actual API keys to GitHub. The `.env` and `.streamlit/secrets.toml` files are included in `.gitignore` to prevent accidental commits. 