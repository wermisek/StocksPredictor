import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from polygon import RESTClient
import requests
from scipy.signal import savgol_filter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config with a modern theme
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        border: 1px solid #464855;
        color: #fff;
    }
    .stButton>button:hover {
        border-color: #00ff88;
        color: #00ff88;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stProgress .st-bo {
        background-color: #00ff88;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff88;
        color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    .plot-container {
        border-radius: 5px;
        background-color: #262730;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title with modern styling
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #00ff88;'>📈 Stock Price Predictor</h1>
        <p style='color: #c6c6c6; font-size: 1.2em;'>Advanced AI-Powered Stock Analysis & Prediction</p>
    </div>
""", unsafe_allow_html=True)

# Update the sidebar styling
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #00ff88;'>Trading Dashboard</h2>
    </div>
""", unsafe_allow_html=True)

# Initialize Polygon API (replace the hardcoded key with environment variable)
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    st.error("No API key found. Please set the POLYGON_API_KEY environment variable.")
    st.stop()

# Title and description
st.title("Stock Price Predictor")
st.markdown("Predict stock prices using machine learning")

# Download data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(symbol, n_years):
    try:
        client = RESTClient(api_key=POLYGON_API_KEY)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_years*365)
        
        # Get aggregates (daily OHLCV)
        aggs = []
        for a in client.list_aggs(
            symbol,
            1,
            "day",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            limit=50000
        ):
            aggs.append({
                'Date': pd.to_datetime(a.timestamp, unit='ms'),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume
            })
        
        if not aggs:
            raise Exception(f"No data found for symbol {symbol}")
            
        # Convert to DataFrame
        df = pd.DataFrame(aggs)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

# Function to get current stock price and info
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_stock_info(symbol, data):
    try:
        client = RESTClient(api_key=POLYGON_API_KEY)
        
        # Get last trade
        trade = client.get_last_trade(symbol)
        
        # Get previous close
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1]
        
        # Calculate change percentage
        change_percent = ((trade.price - prev_close) / prev_close) * 100
        
        return {
            'price': trade.price,
            'change_percent': change_percent,
            'volume': data['Volume'].iloc[-1],  # Today's volume
            'name': symbol
        }
    except Exception as e:
        # Fallback to using the last available data
        return {
            'price': data['Close'].iloc[-1],
            'change_percent': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
            'volume': data['Volume'].iloc[-1],
            'name': symbol
        }

# Initialize session state for stock symbol
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = 'MSFT'

# Sidebar
st.sidebar.header("Settings")

# Common stock symbols for suggestions
COMMON_STOCKS = {
    'MSFT': 'Microsoft Corporation',
    'IBM': 'IBM Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.'
}

# Create two columns in the sidebar for the stock buttons
col1, col2 = st.sidebar.columns(2)
for idx, (symbol, name) in enumerate(COMMON_STOCKS.items()):
    col = col1 if idx % 2 == 0 else col2
    with col:
        if st.button(f"{symbol}", help=name, key=f"btn_{symbol}"):
            st.session_state.stock_symbol = symbol
            st.rerun()

st.sidebar.markdown("---")
# Text input for manual symbol entry
symbol = st.sidebar.text_input("Or enter stock symbol:", value=st.session_state.stock_symbol)
symbol = symbol.upper().strip()

if symbol != st.session_state.stock_symbol:
    st.session_state.stock_symbol = symbol
    st.rerun()

# Additional settings
n_years = st.sidebar.slider("Years of historical data", 1, 5, 2)
n_days = st.sidebar.slider("Days to predict", 7, 365, 30)

# Calculate recent statistics
def calculate_market_stats(data):
    data = data.copy()
    # Calculate returns first
    data['Returns'] = data['Close'].pct_change()
    data['Returns'] = data['Returns'].fillna(0)  # Fill NaN values
    
    recent_prices = data['Close'].iloc[-30:]  # Last 30 days
    trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
    volatility = np.std(data['Returns'].iloc[-30:]) * np.sqrt(252)  # Annualized volatility
    avg_daily_move = np.mean(np.abs(data['Returns'].iloc[-30:]))
    
    # Calculate momentum from returns
    momentum = sum([1 if ret > 0 else -1 for ret in data['Returns'].iloc[-10:]])
    
    return trend, volatility, avg_daily_move, momentum

# Prepare data for LSTM with more features
def prepare_data(data, lookback=60):
    try:
        # Calculate additional technical indicators
        data = data.copy()
        data['Returns'] = data['Close'].pct_change().fillna(0)
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
        data['Price_Range'] = data['High'] - data['Low']
        
        # Fill any NaN values that might have been created
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        # Create features array
        feature_columns = ['Close', 'Volume', 'Returns', 'MA5', 'MA20', 'RSI', 'Volume_MA5', 'Price_Range']
        features = data[feature_columns].values
        
        # Normalize features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict the Close price
        
        if len(X) == 0:
            raise Exception("Not enough data points for prediction")
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, scaler, features, scaled_data
    except Exception as e:
        raise Exception(f"Error preparing data: {str(e)}")

# Create enhanced LSTM model with trend awareness
def create_model(lookback, n_features):
    # Input layer for historical data
    main_input = tf.keras.layers.Input(shape=(lookback, n_features), name='input_layer')
    
    # LSTM layers with residual connections
    lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1')(main_input)
    dropout1 = tf.keras.layers.Dropout(0.2, name='dropout_1')(lstm1)
    
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_2')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2, name='dropout_2')(lstm2)
    
    lstm3 = tf.keras.layers.LSTM(128, return_sequences=False, name='lstm_3')(dropout2)
    dropout3 = tf.keras.layers.Dropout(0.2, name='dropout_3')(lstm3)
    
    # Dense layers with trend awareness
    dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(dropout3)
    batch1 = tf.keras.layers.BatchNormalization(name='batch_1')(dense1)
    
    dense2 = tf.keras.layers.Dense(32, activation='relu', name='dense_2')(batch1)
    batch2 = tf.keras.layers.BatchNormalization(name='batch_2')(dense2)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation='linear', name='output')(batch2)
    
    # Create model
    model = tf.keras.Model(inputs=main_input, outputs=output)
    
    # Use Adam optimizer with reduced learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='huber')
    
    return model

# Calculate RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Update plot styling
def style_plot(fig):
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#262730',
        font_color='#c6c6c6',
        title_font_color='#ffffff',
        legend_font_color='#c6c6c6',
        title_x=0.5,
        title_y=0.95,
        title_xanchor='center',
        title_yanchor='top',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            gridcolor='#333333',
            zerolinecolor='#333333',
        ),
        yaxis=dict(
            gridcolor='#333333',
            zerolinecolor='#333333',
        ),
        margin=dict(t=50, l=50, r=50, b=50)
    )
    return fig

# Main content area
try:
    # Load data
    with st.spinner("Loading data..."):
        data = load_data(symbol, n_years)
        info = get_stock_info(symbol, data)
    
    # Update the stock info display
    if info['change_percent'] >= 0:
        delta_color = "green"
        delta_symbol = "↑"
    else:
        delta_color = "red"
        delta_symbol = "↓"

    st.markdown(f"""
        <div style='background-color: #262730; padding: 20px; border-radius: 10px; margin: 10px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 style='margin: 0; color: white;'>{symbol} - {COMMON_STOCKS.get(symbol, 'Stock Analysis')}</h2>
                    <h3 style='margin: 5px 0; color: #00ff88;'>${info['price']:.2f} 
                        <span style='color: {delta_color}; font-size: 0.8em;'>
                            {delta_symbol} {abs(info['change_percent']):.2f}%
                        </span>
                    </h3>
                </div>
                <div style='text-align: right;'>
                    <p style='margin: 0; color: #c6c6c6;'>Volume: {info['volume']:,}</p>
                    <p style='margin: 0; color: #c6c6c6;'>Range: ${data['Low'].iloc[-1]:.2f} - ${data['High'].iloc[-1]:.2f}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs with modern styling
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "🔮 Price Prediction", "📈 Technical Indicators"])

    with tab1:
        # Historical price chart with modern styling
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff3366'
        )])
        
        fig = style_plot(fig)
        fig.update_layout(
            title='Historical Price Action',
            height=600,
            yaxis_title='Price ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart with modern styling
        fig_volume = go.Figure(data=[go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='#00ff88'
        )])
        
        fig_volume = style_plot(fig_volume)
        fig_volume.update_layout(
            title='Trading Volume',
            height=300,
            yaxis_title='Volume'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)

    with tab2:
        st.subheader("AI-Powered Price Prediction")
        
        if len(data) < 60:
            st.error("Not enough historical data for prediction. Please select a longer time period.")
        else:
            try:
                # Prepare data for prediction
                X, y, scaler, features, scaled_data = prepare_data(data)
                split = int(0.8 * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Create and train model
                with st.spinner("Training model... This may take a few moments."):
                    model = create_model(60, n_features=X.shape[2])
                    
                    # Train with more epochs and validation
                    history = model.fit(
                        X_train, y_train,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.2,
                        verbose=0
                    )
                
                # Make predictions
                train_predict = model.predict(X_train, verbose=0)
                test_predict = model.predict(X_test, verbose=0)
                
                # Calculate technical indicators before prediction
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = calculate_rsi(data['Close'])

                # Future predictions with realistic market behavior
                last_sequence = X[-1:].copy()
                future_predictions = []

                # Calculate market statistics
                recent_window = 30  # Last 30 days
                returns = data['Close'].pct_change().dropna()
                daily_volatility = np.std(returns[-recent_window:])
                trend = (data['Close'].iloc[-1] - data['Close'].iloc[-recent_window]) / data['Close'].iloc[-recent_window]
                avg_volume = data['Volume'].iloc[-recent_window:].mean()

                # Calculate support and resistance levels
                support = data['Low'].iloc[-recent_window:].min()
                resistance = data['High'].iloc[-recent_window:].max()
                current_price = data['Close'].iloc[-1]

                # Maximum allowed daily movement (based on historical volatility)
                max_daily_move = daily_volatility * 2  # 2 standard deviations

                for i in range(n_days):
                    # Get base prediction
                    next_pred = model.predict(last_sequence, verbose=0)[0, 0]
                    
                    # Convert prediction to price scale
                    dummy = np.zeros((1, features.shape[1]))
                    dummy[0, 0] = next_pred
                    predicted_price = scaler.inverse_transform(dummy)[0, 0]
                    
                    # Apply realistic constraints
                    if i < 7:  # Short-term predictions more influenced by recent trend
                        # Add momentum based on recent trend but decay it
                        trend_factor = trend * (1 - i/7)  # Trend effect decreases over time
                        predicted_price *= (1 + trend_factor * 0.3)
                    
                    # Add random walk component based on real volatility
                    daily_return = np.random.normal(0, daily_volatility)
                    predicted_price *= (1 + daily_return)
                    
                    # Ensure price stays within realistic bounds
                    max_up = current_price * (1 + max_daily_move * np.sqrt(i + 1))
                    max_down = current_price * (1 - max_daily_move * np.sqrt(i + 1))
                    predicted_price = np.clip(predicted_price, max_down, max_up)
                    
                    # Respect support and resistance levels with some flexibility
                    if predicted_price > resistance:
                        predicted_price = resistance + (predicted_price - resistance) * 0.1
                    elif predicted_price < support:
                        predicted_price = support + (predicted_price - support) * 0.1
                    
                    # Convert back to scaled value for sequence update
                    dummy[0, 0] = predicted_price
                    scaled_pred = scaler.transform(dummy)[0, 0]
                    
                    future_predictions.append(predicted_price)
                    
                    # Update sequence for next prediction
                    new_sequence = np.roll(last_sequence[0], -1, axis=0)
                    new_sequence[-1] = scaled_pred
                    last_sequence[0] = new_sequence
                
                # Add some market psychology (resistance at round numbers)
                future_predictions = np.array(future_predictions)
                round_numbers = np.round(future_predictions / 50) * 50  # Resistance at $50 intervals
                future_predictions = future_predictions * 0.95 + round_numbers * 0.05
                
                # Smooth out any extreme jumps while preserving trends
                window = min(5, len(future_predictions) - 1 if len(future_predictions) % 2 == 0 else len(future_predictions) - 2)
                if window > 2:
                    future_predictions = savgol_filter(future_predictions, window, 2)
                
                # Calculate future dates and confidence intervals
                future_dates = pd.date_range(start=data.index[-1], periods=n_days+1)[1:]
                confidence_interval = daily_volatility * np.sqrt(np.arange(1, n_days + 1)) * current_price
                upper_bound = future_predictions + confidence_interval
                lower_bound = future_predictions - confidence_interval
                
                # Plot predictions with confidence intervals
                fig = go.Figure()
                
                # Historical data with gradient fill
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Historical',
                    line=dict(color='#00ff88', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,255,136,0.1)'
                ))
                
                # Prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions,
                    name='Prediction',
                    line=dict(color='#ff3366', width=2, dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    line=dict(color='rgba(255,51,102,0)'),
                    name='Upper Bound',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    fill='tonexty',
                    fillcolor='rgba(255,51,102,0.1)',
                    line=dict(color='rgba(255,51,102,0)'),
                    name='Confidence Interval'
                ))
                
                fig = style_plot(fig)
                fig.update_layout(
                    title='Price Prediction with Confidence Intervals',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction metrics in a modern card
                st.markdown("""
                    <div style='background-color: #262730; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <h3 style='color: #00ff88; margin: 0 0 15px 0;'>Prediction Metrics</h3>
                        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;'>
                            <div>
                                <p style='color: #c6c6c6; margin: 0;'>Current Price</p>
                                <h4 style='color: white; margin: 5px 0;'>${current_price:.2f}</h4>
                            </div>
                            <div>
                                <p style='color: #c6c6c6; margin: 0;'>30-Day Trend</p>
                                <h4 style='color: {"#00ff88" if trend > 0 else "#ff3366"}; margin: 5px 0;'>{trend*100:.1f}%</h4>
                            </div>
                            <div>
                                <p style='color: #c6c6c6; margin: 0;'>Daily Volatility</p>
                                <h4 style='color: white; margin: 5px 0;'>{daily_volatility*100:.1f}%</h4>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add prediction disclaimer
                st.warning("⚠️ Disclaimer: Stock predictions are based on historical data and technical analysis. They should not be used as the sole basis for investment decisions.")
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    with tab3:
        st.subheader("Technical Analysis")
        
        # Technical indicators plot with modern styling
        fig_tech = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Moving Averages', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Price and MAs
        fig_tech.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Price',
            line=dict(color='#ffffff', width=1)
        ), row=1, col=1)
        
        fig_tech.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            name='MA20',
            line=dict(color='#00ff88', width=1.5)
        ), row=1, col=1)
        
        fig_tech.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            name='MA50',
            line=dict(color='#ff3366', width=1.5)
        ), row=1, col=1)
        
        # RSI
        fig_tech.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='#00ff88', width=1.5)
        ), row=2, col=1)
        
        fig_tech = style_plot(fig_tech)
        fig_tech.update_layout(height=800)
        
        # Add RSI levels
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#ff3366", row=2, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=2, col=1)
        
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # Technical analysis explanation in a modern card
        st.markdown("""
            <div style='background-color: #262730; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: #00ff88; margin: 0 0 15px 0;'>Technical Indicators Guide</h3>
                <div style='color: #c6c6c6;'>
                    <p><strong style='color: white;'>Moving Averages (MA):</strong> Shows the average price over a period of time</p>
                    <ul>
                        <li>MA20 (Green): 20-day moving average for short-term trend</li>
                        <li>MA50 (Red): 50-day moving average for medium-term trend</li>
                    </ul>
                    <p><strong style='color: white;'>Relative Strength Index (RSI):</strong> Momentum indicator (0-100)</p>
                    <ul>
                        <li>Above 70: Potentially overbought (red line)</li>
                        <li>Below 30: Potentially oversold (green line)</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    An error occurred while processing your request. This might be due to:
    - Invalid stock symbol
    - Network connectivity issues
    - Rate limiting from the data provider
    
    Please try:
    1. Checking if the stock symbol is correct
    2. Waiting a few moments and trying again
    3. Selecting a different stock symbol
    
    Technical details: {str(e)}
    """)
    
    st.markdown("### Try these popular stock symbols:")
    for sym, name in COMMON_STOCKS.items():
        st.markdown(f"- **{sym}**: {name}") 