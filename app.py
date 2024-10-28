# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from googletrans import Translator
import datetime

# Initialize Translator
translator = Translator()

# Streamlit app setup
st.title("Stock Forecaster")
st.write("Forecasting stock prices using linear regression.")

# Language translation option
translate = st.button("Translate to Russian")

# Function to fetch recent daily data from Yahoo Finance
def get_stock_data(symbol):
    try:
        stock_data = yf.Ticker(symbol)
        # Fetch daily data for the last 3 months
        data = stock_data.history(period="3mo", interval="1d")
        data = data.sort_index(ascending=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Input for stock symbol
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
if symbol:
    data = get_stock_data(symbol)
    
    if data is not None:
        # Display last week's trading data (previous 5 trading days)
        last_week_data = data[-5:]  # Last 5 trading days
        # Check if 'Adj Close' is in the data; if not, exclude it
        columns_to_display = ["Open", "Close", "Volume"]
        if "Adj Close" in last_week_data.columns:
            columns_to_display.append("Adj Close")

        st.write("Last Week's Trading Data")
        st.dataframe(last_week_data[columns_to_display])

        # Preprocess data for linear regression
        data["date_num"] = (data.index - data.index[0]).days
        X = data["date_num"].values.reshape(-1, 1)
        y = data["Close"].values  # Using close price for predictions

        # Train linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Forecasting for the next 5 days (one price per day)
        forecast_days = 5
        last_day = X[-1][0]
        forecast_dates = [last_day + i for i in range(1, forecast_days + 1)]
        forecast = model.predict(np.array(forecast_dates).reshape(-1, 1))

        # Display chart of past month
        past_month_data = data[-22:]  # Approx. 22 trading days in a month
        st.line_chart(past_month_data["Close"])

        # Display forecast table for the next week (one price per day)
        forecast_df = pd.DataFrame({
            "Date": [data.index[-1] + pd.Timedelta(days=int(day)) for day in range(1, forecast_days + 1)],
            "Predicted Price": forecast
        })
        st.write("Next Week's Forecast (Daily Predictions)")
        st.table(forecast_df)

        # Generate Buy/Sell/Hold Recommendation
        if forecast[-1] > y[-1]:
            recommendation = "Buy"
        elif forecast[-1] < y[-1]:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        st.subheader(f"Recommendation: {recommendation}")

        # Translation to Russian
        if translate:
            st.subheader("Russian Translation")
            st.write(translator.translate("Stock Forecaster", dest="ru").text)
            st.write(translator.translate("Forecasting stock prices using linear regression.", dest="ru").text)
            st.write(translator.translate(f"Recommendation: {recommendation}", dest="ru").text)
    else:
        st.warning("No data available for the entered symbol.")
