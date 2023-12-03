# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger



import numpy as np, pandas as pd, yfinance as yf

ticker= 'ABEV3.SA'

prices = yf.download(ticker,start='2023-01-01', end='2023-11-25', group_by='tickers')['Close']



def mr(prices, mean_lb, std_lb, buy_th, sell_th, stop_loss=None, take_profit=None):

    returns = prices.pct_change()
    daily_returns = np.zeros_like(returns)

    mean = np.mean(prices[:mean_lb])
    std = np.std(prices[:std_lb])

    signal_entered = 0

    for i in range(mean_lb, len(returns)):    # len(returns) - 1  (?)

        z_score = (prices[i] - mean) / std
        signal = -1 if z_score < -buy_th else (1 if z_score > sell_th else 0)

        if signal != 0 and signal_entered == 0:  # Open new position

            signal_entered = signal
            price_entered = prices[i]

        daily_returns[i] = signal_entered * returns[i]


        if stop_loss is not None and signal_entered != 0:

          if signal_entered == 1 and prices[i] < (price_entered * (1 - stop_loss)):
            daily_returns[i] = ((price_entered * (1 - stop_loss)) / prices[i - 1]) - 1    # Assume que vendeu no stop loss
            signal_entered = 0

          elif signal_entered == -1 and prices[i] > (price_entered * (1 + stop_loss)):
            daily_returns[i] =  - (((price_entered * (1 + stop_loss)) / prices[i - 1]) - 1)    # Assume que recomprou no stop loss
            signal_entered = 0

        if take_profit is not None and signal_entered != 0:

          if signal_entered == 1 and prices[i] > (price_entered * (1 + take_profit)):
            daily_returns[i] = ((price_entered * (1 + take_profit)) / prices[i - 1]) - 1    # Assume que vendeu no take profit
            signal_entered = 0

          elif signal_entered == -1 and prices[i] < (price_entered * (1 - stop_loss)):
            daily_returns[i] =  ((price_entered * (1 - take_profit)) / prices[i - 1]) - 1    # Assume que recomprou no take profit
            signal_entered = 0


        mean = np.mean(prices[i - mean_lb + 1 : i + 1])
        std = np.std(prices[i - std_lb + 1 : i + 1])
    
    
    portfolio_value = np.cumprod(1 + daily_returns)
    returns_series = pd.Series(portfolio_value, index=returns.index)

    return returns_series








dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
time_series_data = np.random.normal(0.001, 0.01, len(dates))

df = pd.DataFrame({'Date': dates, 'Time Series Data': time_series_data})
df.set_index('Date', inplace=True)

# Streamlit App
st.title('Time Series Plotting App')

# Sidebar for user input
st.sidebar.header('User Input')
selected_column = st.sidebar.selectbox('Select Time Series Column', df.columns)
date_range = st.sidebar.date_input('Select Date Range', [df.index.min(), df.index.max()])

# Filter data based on user input
result_series = df.loc[date_range[0]:date_range[1], [selected_column]]

# Plotting
st.line_chart(result_series)

# Show DataFrame (optional)
st.write('Filtered Data:')

st.write("#caio Streamlit! 👋")
