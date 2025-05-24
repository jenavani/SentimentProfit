import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Crypto Trading Dashboard")

# --- 1. Data Loading (Cached for performance) ---
@st.cache_data
def load_data():
    try:
        df_sentiment = pd.read_csv('fear_greed_index.csv')
        df_trader = pd.read_csv('historical_data.csv')

        # Preprocessing for Sentiment Data
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        df_sentiment = df_sentiment.sort_values('date').reset_index(drop=True)
        # Map sentiment classification to numerical (0 for Fear, 1 for Greed)
        df_sentiment['Sentiment_Numerical'] = df_sentiment['classification'].apply(lambda x: 1 if x == 'Greed' else 0)

        # Preprocessing for Trader Data
        df_trader['Timestamp IST'] = pd.to_datetime(df_trader['Timestamp IST']) # Corrected column name
        df_trader['Date'] = df_trader['Timestamp IST'].dt.date
        df_trader['Date'] = pd.to_datetime(df_trader['Date']) # Convert back to datetime object for merging

        # Ensure numerical columns are truly numerical, coerce errors to NaN
        # Corrected column names based on your df_trader.info()
        for col in ['Execution Price', 'Size Tokens', 'Start Position', 'Closed PnL']:
            df_trader[col] = pd.to_numeric(df_trader[col], errors='coerce')

        # Drop rows where critical numerical data might be missing after coercion
        df_trader.dropna(subset=['Execution Price', 'Size Tokens', 'Closed PnL'], inplace=True)

        return df_sentiment, df_trader
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}. Please ensure 'fear_greed_index.csv' and 'historical_data.csv' are in the same directory.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        st.stop()

# --- 2. Feature Engineering (Cached for performance) ---
@st.cache_data
def engineer_features(df_trader, df_sentiment):
    # Calculate daily PnL, num_trades, total_size for each account
    # Corrected column names for grouping and aggregation
    daily_trader_performance = df_trader.groupby(['Account', 'Date']).agg(
        daily_PnL=('Closed PnL', 'sum'),
        num_trades=('Closed PnL', 'count'),
        total_size=('Size Tokens', 'sum') # Corrected column name
    ).reset_index()

    # Calculate win rate for each account per day
    df_trader['is_win'] = df_trader['Closed PnL'] > 0 # Corrected column name
    daily_wins = df_trader.groupby(['Account', 'Date'])['is_win'].sum().reset_index()
    daily_trades_count = df_trader.groupby(['Account', 'Date'])['Closed PnL'].count().reset_index(name='total_trades_day')

    daily_trader_performance = pd.merge(daily_trader_performance, daily_wins, on=['Account', 'Date'], how='left')
    daily_trader_performance = pd.merge(daily_trader_performance, daily_trades_count, on=['Account', 'Date'], how='left')
    daily_trader_performance['win_rate'] = (daily_trader_performance['is_win'] / daily_trader_performance['total_trades_day']).fillna(0)

    # Merge daily trader performance with sentiment data on 'Date'
    merged_df = pd.merge(daily_trader_performance, df_sentiment[['date', 'classification', 'Sentiment_Numerical']], left_on='Date', right_on='date', how='left')

    # Drop the duplicate 'date' column and rename for consistency in plotting
    merged_df.drop(columns=['date'], inplace=True)
    merged_df.rename(columns={'classification': 'Classification', 'Account': 'account'}, inplace=True)

    # Drop rows where sentiment data might be missing after merging
    merged_df.dropna(subset=['Classification'], inplace=True)

    return merged_df

# --- Load Data and Engineer Features ---
df_sentiment, df_trader = load_data()

if df_sentiment is None or df_trader is None:
    st.stop() # Stop if data loading failed

merged_df = engineer_features(df_trader.copy(), df_sentiment.copy())

# --- Streamlit Dashboard Title ---
st.title("Crypto Trader Performance & Market Sentiment Analysis")

# --- Filters ---
with st.sidebar:
    st.header("Filters")
    
    # Date range filter
    min_date = merged_df['Date'].min()
    max_date = merged_df['Date'].max()
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    filtered_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)].copy()

    # Account filter
    all_accounts = ['All Accounts'] + sorted(filtered_df['account'].unique().tolist())
    selected_accounts = st.multiselect(
        "Select Trader Accounts",
        options=all_accounts,
        default='All Accounts'
    )
    
    if 'All Accounts' not in selected_accounts:
        filtered_df = filtered_df[filtered_df['account'].isin(selected_accounts)]

    # Sentiment filter
    all_sentiments = ['All Sentiments'] + sorted(filtered_df['Classification'].unique().tolist())
    selected_sentiments = st.multiselect(
        "Select Market Sentiment",
        options=all_sentiments,
        default='All Sentiments'
    )
    
    if 'All Sentiments' not in selected_sentiments:
        filtered_df = filtered_df[filtered_df['Classification'].isin(selected_sentiments)]

# Check if filtered_df is empty
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
    st.stop()

# --- Key Performance Indicators (KPIs) ---
st.header("Key Performance Indicators (KPIs)")

total_pnl = filtered_df['daily_PnL'].sum()
avg_daily_pnl = filtered_df['daily_PnL'].mean()
avg_win_rate = filtered_df['win_rate'].mean()
total_trades = filtered_df['num_trades'].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total PnL", f"${total_pnl:,.2f}")
with col2:
    st.metric("Avg Daily PnL", f"${avg_daily_pnl:,.2f}")
with col3:
    st.metric("Avg Win Rate", f"{avg_win_rate:.2%}")
with col4:
    st.metric("Total Trades", f"{total_trades:,.0f}")

st.markdown("---")

# --- Visualizations ---

# 1. Total Market Daily PnL and Sentiment Over Time
st.header("Total Market Daily PnL and Sentiment Over Time")

# Aggregate daily PnL for the entire market based on filters
market_daily_pnl = filtered_df.groupby('Date')['daily_PnL'].sum().reset_index()
market_daily_pnl = pd.merge(market_daily_pnl, df_sentiment[['date', 'classification', 'Sentiment_Numerical']], left_on='Date', right_on='date', how='left')
market_daily_pnl.drop(columns=['date'], inplace=True)
market_daily_pnl.rename(columns={'classification': 'Classification'}, inplace=True)

# Filter market_daily_pnl by the date range selected
market_daily_pnl = market_daily_pnl[(market_daily_pnl['Date'] >= start_date) & (market_daily_pnl['Date'] <= end_date)]

fig_time_series = go.Figure()

# PnL Line
fig_time_series.add_trace(go.Scatter(
    x=market_daily_pnl['Date'],
    y=market_daily_pnl['daily_PnL'],
    mode='lines',
    name='Total Market Daily PnL',
    line=dict(color='blue')
))

# Sentiment Markers
fig_time_series.add_trace(go.Scatter(
    x=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 1]['Date'],
    y=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 1]['daily_PnL'],
    mode='markers',
    name='Greed Day',
    marker=dict(color='green', size=8, symbol='circle'),
    hoverinfo='text',
    hovertext=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 1].apply(lambda row: f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>PnL: ${row['daily_PnL']:,.2f}<br>Sentiment: {row['Classification']}", axis=1)
))

fig_time_series.add_trace(go.Scatter(
    x=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 0]['Date'],
    y=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 0]['daily_PnL'],
    mode='markers',
    name='Fear Day',
    marker=dict(color='red', size=8, symbol='x'),
    hoverinfo='text',
    hovertext=market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 0].apply(lambda row: f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>PnL: ${row['daily_PnL']:,.2f}<br>Sentiment: {row['Classification']}", axis=1)
))

fig_time_series.update_layout(
    title='Total Market Daily PnL and Market Sentiment Over Time',
    xaxis_title='Date',
    yaxis_title='Total Daily PnL',
    hovermode="x unified",
    legend_title="Sentiment"
)
st.plotly_chart(fig_time_series, use_container_width=True)

st.markdown("---")

# 2. Average Daily PnL & Win Rate by Sentiment
st.header("Average Daily PnL and Win Rate by Market Sentiment")

avg_pnl_by_sentiment = filtered_df.groupby('Classification')['daily_PnL'].mean().reset_index()
avg_winrate_by_sentiment = filtered_df.groupby('Classification')['win_rate'].mean().reset_index()

col_sentiment_pnl, col_sentiment_winrate = st.columns(2)

with col_sentiment_pnl:
    fig_pnl_sentiment = px.bar(
        avg_pnl_by_sentiment,
        x='Classification',
        y='daily_PnL',
        color='Classification',
        color_discrete_map={'Fear': 'darkred', 'Greed': 'darkgreen'},
        title='Average Daily PnL by Market Sentiment'
    )
    fig_pnl_sentiment.update_layout(yaxis_title='Average Daily PnL')
    st.plotly_chart(fig_pnl_sentiment, use_container_width=True)

with col_sentiment_winrate:
    fig_winrate_sentiment = px.bar(
        avg_winrate_by_sentiment,
        x='Classification',
        y='win_rate',
        color='Classification',
        color_discrete_map={'Fear': 'indianred', 'Greed': 'lightgreen'},
        title='Average Daily Win Rate by Market Sentiment'
    )
    fig_winrate_sentiment.update_layout(yaxis_title='Average Daily Win Rate')
    st.plotly_chart(fig_winrate_sentiment, use_container_width=True)

st.markdown("---")

# 3. Performance Distribution by Sentiment (Box Plots)
st.header("Daily PnL Distribution by Market Sentiment")

if not filtered_df.empty:
    fig_boxplot = px.box(
        filtered_df,
        x='Classification',
        y='daily_PnL',
        color='Classification',
        color_discrete_map={'Fear': 'darkred', 'Greed': 'darkgreen'},
        points="outliers",
        title='Distribution of Daily PnL for All Traders by Market Sentiment'
    )
    fig_boxplot.update_layout(yaxis_title='Daily PnL')
    st.plotly_chart(fig_boxplot, use_container_width=True)
else:
    st.info("No data available for PnL distribution plot with current filters.")

st.markdown("---")

# 4. Top Performing Traders Analysis
st.header("Top Performing Traders & Sentiment Interaction")

# Calculate top traders based on overall PnL from the filtered data
if not filtered_df.empty:
    top_n = st.slider("Select number of top traders to display", 1, min(10, len(filtered_df['account'].unique())), 5)
    
    if top_n > 0:
        top_traders_list = filtered_df.groupby('account')['daily_PnL'].sum().nlargest(top_n).index.tolist()
        df_top_traders = filtered_df[filtered_df['account'].isin(top_traders_list)].copy()

        if not df_top_traders.empty:
            fig_top_traders_pnl = px.box(
                df_top_traders,
                x='Classification',
                y='daily_PnL',
                color='account', # Hue for individual traders
                title=f'Daily PnL of Top {top_n} Traders During Fear vs. Greed',
                labels={'daily_PnL': 'Daily PnL', 'Classification': 'Market Sentiment', 'account': 'Trader Account'}
            )
            fig_top_traders_pnl.update_layout(
                yaxis_title='Daily PnL',
                xaxis_title='Sentiment',
                legend_title="Trader Account"
            )
            st.plotly_chart(fig_top_traders_pnl, use_container_width=True)
        else:
            st.info("No data for top traders with current filters.")
    else:
        st.info("Select at least 1 top trader to display this chart.")
else:
    st.info("No data available to determine top traders with current filters.")


st.markdown("---")

# 5. Hypothesis Testing
st.header("Hypothesis Testing: Fear vs. Greed PnL")

fear_pnl = filtered_df[filtered_df['Classification'] == 'Fear']['daily_PnL']
greed_pnl = filtered_df[filtered_df['Classification'] == 'Greed']['daily_PnL']

if len(fear_pnl) > 1 and len(greed_pnl) > 1:
    t_stat, p_val = ttest_ind(fear_pnl, greed_pnl, equal_var=False, nan_policy='omit') # Welch's t-test
    
    st.subheader("T-test Results:")
    st.write(f"**Mean PnL (Fear periods):** ${fear_pnl.mean():,.2f}")
    st.write(f"**Mean PnL (Greed periods):** ${greed_pnl.mean():,.2f}")
    st.write(f"**T-statistic:** {t_stat:.3f}")
    st.write(f"**P-value:** {p_val:.3f}")

    if p_val < 0.05:
        st.success("Conclusion: There is a **statistically significant difference** in daily PnL between Fear and Greed periods. (p-value < 0.05)")
    else:
        st.info("Conclusion: There is **NO statistically significant difference** in daily PnL between Fear and Greed periods. (p-value >= 0.05)")
elif filtered_df.empty:
    st.warning("Not enough data to perform T-test. Please adjust filters.")
else:
    st.warning("Not enough data points for both 'Fear' and 'Greed' sentiments to perform a T-test with current filters. Need at least 2 data points for each category.")


st.markdown("---")

# 6. Correlation Analysis (Interactive)
st.header("Correlation Between Sentiment & Trader Metrics")

# Use Sentiment_Numerical from merged_df for correlation
correlation_df = filtered_df[['Sentiment_Numerical', 'daily_PnL', 'win_rate', 'total_size']].copy()

# Add individual metrics if they were used in the original data and passed through
# Example: if 'avg_leverage' was available, it could be added here
# Make sure to handle NaN values appropriately for correlation
correlation_matrix = correlation_df.corr().fillna(0) # Fill NaN with 0 for display, adjust if needed

fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis",
    title="Correlation Matrix of Sentiment and Trader Metrics"
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# --- Insights and Strategies Section ---
st.header("Key Insights & Potential Trading Strategies")

st.markdown("""
Based on the analysis, here are some insights and potential trading strategies:

### Insights:
* **Sentiment vs. Overall Performance:** Observe how average daily PnL and win rates differ between 'Fear' and 'Greed' market sentiments. A significant difference (supported by the T-test) indicates that market mood plays a role in collective trader profitability.
* **Individual Trader Resilience:** The 'Daily PnL of Top Traders During Fear vs. Greed' chart helps identify if top traders maintain profitability regardless of sentiment, or if they adapt their strategies to excel in specific market conditions.
* **Correlation Patterns:** The correlation matrix reveals the linear relationships between market sentiment (numerical), daily PnL, win rate, and total trade size. Positive correlations suggest metrics move together, while negative ones suggest opposite movements.

### Potential Trading Strategies:
1.  **Sentiment-Adjusted Risk Management:**
    * **During Fear:** If 'Fear' periods correlate with lower PnL or higher volatility, consider reducing leverage or position sizes to mitigate potential losses.
    * **During Greed:** If 'Greed' periods are more profitable, cautiously increase exposure, but always with strict stop-losses to protect against sudden reversals.
2.  **Contrarian vs. Trend-Following:**
    * If 'Fear' periods often mark market bottoms, a contrarian "buy the dip" strategy might be effective.
    * If 'Greed' indicates strong upward momentum, a trend-following approach could be more suitable.
3.  **Dynamic Portfolio Rebalancing:**
    * Adjust your asset allocation based on sentiment shifts. Reduce exposure to highly volatile assets during 'Fear' and potentially increase it during 'Greed' if profitability aligns.
4.  **Identify Sentiment-Resilient Traders:**
    * Study traders who consistently perform well across all sentiment types. Analyze their strategies: trade frequency, preferred assets, entry/exit points, or unique risk management.
5.  **Automated Sentiment Alerts:**
    * Develop a system that notifies you when the market sentiment index changes significantly. This can prompt a review of your current trading strategies and risk parameters.

---

### Innovation & Optimization Notes:
This dashboard provides a foundation. For future enhancements:
* **Advanced Feature Engineering**: Incorporate risk-adjusted returns (Sharpe, Sortino ratios), analyze maximum drawdowns, or explore lagged sentiment indicators.
* **Machine Learning**: Build predictive models to forecast profitable trading days based on sentiment and other market data.
* **Trader Archetyping**: Use clustering algorithms to group traders by behavior patterns (e.g., "scalpers", "swing traders", "long-term holders") and analyze how these archetypes interact with sentiment.
* **Causal Inference**: Move beyond correlation to investigate causal relationships between sentiment and trading outcomes using techniques like Granger Causality.
* **Efficiency for Large Data**: For massive datasets, consider distributed computing frameworks like Dask or PySpark.
""")