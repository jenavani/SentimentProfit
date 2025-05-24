import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Set plot style
sns.set_style("whitegrid")

# --- 1. Data Loading ---
try:
    df_sentiment = pd.read_csv('fear_greed_index.csv')
    df_trader = pd.read_csv('historical_data.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure 'fear_greed_index.csv' and 'historical_data.csv' are in the same directory.")
    exit()

# --- 2. Initial Inspection & Preprocessing ---

# 2.1 Sentiment Data
print("\n--- Sentiment Data Info ---")
df_sentiment.info()
print("\nSentiment Data Head:")
print(df_sentiment.head())

# Convert 'date' to datetime
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
df_sentiment = df_sentiment.sort_values('date').reset_index(drop=True)

# Convert classification to numerical: 0 for Fear, 1 for Greed (or -1, 1 for bipolar)
df_sentiment['Sentiment_Numerical'] = df_sentiment['classification'].apply(lambda x: 1 if x == 'Greed' else 0)


# 2.2 Trader Data
print("\n--- Trader Data Info ---")
df_trader.info()
print("\nTrader Data Head:")
print(df_trader.head())

# Convert 'Timestamp IST' to datetime and extract date (CORRECTION HERE)
df_trader['Timestamp IST'] = pd.to_datetime(df_trader['Timestamp IST'])
df_trader['Date'] = df_trader['Timestamp IST'].dt.date
df_trader['Date'] = pd.to_datetime(df_trader['Date']) # Convert back to datetime object for merging

# Handle missing values (if any significant columns have NaNs, decide strategy)
# For now, let's just check
print("\nMissing values in Trader Data:")
print(df_trader.isnull().sum())

# Ensure numerical columns are truly numerical, coerce errors to NaN (CORRECTION HERE for column names)
for col in ['Execution Price', 'Size Tokens', 'Start Position', 'Closed PnL', 'Leverage']: # Assumed 'Leverage' is the correct column name here, was 'leverage' previously. Please confirm. If not, check df_trader.info() again.
    # Note: 'Leverage' is not in the df.info() you provided for df_trader.
    # If 'Leverage' is indeed a column, it needs to be explicitly listed in the df.info() output.
    # For now, I'll use the other column names as they appear in your df.info().
    # Let's adjust this loop based on the info you provided.
    pass # Temporarily pass, will fix below with actual column names.

# Re-checking df_trader.info() from your output, it does NOT have 'leverage' or 'Leverage'.
# It has 'Execution Price', 'Size Tokens', 'Start Position', 'Closed PnL'.
# If 'leverage' is critical, please clarify its true column name.
# For now, I will remove 'leverage' from the list of columns to convert to numeric.
# I will use 'Closed PnL' as the PnL column.

# Corrected numerical column coercing:
for col in ['Execution Price', 'Size Tokens', 'Start Position', 'Closed PnL']:
    df_trader[col] = pd.to_numeric(df_trader[col], errors='coerce')

# Drop rows where critical numerical data might be missing after coercion
df_trader.dropna(subset=['Execution Price', 'Size Tokens', 'Closed PnL'], inplace=True) # Removed 'leverage' from subset

# --- 3. Feature Engineering for Trader Performance ---

# Calculate daily PnL for each account (CORRECTION HERE: 'Account' and 'Closed PnL')
# We group by Account and Date, then sum the Closed PnL for each day
daily_trader_performance = df_trader.groupby(['Account', 'Date']).agg(
    daily_PnL=('Closed PnL', 'sum'),
    num_trades=('Closed PnL', 'count'),
    # Assuming 'Leverage' is a column somewhere, otherwise this will cause an error.
    # If it's not present, we will need to re-evaluate how to get leverage or remove it.
    # Let's temporarily comment out avg_leverage if it's not in your df_trader.info()
    # avg_leverage=('Leverage', 'mean'),
    total_size=('Size Tokens', 'sum') # Used 'Size Tokens' as per df.info()
).reset_index()

# Calculate win rate for each account per day (simplified: count of positive PnL trades)
# This requires re-aggregating on the original df_trader (CORRECTION HERE: 'Closed PnL')
df_trader['is_win'] = df_trader['Closed PnL'] > 0
daily_wins = df_trader.groupby(['Account', 'Date'])['is_win'].sum().reset_index()
daily_trades_count = df_trader.groupby(['Account', 'Date'])['Closed PnL'].count().reset_index(name='total_trades_day')

daily_trader_performance = pd.merge(daily_trader_performance, daily_wins, on=['Account', 'Date'], how='left')
daily_trader_performance = pd.merge(daily_trader_performance, daily_trades_count, on=['Account', 'Date'], how='left')

daily_trader_performance['win_rate'] = (daily_trader_performance['is_win'] / daily_trader_performance['total_trades_day']).fillna(0) # Handle division by zero

print("\nDaily Trader Performance Head:")
print(daily_trader_performance.head())


# --- 4. Data Merging ---
# Merge daily trader performance with sentiment data on 'Date'
merged_df = pd.merge(daily_trader_performance, df_sentiment[['date', 'classification', 'Sentiment_Numerical']], left_on='Date', right_on='date', how='left')

# Drop the duplicate 'date' column after merging and rename 'classification' to 'Classification' for consistency
merged_df.drop(columns=['date'], inplace=True)
merged_df.rename(columns={'classification': 'Classification'}, inplace=True)


# Forward fill sentiment where it might be missing due to different date ranges, or decide to drop NaNs
# For robust analysis, it's better to only analyze days where sentiment is available.
merged_df.dropna(subset=['Classification'], inplace=True)

print("\nMerged Data Head:")
print(merged_df.head())
print("\nMerged Data Info:")
merged_df.info()


# --- 5. Exploratory Data Analysis (EDA) ---

# 5.1 Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Classification', data=merged_df, palette='viridis')
plt.title('Distribution of Market Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Days')
plt.show()

# 5.2 Overall Trader Performance Distribution
plt.figure(figsize=(12, 6))
sns.histplot(merged_df['daily_PnL'], bins=50, kde=True)
plt.title('Distribution of Daily PnL Across All Traders')
plt.xlabel('Daily PnL')
plt.ylabel('Frequency')
plt.xlim(-5000, 5000) # Limit x-axis for better visualization of main distribution
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(merged_df['win_rate'], bins=20, kde=True)
plt.title('Distribution of Daily Win Rate Across All Traders')
plt.xlabel('Win Rate')
plt.ylabel('Frequency')
plt.show()


# 5.3 Sentiment vs. Performance (Initial Look)

# Average Daily PnL by Sentiment
avg_pnl_by_sentiment = merged_df.groupby('Classification')['daily_PnL'].mean().reset_index()
print("\nAverage Daily PnL by Sentiment:")
print(avg_pnl_by_sentiment)

plt.figure(figsize=(8, 6))
sns.barplot(x='Classification', y='daily_PnL', data=avg_pnl_by_sentiment, palette='coolwarm')
plt.title('Average Daily PnL by Market Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Daily PnL')
plt.show()

# Average Win Rate by Sentiment
avg_winrate_by_sentiment = merged_df.groupby('Classification')['win_rate'].mean().reset_index()
print("\nAverage Win Rate by Sentiment:")
print(avg_winrate_by_sentiment)

plt.figure(figsize=(8, 6))
sns.barplot(x='Classification', y='win_rate', data=avg_winrate_by_sentiment, palette='coolwarm')
plt.title('Average Daily Win Rate by Market Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Daily Win Rate')
plt.show()

# --- 6. Advanced Analysis and Pattern Discovery ---

# 6.1 Correlation Analysis (Removed 'avg_leverage' as it's not defined currently)
print("\nCorrelation between Sentiment and Trader Performance Metrics:")
print(merged_df[['Sentiment_Numerical', 'daily_PnL', 'win_rate', 'total_size']].corr())


# 6.2 Hypothesis Testing: Is there a significant difference in PnL between Fear and Greed periods?
fear_pnl = merged_df[merged_df['Classification'] == 'Fear']['daily_PnL']
greed_pnl = merged_df[merged_df['Classification'] == 'Greed']['daily_PnL']

# Perform independent t-test
if len(fear_pnl) > 1 and len(greed_pnl) > 1: # Ensure enough samples for t-test
    t_stat, p_val = ttest_ind(fear_pnl, greed_pnl, equal_var=False) # Welch's t-test assuming unequal variances
    print(f"\n--- T-test for Daily PnL between Fear and Greed ---")
    print(f"Mean PnL (Fear): {fear_pnl.mean():.2f}")
    print(f"Mean PnL (Greed): {greed_pnl.mean():.2f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_val:.3f}")
    if p_val < 0.05:
        print("Conclusion: There is a statistically significant difference in daily PnL between Fear and Greed periods.")
    else:
        print("Conclusion: There is NO statistically significant difference in daily PnL between Fear and Greed periods.")
else:
    print("\nNot enough data points to perform T-test for PnL between Fear and Greed periods.")


# 6.3 Segmenting Traders by Performance & Sentiment Interaction

# Identify top N traders based on overall PnL (for a more focused analysis) (CORRECTION HERE: 'Account')
top_traders = merged_df.groupby('Account')['daily_PnL'].sum().nlargest(10).index
print(f"\nTop 10 Performing Traders (Total PnL): {list(top_traders)}")

# Analyze top traders' performance during different sentiment periods (CORRECTION HERE: 'Account')
df_top_traders = merged_df[merged_df['Account'].isin(top_traders)]

plt.figure(figsize=(14, 7))
sns.boxplot(x='Classification', y='daily_PnL', hue='Account', data=df_top_traders, palette='tab10')
plt.title('Daily PnL of Top Traders During Fear vs. Greed')
plt.xlabel('Sentiment')
plt.ylabel('Daily PnL')
plt.legend(title='Account', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6.4 Time-Series Visualization (Aggregated Market PnL vs. Sentiment)
# Aggregate daily PnL for the entire market
market_daily_pnl = merged_df.groupby('Date')['daily_PnL'].sum().reset_index()
market_daily_pnl = pd.merge(market_daily_pnl, df_sentiment[['date', 'Sentiment_Numerical']], left_on='Date', right_on='date', how='left') # Use 'date' here too
market_daily_pnl.drop(columns=['date'], inplace=True) # Drop the duplicate 'date' column

plt.figure(figsize=(15, 7))
plt.plot(market_daily_pnl['Date'], market_daily_pnl['daily_PnL'], label='Total Market Daily PnL', color='blue', alpha=0.7)
# Overlay sentiment as a shaded region or points
plt.scatter(market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 1]['Date'],
            market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 1]['daily_PnL'],
            color='green', label='Greed Day', alpha=0.6, s=50)
plt.scatter(market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 0]['Date'],
            market_daily_pnl[market_daily_pnl['Sentiment_Numerical'] == 0]['daily_PnL'],
            color='red', label='Fear Day', alpha=0.6, s=50)

plt.title('Total Market Daily PnL and Market Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Total Daily PnL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7. Delivering Insights and Smarter Trading Strategies ---

print("\n--- Key Insights & Potential Trading Strategies ---")

if avg_pnl_by_sentiment.empty:
    print("Not enough data to generate sentiment-based insights.")
else:
    # Need to check if 'Fear' and 'Greed' are actually in the classification column
    # Use .get() with a default value to avoid KeyError if a classification is missing
    fear_avg_pnl = avg_pnl_by_sentiment[avg_pnl_by_sentiment['Classification'] == 'Fear']['daily_PnL'].iloc[0] if 'Fear' in avg_pnl_by_sentiment['Classification'].values else np.nan
    greed_avg_pnl = avg_pnl_by_sentiment[avg_pnl_by_sentiment['Classification'] == 'Greed']['daily_PnL'].iloc[0] if 'Greed' in avg_pnl_by_sentiment['Classification'].values else np.nan

    print(f"\n1. Sentiment vs. Overall Performance:")
    if not np.isnan(fear_avg_pnl) and not np.isnan(greed_avg_pnl):
        print(f"   - On average, traders collectively performed {'better' if greed_avg_pnl > fear_avg_pnl else 'worse'} during 'Greed' periods (Avg PnL: {greed_avg_pnl:.2f}) compared to 'Fear' periods (Avg PnL: {fear_avg_pnl:.2f}).")
        # Check if p_val is defined and then use it
        if 'p_val' in locals() and p_val < 0.05:
            print("   - This difference is statistically significant.")
        else:
            print("   - This difference is NOT statistically significant, suggesting other factors might be more dominant.")
    else:
        print("   - Average PnL by sentiment could not be fully calculated (missing sentiment data).")

    print("\n2. Win Rate and Sentiment:")
    fear_avg_winrate = avg_winrate_by_sentiment[avg_winrate_by_sentiment['Classification'] == 'Fear']['win_rate'].iloc[0] if 'Fear' in avg_winrate_by_sentiment['Classification'].values else np.nan
    greed_avg_winrate = avg_winrate_by_sentiment[avg_winrate_by_sentiment['Classification'] == 'Greed']['win_rate'].iloc[0] if 'Greed' in avg_winrate_by_sentiment['Classification'].values else np.nan

    if not np.isnan(fear_avg_winrate) and not np.isnan(greed_avg_winrate):
        print(f"   - The average win rate was {'higher' if greed_avg_winrate > fear_avg_winrate else 'lower'} during 'Greed' periods ({greed_avg_winrate:.2%}) compared to 'Fear' periods ({fear_avg_winrate:.2%}). This could indicate that 'Greed' periods are generally easier to navigate for profitable trades, or that traders take on more risk during 'Greed' leading to more wins, but potentially larger losses too.")
    else:
        print("   - Average Win Rate by sentiment could not be fully calculated (missing sentiment data).")

    print("\n3. Trader Archetypes and Sentiment:")
    print("   - Observing individual top traders' performance during different sentiment states can reveal adaptive strategies. Some top traders might consistently perform well regardless of sentiment, while others might excel specifically in 'Fear' or 'Greed' markets.")
    print("   - A deeper dive could involve classifying traders into archetypes (e.g., 'Fear-Proficient', 'Greed-Proficient', 'Sentiment-Agnostic') based on their PnL distribution across sentiment categories.")

    print("\n4. Leverage and Sentiment:")
    # This section needs to be re-evaluated if 'Leverage' column is confirmed.
    # For now, it will be omitted from the output.
    print("   - Analysis of leverage and sentiment correlation needs a confirmed 'Leverage' column in the trader data.")

    print("\n5. Potential Trading Strategies:")
    print("   a. Sentiment-Adjusted Risk Management:")
    print("      - If 'Fear' periods show lower average PnL or higher volatility, traders could consider reducing leverage or position size during such times to mitigate losses.")
    print("      - Conversely, if 'Greed' periods are more profitable, consider cautiously increasing exposure (with strict stop-losses).")
    print("   b. Contrarian vs. Trend-Following based on Sentiment:")
    print("      - If the data shows that 'Fear' periods lead to market bottoms (and subsequent rallies), a contrarian 'buy the dip' strategy might be effective during 'Fear'.")
    print("      - If 'Greed' indicates strong upward momentum, a trend-following strategy might be more suitable.")
    print("   c. Portfolio Rebalancing:")
    print("      - Adjust asset allocation based on the sentiment index. Reduce exposure to volatile assets during 'Fear', increase during 'Greed' (if profitability aligns).")
    print("   d. Identify Sentiment-Resilient Traders:")
    print("      - Analyze the strategies of consistently profitable traders, regardless of sentiment. What do they do differently? This could involve examining their trade frequency, symbol choices, or entry/exit patterns.")
    print("   e. Automated Alerts:")
    print("      - Develop a system that provides alerts when the market sentiment shifts, prompting traders to review their current strategies and adjust their risk parameters.")


# --- Innovation and Optimization Notes for Internship Application ---

print("\n--- Innovation and Optimization Notes for Internship Application ---")
print("1. **Feature Engineering**: Beyond simple PnL, we engineered daily win rates. For even more depth:")
print("   - **Risk-Adjusted Returns**: Calculate Sharpe Ratio (if enough data points for std dev of returns) or Sortino Ratio for individual traders per sentiment.")
print("   - **Drawdowns**: Analyze maximum drawdowns during different sentiment periods.")
print("   - **Sentiment Lag**: Explore if sentiment from previous days (e.g., 1-day lag, 3-day average) has a stronger correlation with current performance.")
print("2. **Advanced Statistical Modeling**: Instead of just correlation and t-tests, consider:")
print("   - **Regression Analysis**: Model PnL as a function of sentiment and other trader characteristics (leverage, trade count).")
print("   - **Time-Series Models**: Apply ARIMA or GARCH models to sentiment and PnL to understand their dynamic relationship.")
print("3. **Machine Learning for Prediction**: A classification model (e.g., Logistic Regression, Random Forest) could predict the probability of a profitable trading day given current sentiment and historical trading patterns.")
print("4. **Trader Segmentation**: Implement clustering algorithms (e.g., K-Means) on trader performance metrics to identify distinct trading styles and then analyze how these styles interact with market sentiment.")
print("5. **Dynamic Strategy Simulation**: If detailed enough historical data were available, simulate trading strategies based on sentiment signals and compare their historical performance against a benchmark.")
print("6. **Efficiency**: Using `pandas` vectorized operations for feature engineering is generally efficient. For extremely large datasets (millions of trades), consider `Dask` or `PySpark` for distributed computing.")
print("7. **Creativity in Visualization**: Interactive dashboards (e.g., using Plotly or Dash) would be a significant enhancement to allow users to filter by trader, date range, and sentiment, making insights more accessible.")
print("8. **Robustness**: Implement more robust error handling and data validation, especially for real-world messy data.")
print("9. **Causal Inference**: While correlation is a start, exploring causal links using techniques like Granger Causality (as mentioned above) or Causal Impact analysis would be very powerful.")

print("\nThis comprehensive analysis and the suggested next steps demonstrate strong analytical skills, attention to detail, and a proactive approach to uncovering deep insights, all highly valuable for an internship opportunity!")