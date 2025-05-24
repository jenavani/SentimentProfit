# Crypto Trader Performance & Market Sentiment Analysis Dashboard

This project provides an interactive Streamlit dashboard to analyze the performance of crypto traders in conjunction with the broader market sentiment, as indicated by the Fear & Greed Index. It aims to uncover insights into how market emotions might influence trading outcomes and identify patterns in trader behavior across different sentiment states.

## Table of Contents
- [Features](#features)
- [Data Sources](#data-sources)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Application](#how-to-run-the-application)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Insights & Strategies](#insights--strategies)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

## Features

The dashboard offers the following key functionalities and visualizations:

* **Key Performance Indicators (KPIs):** Displays aggregated metrics such as Total PnL, Average Daily PnL, Average Win Rate, and Total Trades.
* **Interactive Filters:**
    * **Date Range:** Analyze performance within specific time periods.
    * **Trader Accounts:** Filter data to focus on specific traders or groups of traders.
    * **Market Sentiment:** Examine performance exclusively during 'Fear' or 'Greed' periods.
* **Time-Series Visualization:** A dynamic plot showing the total market daily PnL over time, with markers indicating 'Fear' and 'Greed' days.
* **Sentiment vs. Performance Analysis:** Bar charts illustrating the average daily PnL and win rate categorized by market sentiment (Fear vs. Greed).
* **Performance Distribution:** Box plots showcasing the distribution of daily PnL across all traders, segmented by market sentiment, to reveal spread and outliers.
* **Top Performing Traders Analysis:** A customizable view (e.g., top 5, top 10 traders) showing their individual PnL distributions during different sentiment phases, helping identify sentiment-resilient traders.
* **Hypothesis Testing:** Performs a statistical t-test to determine if there's a significant difference in daily PnL between 'Fear' and 'Greed' periods.
* **Correlation Matrix:** Visualizes the correlations between sentiment (numerical) and various trader performance metrics.
* **Key Insights & Strategies:** A dedicated section outlining actionable insights derived from the analysis and suggesting potential trading strategies.

## Data Sources

The application utilizes two primary datasets:

1.  `fear_greed_index.csv`: Contains historical market sentiment data (e.g., Fear, Greed, Extreme Fear, Extreme Greed) along with a numerical value and timestamp.
2.  `historical_data.csv`: Contains granular historical trading data, including account IDs, execution details, and profit/loss information.

**Note:** Ensure these CSV files are placed in the same directory as the `app.py` script.

## Setup and Installation

To get this project up and running on your local machine, follow these steps:

1.  **Prerequisites:**
    * Python 3.8+ installed.
    * `pip` (Python package installer).

2.  **Clone the Repository (or download files):**
    If this is hosted on GitHub, you would clone it:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    If you just have the files locally, navigate to the directory containing `app.py` and the CSV files.

3.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    # Install required packages
    pip install streamlit pandas plotly scikit-learn scipy numpy
    ```

## How to Run the Application

Once you have installed the dependencies and placed the CSV files in the correct directory:

1.  **Activate your virtual environment** (if you haven't already).
2.  **Run the Streamlit application** from your terminal in the project directory:
    ```bash
    streamlit run app.py
    ```
3.  This command will open a new tab in your web browser with the interactive dashboard.

## Usage

* **Sidebar Filters:** Use the filters on the left sidebar to refine the data displayed in the dashboard. You can select date ranges, specific trader accounts, and market sentiment categories.
* **Explore Visualizations:** Interact with the Plotly charts by hovering over data points for details, zooming, and panning.
* **Insights Section:** Read the "Key Insights & Potential Trading Strategies" section for a summary of findings and actionable recommendations.

## File Structure
.
├── app.py                      # Main Streamlit application script
├── fear_greed_index.csv        # Market sentiment data
└── historical_data.csv         # Historical trader performance data
└── README.md                   # This README file


**(Optional: Create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all dependencies.)**

## Insights & Strategies

(You can copy and paste the detailed insights from the "Key Insights & Potential Trading Strategies" section of your `app.py` here for a more comprehensive README.)

## Future Enhancements

This project can be further optimized and expanded. Some ideas include:

* **Advanced Feature Engineering:** Incorporate risk-adjusted returns (Sharpe Ratio, Sortino Ratio), maximum drawdowns, or explore lagged sentiment indicators.
* **Machine Learning Models:** Implement predictive models to forecast profitable trading days based on sentiment and other market signals.
* **Trader Segmentation:** Utilize clustering algorithms (e.g., K-Means) to identify distinct trading styles and analyze their interaction with market sentiment.
* **Causal Inference:** Investigate causal relationships between sentiment shifts and trading outcomes using advanced statistical methods.
* **More Granular Data:** Integrate data at a higher frequency (e.g., hourly, minute-by-minute) for more detailed analysis.
* **Real-time Data Integration:** Connect to live APIs for real-time sentiment and tradi
