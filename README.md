# A Research Analysis of UTBot Alerts Using a Trading Bot

## Overview

This project is a research initiative to evaluate the effectiveness of the **UT Bot Alerts** indicator in real-world trading scenarios. The main objective is to test and analyze the performance of UT Bot Alerts by developing and deploying real-time trading bots that operate on various time intervals and across different assets, including both stocks and cryptocurrencies.

## Features

- **Real-Time Trading Bots:** Automated bots for live trading and signal testing.
- **Multiple Timeframes:** Supports daily, hourly, and minute-based strategies.
- **Multi-Asset Support:** Easily switch between different cryptocurrencies and stocks.
- **Alpaca Integration:** Uses [Alpaca](https://alpaca.markets/) API for live and paper trading.
- **WhatsApp Alerts:** Sends trade and signal notifications to WhatsApp for real-time monitoring.
- **Comprehensive Logging:** Colored and structured logs for easy monitoring and debugging.
- **Performance Analysis:** Calculates and logs ROI summaries for each interval and overall performance.

## Project Structure

- `DailyBot.py`, `HourlyBot.py`, `1MinuteBot.py`: First Version Without WhatsApp Alert Bot Scripts
- `Daily.py`, `Hourly.py`, `Minute.py`: Advanced Bots using Alpaca data with WhatsApp alert integration for different timeframes.
- `YfinanceBot_Daily.py`, `YfinanceBot_Hourly.py`: Bots using Yahoo Finance data for backtesting or signal alerts.
- `specific_crypto/`: Folder containing bot scripts for specific cryptocurrencies (e.g., ADA, CRV, SUSHI, XRP, etc.).
- `TradingBot.ipynb`: Jupyter notebook for interactive research, testing, and demonstration.
- `backup.py`: Backup or alternative implementation.

## How It Works

1. **Signal Generation:** The bots use the UT Bot Alerts indicator (ATR Trailing Stop logic) to generate buy/sell signals.
2. **Automated Execution:** When a signal is detected, the bot can execute trades automatically via Alpaca (paper/live).
3. **WhatsApp Notifications:** Each trade or signal can trigger a WhatsApp message for real-time updates.
4. **Performance Tracking:** All trades are logged, and ROI is calculated for each interval and overall.

## Getting Started

### Prerequisites

- Python 3.8+
- [Alpaca API keys](https://alpaca.markets/)
- WhatsApp Web (for pywhatkit notifications)
- Required Python packages:
  - `pandas`, `numpy`, `talib`, `vectorbt`, `colorama`, `keyboard`, `alpaca-py`, `pywhatkit`, `yfinance` (for Yahoo Finance bots)

### Installation

Install dependencies:

```bash
pip install pandas numpy ta-lib vectorbt colorama keyboard alpaca-py pywhatkit yfinance
```

> **Note:** TA-Lib may require additional system dependencies. See [TA-Lib installation instructions](https://mrjbq7.github.io/ta-lib/install.html).

### Configuration

- Set your Alpaca API keys in the relevant bot scripts.
- Update WhatsApp phone numbers in the scripts if you want to receive alerts.

### Running the Bots

- **Daily Bot:**  
  ```bash
  python Daily.py
  ```
- **Hourly Bot:**  
  ```bash
  python Hourly.py
  ```
- **1-Minute Bot:**  
  ```bash
  python Minute.py
  ```
- **Yahoo Finance Bots:**  
  ```bash
  python YfinanceBot_Daily.py
  python YfinanceBot_Hourly.py
  ```
- **Jupyter Notebook:**  
  Open `TradingBot.ipynb` in VS Code or Jupyter Lab for interactive research.

## Customization

- **Change Asset:**  
  Modify the `TICKER` variable in the script to trade a different stock or crypto.
- **Change Timeframe:**  
  Adjust the `INTERVAL` variable and use the corresponding bot script.
- **WhatsApp Alerts:**  
  Toggle `ENABLE_WHATSAPP_ALERTS` and update phone numbers as needed.

## Research Goals

- **Evaluate UT Bot Alerts:**  
  Assess the reliability and profitability of the UT Bot Alerts indicator in live and historical trading.
- **Compare Timeframes:**  
  Analyze performance across daily, hourly, and minute intervals.
- **Asset Versatility:**  
  Test the indicator on various cryptocurrencies and stocks.

## Disclaimer

This project is for research and educational purposes only. **Trading involves risk.** Use paper trading for testing and do not risk real capital unless you fully understand the risks involved.

**License:**  
MIT License
