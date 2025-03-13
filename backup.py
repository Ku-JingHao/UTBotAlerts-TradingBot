import pandas as pd
import numpy as np
import talib
import vectorbt as vbt
import time
import logging
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Custom logging formatter with colors and better structure
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better structure for log messages"""
    
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.INFO: Fore.GREEN + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.WARNING: Fore.YELLOW + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.ERROR: Fore.RED + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.CRITICAL: Fore.RED + Back.WHITE + "%(asctime)s | %(levelname)-8s | %(message)s" + Style.RESET_ALL
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Setup logging with custom formatter
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with standard formatting
    file_handler = logging.FileHandler("ut_bot_trading.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Console handler with colored formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# UT Bot Parameters
SENSITIVITY = 1
ATR_PERIOD = 10

# Trading Parameters
TICKER = "BTC/USD" 
INTERVAL = "1d"
TRADE_AMOUNT = 100  # Amount in USD to trade per position

# Alpaca API credentials
ALPACA_API_KEY = "PKZ4QV53VSJLHL90ZR0C"
ALPACA_SECRET_KEY = "nnGnqdiCfFKGpERVs3WxYyNgGozuCof7cKaNs0Wd"

# Initialize Alpaca clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient()

def get_historical_data():
    """Fetch historical data from Alpaca for the specified ticker and timeframe"""
    # For daily timeframe, get 100 days of data
    timeframe = TimeFrame.Day
    
    # Calculate start and end dates
    end = datetime.now()
    start = end - timedelta(days=100)
    
    # Create the request
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[TICKER],
        timeframe=timeframe,
        start=start,
        end=end
    )
    
    # Get the data
    bars = data_client.get_crypto_bars(request_params)
    
    # Convert to dataframe
    df = bars.df.reset_index()
    
    # Restructure the dataframe
    df = df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    logger.info(f"Fetched {len(df)} historical bars for {TICKER}")
    return df

def calculate_signals(pd_data):
    """Calculate UT Bot signals based on ATR Trailing Stop"""
    # Compute ATR And nLoss variable
    pd_data["xATR"] = talib.ATR(pd_data["High"], pd_data["Low"], pd_data["Close"], timeperiod=ATR_PERIOD)
    pd_data["nLoss"] = SENSITIVITY * pd_data["xATR"]
    
    # Drop all rows that have nan, first X rows depending on the ATR period for the moving average
    pd_data = pd_data.dropna()
    pd_data = pd_data.reset_index(drop=True)
    
    # Filling ATRTrailingStop Variable
    pd_data["ATRTrailingStop"] = [0.0] + [np.nan for i in range(len(pd_data) - 1)]
    
    for i in range(1, len(pd_data)):
        pd_data.loc[i, "ATRTrailingStop"] = xATRTrailingStop_func(
            pd_data.loc[i, "Close"],
            pd_data.loc[i - 1, "Close"],
            pd_data.loc[i - 1, "ATRTrailingStop"],
            pd_data.loc[i, "nLoss"],
        )
    
    # Calculating signals
    ema = vbt.MA.run(pd_data["Close"], 1, short_name='EMA', ewm=True)
    
    pd_data["Above"] = ema.ma_crossed_above(pd_data["ATRTrailingStop"])
    pd_data["Below"] = ema.ma_crossed_below(pd_data["ATRTrailingStop"])
    
    pd_data["Buy"] = (pd_data["Close"] > pd_data["ATRTrailingStop"]) & (pd_data["Above"]==True)
    pd_data["Sell"] = (pd_data["Close"] < pd_data["ATRTrailingStop"]) & (pd_data["Below"]==True)
    
    return pd_data

def xATRTrailingStop_func(close, prev_close, prev_atr, nloss):
    """Calculate the ATR Trailing Stop value"""
    if close > prev_atr and prev_close > prev_atr:
        return max(prev_atr, close - nloss)
    elif close < prev_atr and prev_close < prev_atr:
        return min(prev_atr, close + nloss)
    elif close > prev_atr:
        return close - nloss
    else:
        return close + nloss

def get_current_position():
    """Get the current position for the ticker"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == "BTCUSD":  # Hard-coded symbol format for Alpaca
                return float(position.qty)
        return 0
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        return 0

def execute_trade(side, price):
    """Execute a trade on Alpaca"""
    try:
        if side not in [OrderSide.BUY, OrderSide.SELL]:
            logger.error(f"Invalid order side: {side}")
            return False
        
        # Calculate quantity based on trade amount
        quantity = TRADE_AMOUNT / price
        
        # Prepare market order with correct symbol format for Alpaca
        order_data = MarketOrderRequest(
            symbol="BTCUSD",  # Hard-coded correct format for Alpaca
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        # Submit order
        order = trading_client.submit_order(order_data)
        
        # Print detailed trade information with better formatting
        logger.info(f"{'=' * 50}")
        logger.info(f"TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'Type:':<15} {side.name}")
        logger.info(f"{'Symbol:':<15} {TICKER}")
        logger.info(f"{'Quantity:':<15} {quantity:.6f}")
        logger.info(f"{'Price:':<15} ${price:.2f}")
        logger.info(f"{'Total Value:':<15} ${TRADE_AMOUNT:.2f}")
        logger.info(f"{'Order ID:':<15} {order.id}")
        logger.info(f"{'=' * 50}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_bot_header():
    """Print a nicely formatted header when the bot starts"""
    header = f"""
{'=' * 60}
{Fore.CYAN}
  _   _ _____   ____        _     _____               _         
 | | | |_   _| | __ )  ___ | |_  |_   _| __ __ _  __| | ___ _ __ 
 | | | | | |   |  _ \\ / _ \\| __|   | || '__/ _` |/ _` |/ _ \\ '__|
 | |_| | | |   | |_) | (_) | |_    | || | | (_| | (_| |  __/ |   
  \\___/  |_|   |____/ \\___/ \\__|   |_||_|  \\__,_|\\__,_|\\___|_|   
{Style.RESET_ALL}
{'=' * 60}
"""
    print(header)
    
    # Log bot configuration
    logger.info(f"{'STRATEGY:':<15} UT Bot with ATR Trailing Stop")
    logger.info(f"{'SYMBOL:':<15} {TICKER}")
    logger.info(f"{'TIMEFRAME:':<15} {INTERVAL}")
    logger.info(f"{'SENSITIVITY:':<15} {SENSITIVITY}")
    logger.info(f"{'ATR PERIOD:':<15} {ATR_PERIOD}")
    logger.info(f"{'TRADE AMOUNT:':<15} ${TRADE_AMOUNT}")
    logger.info(f"{'TRADING MODE:':<15} Paper Trading (Alpaca)")

def print_signal_update(latest_bar, current_price, current_position):
    """Print a nicely formatted signal update"""
    print(f"{'=' * 60}")
    logger.info(f"DAILY SIGNAL UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'Date:':<15} {latest_bar['Date']}")
    logger.info(f"{'Price:':<15} ${current_price:.2f}")
    logger.info(f"{'ATR Stop:':<15} ${latest_bar['ATRTrailingStop']:.2f}")
    logger.info(f"{'Position:':<15} {current_position} {TICKER.split('/')[0]}")
    
    # Use different colored signals based on buy/sell
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}{'SIGNAL:':<15} BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}{'SIGNAL:':<15} SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}{'SIGNAL:':<15} NO SIGNAL ⚪{Style.RESET_ALL}")

def run_live_trading():
    """Run the UT Bot strategy continuously in live trading mode"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    
    last_check_date = None
    
    while True:
        try:
            # Get current day
            current_date = datetime.now().date()
            
            # Only process once per day for daily timeframe
            if last_check_date != current_date:
                # Get historical data
                df = get_historical_data()
                
                # Calculate signals
                signals_df = calculate_signals(df)
                
                # Check latest signal (most recent bar)
                latest_bar = signals_df.iloc[-1]
                current_price = latest_bar['Close']
                
                # Get current position
                current_position = get_current_position()
                position_exists = abs(current_position) > 0
                
                # Log detailed signal information with better formatting
                print_signal_update(latest_bar, current_price, current_position)
                
                # Process buy signal
                if latest_bar['Buy']:
                    # If we have a short position, close it
                    if current_position < 0:
                        logger.info(f"Closing short position of {abs(current_position)} {TICKER}")
                        execute_trade(OrderSide.BUY, current_price)
                    
                    # Open a new long position if we don't have one
                    if current_position <= 0:
                        logger.info(f"Opening new long position at ${current_price:.2f}")
                        execute_trade(OrderSide.BUY, current_price)
                
                # Process sell signal
                elif latest_bar['Sell']:
                    # If we have a long position, close it
                    if current_position > 0:
                        logger.info(f"Closing long position of {current_position} {TICKER}")
                        execute_trade(OrderSide.SELL, current_price)
                    
                    # Open a new short position if we don't have one
                    if current_position >= 0:
                        logger.info(f"Opening new short position at ${current_price:.2f}")
                        execute_trade(OrderSide.SELL, current_price)
                
                else:
                    logger.info("No new trading signals today")
                
                # Record that we checked today
                last_check_date = current_date
                
                # Log when the next check will occur
                #next_check_time = datetime.now() + timedelta(hours=1)
                #logger.info(f"{'Next check:':<15} {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                #logger.info(f"{'=' * 60}")
            
            # Sleep for 1 hour before checking again
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in live trading execution: {e}")
            logger.error(f"{'=' * 60}")
            # Sleep for 5 minutes before retrying
            time.sleep(300)

def calculate_roi_summary():
    """Calculate and print ROI summary for each week"""
    if not trade_history:
        logger.info("No trades executed. ROI summary not available.")
        return
    
    # Group trades by week
    for trade in trade_history:
        week_start = trade['timestamp'] - timedelta(days=trade['timestamp'].weekday())
        week_key = week_start.strftime('%Y-%m-%d')
        if week_key not in weekly_performance:
            weekly_performance[week_key] = {
                'initial_cash': trade['cash_left'] + (trade['quantity'] * trade['price'] if trade['side'] == OrderSide.SELL else trade['cash_left']),
                'final_cash': trade['cash_left'],
                'trades': []
            }
        weekly_performance[week_key]['trades'].append(trade)
        weekly_performance[week_key]['final_cash'] = trade['cash_left']
    
    # Calculate ROI for each week
    week_counter = 1  # Initialize week counter
    for week, data in weekly_performance.items():
        initial_cash = data['initial_cash']
        final_cash = data['final_cash']
        roi = ((final_cash - initial_cash) / initial_cash) * 100
        logger.info(f"Week {week_counter} ({week}) ROI: {roi:.2f}%")
        logger.info(f"Initial Cash: ${initial_cash:.2f}")
        logger.info(f"Final Cash: ${final_cash:.2f}")
        logger.info(f"Number of Trades: {len(data['trades'])}")
        print(f"{'=' * 60}")
        week_counter += 1  # Increment week counter
        
if __name__ == "__main__":
    run_live_trading()