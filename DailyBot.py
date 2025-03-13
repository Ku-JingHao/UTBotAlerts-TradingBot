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
ALPACA_API_KEY = "PKCRL2GR9JEO7MWNAXLI"
ALPACA_SECRET_KEY = "808AQEP0SNfbMbGfp0LQANuZfPcxU97etX5hCrOJ"

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
    
    logger.info(f"Fetched {len(df)} days historical bars for {TICKER}")
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
            if position.symbol == "BTC/USD":  # Hard-coded symbol format for Alpaca
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
        print(f"{'=' * 60}")
        logger.info(f"TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Type           : {side.name}")
        logger.info(f"Symbol         : {TICKER}")
        logger.info(f"Quantity       : {quantity:.6f}")
        logger.info(f"Price          : ${price:.2f}")
        logger.info(f"Total Value    : ${TRADE_AMOUNT:.2f}")
        logger.info(f"Order ID       : {order.id}")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_detailed_signals(signals_df):
    """Print detailed information for each bar in the dataframe with all signals"""
    print(f"{'=' * 60}")
    logger.info(f"DETAILED SIGNAL ANALYSIS FOR ALL {len(signals_df)} BARS")
    print("\n")

    # Define headers with fixed spacing for better alignment
    header = (
        f"{'Date':<12} | "
        f"{'Open':>10} | "
        f"{'High':>10} | "
        f"{'Low':>10} | "
        f"{'Close':>10} | "
        f"{'ATR':>8} | "
        f"{'ATR Stop':>10} | "
        f"{'Above':^7} | "
        f"{'Below':^7} | "
        f"{'Buy':^7} | "
        f"{'Sell':^7} | "
        f"{'Signal':<10}"
    )
    
    logger.info(header)
    logger.info(f"{'-' * 60}")
    
    # Print each row with proper formatting
    for i, row in signals_df.iterrows():
        # Format date (keep only the date part if it's a datetime object)
        if isinstance(row['Date'], pd.Timestamp):
            date_str = row['Date'].strftime('%Y-%m-%d')
        else:
            date_str = str(row['Date'])[:10]
        
        # Determine signal color and text
        if row['Buy']:
            signal_color = Fore.CYAN
            signal_text = "BUY 🔵"
        elif row['Sell']:
            signal_color = Fore.MAGENTA
            signal_text = "SELL 🔴"
        else:
            signal_color = Fore.YELLOW
            signal_text = "NONE ⚪"
        
        # Format row data
        row_data = (
            f"{date_str:<12} | "
            f"${row['Open']:>9.2f} | "
            f"${row['High']:>9.2f} | "
            f"${row['Low']:>9.2f} | "
            f"${row['Close']:>9.2f} | "
            f"{row['xATR']:>7.2f} | "
            f"${row['ATRTrailingStop']:>9.2f} | "
            f"{str(row['Above']):^7} | "
            f"{str(row['Below']):^7} | "
            f"{str(row['Buy']):^7} | "
            f"{str(row['Sell']):^7} | "
            f"{signal_color}{signal_text}{Style.RESET_ALL}"
        )
        
        logger.info(row_data)
    
    print(f"{'=' * 60}")

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
    
    # Log bot configuration with better alignment
    logger.info(f"STRATEGY       : UT Bot with ATR Trailing Stop")
    logger.info(f"SYMBOL         : {TICKER}")
    logger.info(f"TIMEFRAME      : {INTERVAL}")
    logger.info(f"SENSITIVITY    : {SENSITIVITY}")
    logger.info(f"ATR PERIOD     : {ATR_PERIOD}")
    logger.info(f"TRADE AMOUNT   : ${TRADE_AMOUNT}")
    logger.info(f"TRADING MODE   : Paper Trading (Alpaca)")

def print_signal_update(latest_bar, current_price, current_position):
    """Print a nicely formatted signal update"""
    print(f"{'=' * 60}")
    logger.info(f"DAILY SIGNAL UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Date           : {latest_bar['Date']}")
    logger.info(f"Price          : ${current_price:.2f}")
    logger.info(f"ATR Stop       : ${latest_bar['ATRTrailingStop']:.2f}")
    logger.info(f"Position       : {current_position} {TICKER.split('/')[0]}")
    
    # Use different colored signals based on buy/sell
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}SIGNAL         : BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}SIGNAL         : SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}SIGNAL         : NO SIGNAL ⚪{Style.RESET_ALL}")

def print_waiting_message(next_check_time):
    """Print a clear waiting message to show bot is active and waiting for next check"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}BOT STATUS      : MONITORING {TICKER}{Style.RESET_ALL}")
    logger.info(f"Current Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Next Check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate time remaining
    time_remaining = next_check_time - datetime.now()
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Time Remaining : {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    if time_remaining.days > 0:
        logger.info(f"Days Remaining : {time_remaining.days}")
        
    # Show current market status
    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5  # 5 Saturday, 6 Sunday
    
    if is_weekend:
        logger.info(f"{Fore.YELLOW}Market Status  : Weekend - Limited Trading Activity{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.GREEN}Market Status  : Weekday - Normal Trading Hours{Style.RESET_ALL}")
        
    print(f"{'=' * 60}")

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
                
                # Print detailed signals for all bars (new function)
                # print_detailed_signals(signals_df)
                
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
                
                # Calculate next check time
                next_check_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
                # logger.info(f"Next check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Print waiting message every hour
            next_check_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
            if last_check_date == current_date:  # Only show waiting message after today's check is complete
                print_waiting_message(next_check_time)
            
            # Sleep for 1 hour before checking again
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in live trading execution: {e}")
            logger.error(f"{'=' * 60}")
            # Sleep for 5 minutes before retrying
            time.sleep(300)

def run_signal_test():
    """Run a test to print all historical signals without executing trades"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    logger.info("RUNNING IN SIGNAL TEST MODE - NO TRADES WILL BE EXECUTED")
    
    # Get historical data
    df = get_historical_data()
    
    # Calculate signals
    signals_df = calculate_signals(df)
    
    # Print detailed signals for all bars
    print_detailed_signals(signals_df)
    
    # Check latest signal (most recent bar)
    latest_bar = signals_df.iloc[-1]
    current_price = latest_bar['Close']
    
    # Get current position
    current_position = get_current_position()
    
    # Log detailed signal information with better formatting
    print_signal_update(latest_bar, current_price, current_position)
    
    # Print summary statistics
    buy_signals = signals_df['Buy'].sum()
    sell_signals = signals_df['Sell'].sum()
    
    print(f"{'=' * 60}")
    logger.info(f"SIGNAL SUMMARY")
    logger.info(f"Total bars     : {len(signals_df)}")
    logger.info(f"Buy signals    : {buy_signals} ({buy_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"Sell signals   : {sell_signals} ({sell_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"No signals     : {len(signals_df) - buy_signals - sell_signals} ({(len(signals_df) - buy_signals - sell_signals)/len(signals_df)*100:.2f}%)")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    #run_signal_test()  # Use this to test signals without trading
    run_live_trading()  # Use this for actual tradingimport pandas as pd
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
ALPACA_API_KEY = "PKG6R1CKY9ZAZJLWNUIU"
ALPACA_SECRET_KEY = "Uj6hrZzkDpep7wplMt2LEoWhOAizDBjfTywoEZ3h"

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
    
    logger.info(f"Fetched {len(df)} days historical bars for {TICKER}")
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
            if position.symbol == "BTC/USD":  # Hard-coded symbol format for Alpaca
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
            symbol="BTC/USD",  # Hard-coded correct format for Alpaca
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        # Submit order
        order = trading_client.submit_order(order_data)
        
        # Print detailed trade information with better formatting
        print(f"{'=' * 60}")
        logger.info(f"TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Type           : {side.name}")
        logger.info(f"Symbol         : {TICKER}")
        logger.info(f"Quantity       : {quantity:.6f}")
        logger.info(f"Price          : ${price:.2f}")
        logger.info(f"Total Value    : ${TRADE_AMOUNT:.2f}")
        logger.info(f"Order ID       : {order.id}")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_detailed_signals(signals_df):
    """Print detailed information for each bar in the dataframe with all signals"""
    print(f"{'=' * 60}")
    logger.info(f"DETAILED SIGNAL ANALYSIS FOR ALL {len(signals_df)} BARS")
    print("\n")

    # Define headers with fixed spacing for better alignment
    header = (
        f"{'Date':<12} | "
        f"{'Open':>10} | "
        f"{'High':>10} | "
        f"{'Low':>10} | "
        f"{'Close':>10} | "
        f"{'ATR':>8} | "
        f"{'ATR Stop':>10} | "
        f"{'Above':^7} | "
        f"{'Below':^7} | "
        f"{'Buy':^7} | "
        f"{'Sell':^7} | "
        f"{'Signal':<10}"
    )
    
    logger.info(header)
    logger.info(f"{'-' * 60}")
    
    # Print each row with proper formatting
    for i, row in signals_df.iterrows():
        # Format date (keep only the date part if it's a datetime object)
        if isinstance(row['Date'], pd.Timestamp):
            date_str = row['Date'].strftime('%Y-%m-%d')
        else:
            date_str = str(row['Date'])[:10]
        
        # Determine signal color and text
        if row['Buy']:
            signal_color = Fore.CYAN
            signal_text = "BUY 🔵"
        elif row['Sell']:
            signal_color = Fore.MAGENTA
            signal_text = "SELL 🔴"
        else:
            signal_color = Fore.YELLOW
            signal_text = "NONE ⚪"
        
        # Format row data
        row_data = (
            f"{date_str:<12} | "
            f"${row['Open']:>9.2f} | "
            f"${row['High']:>9.2f} | "
            f"${row['Low']:>9.2f} | "
            f"${row['Close']:>9.2f} | "
            f"{row['xATR']:>7.2f} | "
            f"${row['ATRTrailingStop']:>9.2f} | "
            f"{str(row['Above']):^7} | "
            f"{str(row['Below']):^7} | "
            f"{str(row['Buy']):^7} | "
            f"{str(row['Sell']):^7} | "
            f"{signal_color}{signal_text}{Style.RESET_ALL}"
        )
        
        logger.info(row_data)
    
    print(f"{'=' * 60}")

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
    
    # Log bot configuration with better alignment
    logger.info(f"STRATEGY       : UT Bot with ATR Trailing Stop")
    logger.info(f"SYMBOL         : {TICKER}")
    logger.info(f"TIMEFRAME      : {INTERVAL}")
    logger.info(f"SENSITIVITY    : {SENSITIVITY}")
    logger.info(f"ATR PERIOD     : {ATR_PERIOD}")
    logger.info(f"TRADE AMOUNT   : ${TRADE_AMOUNT}")
    logger.info(f"TRADING MODE   : Paper Trading (Alpaca)")

def print_signal_update(latest_bar, current_price, current_position):
    """Print a nicely formatted signal update"""
    print(f"{'=' * 60}")
    logger.info(f"DAILY SIGNAL UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Date           : {latest_bar['Date']}")
    logger.info(f"Price          : ${current_price:.2f}")
    logger.info(f"ATR Stop       : ${latest_bar['ATRTrailingStop']:.2f}")
    logger.info(f"Position       : {current_position} {TICKER.split('/')[0]}")
    
    # Use different colored signals based on buy/sell
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}SIGNAL         : BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}SIGNAL         : SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}SIGNAL         : NO SIGNAL ⚪{Style.RESET_ALL}")

def print_waiting_message(next_check_time):
    """Print a clear waiting message to show bot is active and waiting for next check"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}BOT STATUS      : MONITORING {TICKER}{Style.RESET_ALL}")
    logger.info(f"Current Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Next Check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate time remaining
    time_remaining = next_check_time - datetime.now()
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Time Remaining : {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    if time_remaining.days > 0:
        logger.info(f"Days Remaining : {time_remaining.days}")
        
    # Show current market status
    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5  # 5 Saturday, 6 Sunday
    
    if is_weekend:
        logger.info(f"{Fore.YELLOW}Market Status  : Weekend - Limited Trading Activity{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.GREEN}Market Status  : Weekday - Normal Trading Hours{Style.RESET_ALL}")
        
    print(f"{'=' * 60}")

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
                
                # Print detailed signals for all bars (new function)
                # print_detailed_signals(signals_df)
                
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
                
                # Calculate next check time
                next_check_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
                # logger.info(f"Next check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Print waiting message every hour
            next_check_time = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
            if last_check_date == current_date:  # Only show waiting message after today's check is complete
                print_waiting_message(next_check_time)
            
            # Sleep for 1 hour before checking again
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in live trading execution: {e}")
            logger.error(f"{'=' * 60}")
            # Sleep for 5 minutes before retrying
            time.sleep(300)

def run_signal_test():
    """Run a test to print all historical signals without executing trades"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    logger.info("RUNNING IN SIGNAL TEST MODE - NO TRADES WILL BE EXECUTED")
    
    # Get historical data
    df = get_historical_data()
    
    # Calculate signals
    signals_df = calculate_signals(df)
    
    # Print detailed signals for all bars
    print_detailed_signals(signals_df)
    
    # Check latest signal (most recent bar)
    latest_bar = signals_df.iloc[-1]
    current_price = latest_bar['Close']
    
    # Get current position
    current_position = get_current_position()
    
    # Log detailed signal information with better formatting
    print_signal_update(latest_bar, current_price, current_position)
    
    # Print summary statistics
    buy_signals = signals_df['Buy'].sum()
    sell_signals = signals_df['Sell'].sum()
    
    print(f"{'=' * 60}")
    logger.info(f"SIGNAL SUMMARY")
    logger.info(f"Total bars     : {len(signals_df)}")
    logger.info(f"Buy signals    : {buy_signals} ({buy_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"Sell signals   : {sell_signals} ({sell_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"No signals     : {len(signals_df) - buy_signals - sell_signals} ({(len(signals_df) - buy_signals - sell_signals)/len(signals_df)*100:.2f}%)")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    #run_signal_test()  # Use this to test signals without trading
    run_live_trading()  # Use this for actual trading

