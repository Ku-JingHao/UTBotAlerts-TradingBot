import pandas as pd
import numpy as np
import talib
import vectorbt as vbt
import time
import logging
import colorama
import keyboard
import yfinance as yf
from datetime import datetime, timedelta
from colorama import Fore, Back, Style
import pywhatkit as kit

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# WhatsApp notification settings
ENABLE_WHATSAPP_ALERTS = True
WHATSAPP_PHONE_NUMBERS = ["+601169525122", "+60133996236", "+60193617647", "+60132042088", "+60122072556"]

# Trading Parameters
TICKER = "CRV-USD"  # Yahoo Finance format for Curve DAO Token
INTERVAL = "1h"

# UT Bot Parameters
SENSITIVITY = 1
ATR_PERIOD = 10

def send_whatsapp_alert(message):
    """Send a WhatsApp alert message to multiple phone numbers"""
    if not ENABLE_WHATSAPP_ALERTS:
        logger.info("WhatsApp alerts are disabled. Not sending message.")
        return False
    
    success = True
    for phone_number in WHATSAPP_PHONE_NUMBERS:
        try:
            logger.info(f"Sending WhatsApp alert to {phone_number}")
            kit.sendwhatmsg_instantly(
                phone_number, 
                message,
                wait_time=40,
                tab_close=True,
                close_time=10
            )
            logger.info(f"WhatsApp alert sent to {phone_number}")
            time.sleep(15)  # Delay between messages
        except Exception as e:
            logger.error(f"Failed to send WhatsApp alert to {phone_number}: {str(e)}")
            success = False
    return success

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for log messages"""
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

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler("ut_bot_alerts.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_historical_data():
    """Fetch historical data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(TICKER)
        df = ticker.history(period="10d", interval="1h")
        
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        logger.info(f"Fetched {len(df)} hour bars for {TICKER}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

def calculate_signals(pd_data):
    """Calculate UT Bot signals based on ATR Trailing Stop"""
    pd_data["xATR"] = talib.ATR(pd_data["High"], pd_data["Low"], pd_data["Close"], timeperiod=ATR_PERIOD)
    pd_data["nLoss"] = SENSITIVITY * pd_data["xATR"]
    
    pd_data = pd_data.dropna()
    pd_data = pd_data.reset_index(drop=True)
    
    pd_data["ATRTrailingStop"] = [0.0] + [np.nan for i in range(len(pd_data) - 1)]
    
    for i in range(1, len(pd_data)):
        pd_data.loc[i, "ATRTrailingStop"] = xATRTrailingStop_func(
            pd_data.loc[i, "Close"],
            pd_data.loc[i - 1, "Close"],
            pd_data.loc[i - 1, "ATRTrailingStop"],
            pd_data.loc[i, "nLoss"]
        )
    
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

def print_bot_header():
    """Print a nicely formatted header when the bot starts"""
    header = f"""
{'=' * 60}
{Fore.CYAN}
  _   _ _____   ____        _     _____ _                 _       
 | | | |_   _| | __ )  ___ | |_  |  ___(_)_ __ __ _ _ __| |_ ___ 
 | | | | | |   |  _ \\ / _ \\| __| | |_  | | '__/ _` | '__| __/ _ \\
 | |_| | | |   | |_) | (_) | |_  |  _| | | | | (_| | |  | ||  __/
  \\___/  |_|   |____/ \\___/ \\__| |_|   |_|_|  \\__,_|_|   \\__\\___|
{Style.RESET_ALL}
{'=' * 60}
"""
    print(header)
    logger.info(f"STRATEGY       : UT Bot with ATR Trailing Stop")
    logger.info(f"SYMBOL         : {TICKER}")
    logger.info(f"TIMEFRAME      : {INTERVAL}")
    logger.info(f"SENSITIVITY    : {SENSITIVITY}")
    logger.info(f"ATR PERIOD     : {ATR_PERIOD}")
    logger.info(f"MODE           : Signal Alerts Only")

def print_signal_update(latest_bar, current_price, signal_count=1):
    """Print a nicely formatted signal update"""
    print(f"{'=' * 60}")
    logger.info(f"1-Hour SIGNAL UPDATE #{signal_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if isinstance(latest_bar['Date'], pd.Timestamp):
        date_str = latest_bar['Date'].strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_str = str(latest_bar['Date'])
    
    logger.info(f"Date Time      : {date_str}")
    logger.info(f"Price          : ${current_price:.4f}")
    logger.info(f"ATR Stop       : ${latest_bar['ATRTrailingStop']:.4f}")
    
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}SIGNAL         : BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}SIGNAL         : SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}SIGNAL         : NO SIGNAL ⚪{Style.RESET_ALL}")

def print_waiting_message(next_check_time):
    """Print waiting message"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}BOT STATUS      : MONITORING {TICKER}{Style.RESET_ALL}")
    logger.info(f"Current Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Next Check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    time_remaining = next_check_time - datetime.now()
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Time Remaining : {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'=' * 60}")

def run_live_trading():
    """Run the UT Bot strategy continuously in alert mode"""
    global logger
    logger = setup_logging()
    
    print_bot_header()
    
    last_check_hour = None
    signal_count = 0
    waiting_message_shown = False
    last_signal = None

    # if ENABLE_WHATSAPP_ALERTS:
    #     startup_message = (
    #         f"UT Bot Signal Alerts Started ⚙️\n"
    #         f"Strategy: UT Bot with ATR Trailing Stop\n"
    #         f"Symbol: {TICKER}\n"
    #         f"Timeframe: {INTERVAL}\n"
    #         f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    #     )
    #     send_whatsapp_alert(startup_message)

    try:
        while True:
            try:
                if keyboard.is_pressed('q'):
                    logger.info("Bot stopped by user")
                    break
                
                current_datetime = datetime.now()
                current_hour = current_datetime.replace(minute=0, second=0, microsecond=0)

                if last_check_hour != current_hour:
                    waiting_message_shown = False
                    
                    df = get_historical_data()
                    if df.empty:
                        logger.error("No data received, skipping this update")
                        continue
                        
                    signals_df = calculate_signals(df)
                    latest_bar = signals_df.iloc[-1]
                    current_price = latest_bar['Close']

                    signal_count += 1
                    print_signal_update(latest_bar, current_price, signal_count)
                    
                    current_signal = None
                    if latest_bar['Buy']:
                        current_signal = "BUY"
                    elif latest_bar['Sell']:
                        current_signal = "SELL"
                        
                    if current_signal and current_signal != last_signal:
                        signal_type = "BUY 🟢" if current_signal == "BUY" else "SELL 🔴"
                        alert_message = (
                            f"UT Bot Signal Alert!\n"
                            f"Signal: {signal_type}\n"
                            f"Symbol: {TICKER}\n"
                            f"Price: ${current_price:.4f}\n"
                            f"ATR Stop: ${latest_bar['ATRTrailingStop']:.4f}\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        send_whatsapp_alert(alert_message)
                        last_signal = current_signal
                    
                    last_check_hour = current_hour
                    
                next_check_time = current_datetime + timedelta(hours=1)
                
                if last_check_hour == current_hour and not waiting_message_shown:
                    print_waiting_message(next_check_time)
                    waiting_message_shown = True
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in signal monitoring: {e}")
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")

if __name__ == "__main__":
    run_live_trading()