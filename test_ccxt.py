import numpy as np
import ccxt

# Pair we are interested in
symb = 'ETH/BTC'

# List of ids for all supported exchanges
# exchange_list = ccxt.exchanges

# Load exchange and markets
exchange = ccxt.binance()
markets = exchange.load_markets()

# Symbols (e.g. ETH/BTC)
symbols = exchange.symbols
if not symb in symbols:
    print('Pair not supported. Exiting...')
    exit()

# OHLCV (open-highest-lowest-closing-volume) history
if not exchange.has['fetchOHLCV']:
    print('Cannot fetch OHLCV data. Exiting...')
    exit()
    
tframes = exchange.timeframes
time_interval = tframes['1m'] # Time interval between data points

# Build dataset
data = exchange.fetch_ohlcv(symb, time_interval, since=0) # Initial data
data_array = np.array(data)

dataset = np.empty((0,data_array.shape[1])) # Dataset to be built

while data:
    print(str(len(dataset) + len(data)) + ' data points extracted...')
    dataset = np.vstack([dataset,data_array])
    
    last_time = int(data_array[-1][0]) # Timestamp of the last entry
    
    data = exchange.fetch_ohlcv(symb, time_interval, since=last_time+1)
    data_array = np.array(data)