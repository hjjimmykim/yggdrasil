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
data = exchange.fetch_ohlcv(symb, '1d')