import numpy as np
import ccxt
import time

# Extracts and saves ohlcv datasets to txt file if ran as main.

# Extracts ohlcv dataset from binance using ccxt
def extract_ohlcv_binance(pair='BTC/USDT', interval='1m', display=False):
    # pair = trading pair (e.g. 'BTC/USDT', 'ETH/USDT', 'BCH/USDT', 'BNB/USDT', 'ADA/USDT')
    # interval = time interval between data points (e.g. '1m', '1h', '1d', '1w', '1M')
    # display = display progress

    # List of ids for all supported exchanges
    # exchange_list = ccxt.exchanges

    # Load exchange and markets
    exchange = ccxt.binance()
    markets = exchange.load_markets()

    # Available trading pairs
    symbols = exchange.symbols
    if not pair in symbols:
        print('Error: Pair not supported.')
        return 0

    # OHLCV (open-highest-lowest-closing-volume) history
    if not exchange.has['fetchOHLCV']:
        print('Error: Cannot fetch OHLCV data.')
        return 0
        
    # Time interval
    tframes = exchange.timeframes
    if not interval in tframes:
        print('Error: Interval not supported')
        return 0
    time_interval = tframes[interval] # Time interval between data points

    # Build dataset
    data = exchange.fetch_ohlcv(pair, interval, since=0) # Initial data
    data_array = np.array(data)

    dataset = np.empty((0,data_array.shape[1])) # Dataset to be built

    # While there is still data to be extracted
    # Note: Can only extract ~1000 data points per fetch_ohlcv call.
    while data:
        if display:
            print(str(len(dataset) + len(data)) + ' data points extracted...')
            
        # Add extracted data to dataset
        dataset = np.vstack([dataset,data_array])
        
        # Timestamp of the last entry
        last_time = int(data_array[-1][0]) 
    
        # Extract next sequence of data
        data = exchange.fetch_ohlcv(pair, interval, since=last_time+1)
        data_array = np.array(data)
        
    return dataset
        
if __name__ == '__main__':
    # 5 highest volume pairs w/ USDT
    # Bitcoin, Ether, Bitcoin Cash, Binance Coin, Cardano
    pair_list = ['BTC/USDT', 'ETH/USDT', 'BCH/USDT', 'BNB/USDT', 'ADA/USDT']
    interval = '1m'
    
    t_1 = time.time()
    for pair in pair_list:
        # Extract dataset
        dataset = extract_ohlcv_binance(pair, interval, False)
        
        print(pair + ' extracted. # of datapoints = ' + str(len(dataset)))
        
        # Save to txt file
        fname = pair.replace('/','-') + '_' + interval + '.txt'
        np.savetxt(fname, dataset)
        
        t_2 = time.time()

        print(pair + ' saved. Runtime = ' + str(t_2-t_1) + ' s')
        t_1 = t_2