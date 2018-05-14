# Extract historical ohlcv data from coinmarketcap
# Default address is for ether

# Standard
import numpy as np

# html scraping
from lxml import html # Consult https://msdn.microsoft.com/en-us/library/ms256086(v=vs.110).aspx
import requests

def extract_data(address='https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end=20180514'):
    # Scrape page
    page = requests.get(address)
    tree = html.fromstring(page.content)

    # Extract dates
    # dates = tree.xpath('//td[@class="text-left"]/text()') # Get all td elements with attribute class = "text-left" and extract the text nodes from them.
    dates = tree.xpath('//tr/td[1]/text()') # Get all tr elements and take their first td elements and extract the text nodes.

    # Extract open, high, low, close, volume, market cap
    open_list = tree.xpath('//tr/td[2]/text()')
    high_list = tree.xpath('//tr/td[3]/text()')
    low_list = tree.xpath('//tr/td[4]/text()')
    close_list = tree.xpath('//tr/td[5]/text()')
    vol_list = tree.xpath('//tr/td[6]/text()')
    cap_list = tree.xpath('//tr/td[7]/text()')

    # Convert to arrays
    open_array = np.array([float(x.replace(',','')) for x in open_list])[::-1]
    high_array = np.array([float(x.replace(',','')) for x in high_list])[::-1]
    low_array = np.array([float(x.replace(',','')) for x in low_list])[::-1]
    close_array = np.array([float(x.replace(',','')) for x in close_list])[::-1]
    vol_array = np.array([float(x.replace(',','')) for x in vol_list])[::-1]
    # Caps must be treated specially because the last entry is '-'
    cap_array = [float(x.replace(',','').replace('-','0')) for x in cap_list]
    cap_array[-1] = cap_array[-2]
    cap_array = np.array(cap_array)[::-1]

    return open_array,high_array,low_array,close_array,vol_array,cap_array
