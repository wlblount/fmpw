#     https://financialmodelingprep.com/developer/docs
from pandas.tseries.offsets import *
from fmp import fmp_close, fmp_prof, fmp_profF, fmp_search, fmp_priceLoop
import utils 
import time
import certifi
import ssl
ssl_context = ssl.create_default_context(cafile=certifi.where())
import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import json
import re
import string
import urllib.parse
import logging
import pytz
logging.captureWarnings(True)
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen   
from datetime import datetime, time 
from tqdm import notebook, tqdm    #ex: for i in notebook.tqdm(range(1,100000000)):
from requests.utils import requote_uri
from sklearn.preprocessing import StandardScaler

import os

# Get the API key from the environment variable
apikey = os.getenv('FMP_API_KEY')

# Check if the API key is set
if not apikey:
    raise ValueError("API key not found. Please set the environment variable 'FMP_API_KEY'.")



#-----------------------------------------------------  

def fmpw_quote(syms, facs=['name', 'price', 'change', 'pctChng', 'volume', 'avgVolume',
                           'vol_pct', 'mcap(Mil)', 'timestamp', 'earnings']):   
    '''
    input: single str: 'SPY' or multi-symbol list: ['SPY','IWM'].
    returns a DataFrame: 'symbol' as index and columns: 'name', 'price', 'change', 'pctChng',
             'volume', 'avgVolume', 'vol_pct', 'timestamp', 'earnings'
    '''
    if isinstance(syms, str):
        syms = syms
    else:    
        syms = tuple(syms)
        syms = ','.join(syms)

    urlf = f'https://financialmodelingprep.com/api/v3/quote/{syms}?apikey={apikey}'
    response = urlopen(urlf, context=ssl_context)
    data = response.read().decode("utf-8")
    px = pd.DataFrame(json.loads(data))
    px = px.set_index('symbol')

    # Convert Unix timestamp to UTC, then to EDT
    eastern_tz = pytz.timezone('America/New_York')
    px['timestamp'] = px['timestamp'].apply(
        lambda x: 'Na' if pd.isna(x) else dt.datetime.fromtimestamp(x, tz=pytz.UTC).astimezone(eastern_tz)
    )
    px['earningsAnnouncement'] = px['earningsAnnouncement'].apply(lambda x: 'Na' if pd.isna(x) else x[:-12])
    px['changesPercentage'] = px['changesPercentage'].apply(lambda x: 'Na' if pd.isna(x) else round(x, 2))
    
    # Calculate 'vol_pct' and handle division by zero
    px['vol_pct'] = px.apply(lambda row: 'Na' if pd.isna(row['volume']) or pd.isna(row['avgVolume']) or row['avgVolume'] == 0
                             else round(row['volume'] / row['avgVolume'] * 100, 2), axis=1)

    # Calculate 'marketCap' in millions
    px['marketCap'] = px['marketCap'].apply(lambda x: 'Na' if pd.isna(x) else round(x / 1000000, 0))

    # Rename columns
    px.rename(columns={'changesPercentage': 'pctChng', 'earningsAnnouncement': 'earnings', 'marketCap': 'mcap(Mil)'}, inplace=True)
    px.sort_values('pctChng', ascending=False, inplace=True)

    return px[facs]

#--------------------------------------------------------------------------------------------
def fmpw_quotePost(symbol: str) -> dict:
    
    url = f'https://financialmodelingprep.com/api/v4/pre-post-market/{symbol}?apikey={apikey}'
    
    with urlopen(url, context=ssl_context) as response:
        raw_data = response.read()
        data = json.loads(raw_data)

    # Replace the timestamp with a readable datetime string
    if 'timestamp' in data:
        try:
            ts_seconds = data['timestamp'] / 1000  # Convert ms to s
            data['timestamp'] = datetime.fromtimestamp(ts_seconds).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            data['timestamp'] = f'Error: {e}'

    return data
#-------------------------------------------------------------------------------------------

def fmpw_rt(sym, simple=True):
    """
    Fetch stock quote data from Financial Modeling Prep API for one or more symbols.
    
    Args:
        sym (str or list): Single stock symbol (e.g., 'AAPL') or list of symbols (e.g., ['X', 'F', 'BAC'])
        simple (bool, optional): If True, returns a simplified dictionary with only symbol, price,
                                changesPercentage, and change. Defaults to False.
    
    Returns:
        list or dict: 
            - If simple=False and a single symbol is provided, returns a dictionary containing full quote data.
            - If simple=False and multiple symbols are provided, returns a list of dictionaries with full quote data.
            - If simple=True, returns a dictionary (single symbol) or list of dictionaries (multiple symbols)
              with only: symbol (str), price (float), changesPercentage (float), change (float).
            Full quote data includes:
                - symbol (str): Stock symbol
                - name (str): Company name
                - price (float): Current price
                - changesPercentage (float): Percentage change
                - change (float): Absolute change
                - dayLow (float): Day's low price
                - dayHigh (float): Day's high price
                - yearHigh (float): 52-week high price
                - yearLow (float): 52-week low price
                - marketCap (int): Market capitalization
                - priceAvg50 (float): 50-day moving average price
                - priceAvg200 (float): 200-day moving average price
                - exchange (str): Stock exchange
                - volume (int): Current trading volume
                - avgVolume (int): Average trading volume
                - open (float): Opening price
                - previousClose (float): Previous closing price
                - eps (float): Earnings per share
                - pe (float): Price-to-earnings ratio
                - earningsAnnouncement (str): Date of next earnings announcement
                - sharesOutstanding (int): Number of shares outstanding
                - timestamp (str): Formatted date and time of data (from Unix timestamp)
            Returns None or an empty list if data is unavailable
    
    Example:
        >>> fmpw_rt('AAPL', simple=False)
        {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'price': 145.775,
            'changesPercentage': 0.32,
            'change': 0.465,
            'dayLow': 143.9,
            'dayHigh': 146.71,
            'yearHigh': 179.61,
            'yearLow': 124.17,
            'marketCap': 2306437439846,
            'priceAvg50': 140.8724,
            'priceAvg200': 147.18594,
            'exchange': 'NASDAQ',
            'volume': 42478176,
            'avgVolume': 73638864,
            'open': 144.38,
            'previousClose': 145.31,
            'eps': 5.89,
            'pe': 24.75,
            'earningsAnnouncement': '2023-04-26T10:59:00.000+0000',
            'sharesOutstanding': 15821899776,
            'timestamp': '2023-03-02 15:32:53'
        }
        >>> fmpw_rt('AAPL', simple=True)
        {
            'symbol': 'AAPL',
            'price': 145.775,
            'changesPercentage': 0.32,
            'change': 0.465
        }
        >>> fmpw_rt(['X', 'F', 'BAC'], simple=True)
        [
            {'symbol': 'X', 'price': 38.75, 'changesPercentage': 1.23, 'change': 0.47},
            {'symbol': 'F', 'price': 12.34, 'changesPercentage': -0.56, 'change': -0.07},
            {'symbol': 'BAC', 'price': 35.89, 'changesPercentage': 0.89, 'change': 0.32}
        ]
    
    Raises:
        ValueError: If FMP_API_KEY environment variable is not set
        KeyError: If required data fields are missing in the API response
    """
    # Ensure API key is set
    apikey = os.environ.get('FMP_API_KEY')
    if not apikey:
        raise ValueError("FMP_API_KEY environment variable is not set")

    # Handle single symbol or list of symbols
    symbols = sym if isinstance(sym, str) else ','.join(sym)  # Convert list to comma-separated string

    # New endpoint URL
    url = f'https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={apikey}'
    try:
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        result = json.loads(data)
        
        if not result:
            print(f"No data available for symbols {symbols}")
            return [] if isinstance(sym, list) else None
        
        # Process each quote in the result
        quotes = []
        for item in result:
            timestamp = datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            if simple:
                quote = {
                    'symbol': item.get('symbol'),
                    'price': item.get('price'),
                    'change': item.get('change'),
                    'changesPercentage': item.get('changesPercentage'),
                    'volume': item.get('volume'),
                    'timestamp': timestamp
                }
            else:
                quote = {
                    'symbol': item.get('symbol'),
                    'name': item.get('name'),
                    'price': item.get('price'),
                    'changesPercentage': item.get('changesPercentage'),
                    'change': item.get('change'),
                    'dayLow': item.get('dayLow'),
                    'dayHigh': item.get('dayHigh'),
                    'yearHigh': item.get('yearHigh'),
                    'yearLow': item.get('yearLow'),
                    'marketCap': item.get('marketCap'),
                    'priceAvg50': item.get('priceAvg50'),
                    'priceAvg200': item.get('priceAvg200'),
                    'exchange': item.get('exchange'),
                    'volume': item.get('volume'),
                    'avgVolume': item.get('avgVolume'),
                    'open': item.get('open'),
                    'previousClose': item.get('previousClose'),
                    'eps': item.get('eps'),
                    'pe': item.get('pe'),
                    'earningsAnnouncement': item.get('earningsAnnouncement'),
                    'sharesOutstanding': item.get('sharesOutstanding'),
                    'timestamp': timestamp
                }
            quotes.append(quote)
        
        # Return single dictionary if one symbol, list if multiple
        return quotes[0] if isinstance(sym, str) and len(quotes) == 1 else quotes
    except (KeyError, TypeError, ValueError, IndexError) as e:
        print(f"Error processing data for {symbols}: {str(e)}")
        return [] if isinstance(sym, list) else None
    except Exception as e:
        print(f"Unexpected error fetching data for {symbols}: {str(e)}")
        return [] if isinstance(sym, list) else None
#-----------------------------------------------------------
def fmpw_rtMult(symbols):
    """Retrieve stock data for multiple symbols and return as a DataFrame.
    
    Args:
        symbols (list): List of stock symbols as strings (e.g., ['FXE', 'LXS.DE']).
    
    Returns:
       pandas.DataFrame: DataFrame containing stock data with columns:
            - Symbol: Stock ticker symbol
            - Price: Current price
            - Change: Price change
            - Return: Percentage return 
    
    Raises:
        Exception: If fmpw_rt() function fails for any symbol, NaN values 
                  will be used for that row and an error message will be printed.
    
    Example:
        >>> symbols = ['FXE', 'LXS.DE', 'BAS.DE']
        >>> df = get_stock_data(symbols)
        >>> print(df)
           Symbol  Price  Change  Return
        0    FXE  31.92    4.23   15.28
        1 LXS.DE  25.50    2.15   12.30
        2 BAS.DE  45.20   -1.25    8.75
    """
    # Create empty lists to store the data
    prices = []
    changes = []
    returns = []
    symbol_list = []
    
    # Loop through each symbol
    for symbol in symbols:
        try:
            # Get data from your function
            result = fmpw_rt(symbol)
            
            # Append data to lists
            symbol_list.append(symbol)
            prices.append(result['price'])
            changes.append(result['chg'])
            returns.append(result['ret'])
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            # Append NaN values if there's an error
            symbol_list.append(symbol)
            prices.append(np.nan)
            changes.append(np.nan)
            returns.append(np.nan)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Symbol': symbol_list,
        'Price': prices,
        'Change': changes,
        'Return': returns
    })
    
    return df
#----------------------------------------------------------------------------------------------
def fmpw_lbkClose(sym, lbk=15):

    if not isinstance(sym, str):
        sym = sym[0]
      
    try:
        date=utils.ddelt(lbk)

        urlpc='https://financialmodelingprep.com/api/v3/historical-price-full/'+sym+'?from='+date+'&to='+date+'&serietype=line&apikey=deb84eb89cd5f862f8f3216ea4d44719'    
        url = urlpc
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff=json.loads(data)   

        [l]=stuff['historical']   #for single symbol
        
    except KeyError:
        date=utils.ddelt(lbk+1)

        urlpc='https://financialmodelingprep.com/api/v3/historical-price-full/'+sym+'?from='+date+'&to='+date+'&serietype=line&apikey=deb84eb89cd5f862f8f3216ea4d44719'    
        url = urlpc
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff=json.loads(data)   

        [l]=stuff['historical']   #for single symbol
        
    #time.sleep(.04)
    return l['close']

#---------------------------------------------------------------------------------------------
#### fmp module has RSI with a flag for series or most recent.  not sure where to put this
def fmpw_rsi(sym, periods = 3):
    """
    Returns a pd.Series with the relative strength index.
    """
   
    df=fmp_price(sym, facs=['close'], start=utils.ddelt(periods+5))
    
    close_delta = df.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    

	# Use exponential moving average
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
 
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
  
    return np.round(rsi.close[-1],2)

#-------------------------------------------------------------------------------------------------

def fmpw_stoch(sym, length=8, smooth=3):
    
    df=fmp_price(sym, facs=['low', 'high', 'close'], start=utils.tdelt(length+5))
    df['highest'] = df.high.rolling(length).max()
    df['lowest'] = df.low.rolling(length).min()
    df['k'] = 100*(df.close-df.lowest) / (df.highest-df.lowest)
    df['k_smooth'] = df.k.rolling(smooth).mean()
    time.sleep(.03)
    return np.round(df.k_smooth[-1],2)

#----------------------------------------------------------------------------------------------------

def fmpw_hv(sym,lbk=63):
    '''returns the annualized hist vol of a single symbol
       lkb: n-1 days to use in the calculation'''
    df=pd.DataFrame(fmp_close(sym, lbk+1))
    #time.sleep(.03)
    return np.round(np.std(np.log(df.close/df.close.shift()), axis=0)*252**.5*100,1)

#-------------------------------------------------------------------------------------------------------


def fmpw_beta(sym, mkt='SPY', lbk=100):
    '''
    inputs:
    sym: str
    mkt: str
    lbk: int
    returns: float'''
    
    x=[sub['close'] for sub in fmp_close(mkt, lbk+1)]
    y= [sub['close'] for sub in fmp_close(sym, lbk+1)]

    df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
    df = np.log(df/df.shift()).dropna()

    cov = df.cov()
    var = df['x'].var()
    m=cov/var
    #time.sleep(.04)
    return np.round(m.iloc[0,1], 2)
    
#------------------------------------------------------------
def fmpw_returns(syms=['XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLC', 'XLRE', 'XLU', 'XLB', 'XLK', 'SPY'], 
                 days=[1, 5, 15, 42, 'YTD', 250], sort_by=15, styled=True, supress=True):
    '''
    Returns a dataframe of returns

    Inputs:
        syms: list of strings (symbols)
        days: list of ints and/or 'YTD' for lookback periods
        sort_by: element of days list for sorting
        styled: bool, True for styled DataFrame with gradient
        supress: bool, False to show progress bar and symbols
    Outputs:
        Styled DataFrame (if styled=True) or raw DataFrame with symbols as index and returns as columns
    '''
    import numpy as np
    import pandas as pd
    import utils
    from fmp import fmp_priceLoop, fmp_prof

    if sort_by not in days:
        sort_by = days[0]

    # Calculate YTD trading days
    ytd_days = utils.ytd()  # Number of trading days YTD (e.g., 42 on March 3, 2025)
    ndays = [ytd_days if x == 'YTD' else int(x) for x in days]  # Replace 'YTD' with trading days

    # Create column labels, preserving 'YTD'
    unique_ndays = sorted(set(ndays))  # Unique sorted days for calculation
    cols = []
    for d in unique_ndays:
        if d == ytd_days and 'YTD' in days:
            cols.append('YTD')
        else:
            cols.append(str(d))  # Convert others to strings

    # Fetch price data
    df = fmp_priceLoop(syms, start=utils.ddelt(max(ndays)+2), fac='close', supress=supress)

    # Calculate returns
    dff = pd.DataFrame(
        [np.round((df.iloc[-1, :] / df.iloc[-d - 1, :] - 1) * 100, 2) for d in unique_ndays], 
        index=cols
    ).T

    # Remove duplicate symbols
    dff = dff[~dff.index.duplicated(keep='first')]

    # Fetch profile data
    names = fmp_prof(dff.index.tolist())
    names['mktCap'] = names['mktCap'].map('{:,.0f}'.format)
    
    # Combine returns with profile data
    w_names = pd.concat([names, dff], axis=1)
    
    # Sort with explicit column label
    sort_by_col = 'YTD' if sort_by == 'YTD' else str(sort_by)
    sort_by_col = sort_by_col if sort_by_col in cols else next(iter(cols))  # Fallback
    w_names = w_names.sort_values(sort_by_col, ascending=False)
    
    # Return styled or raw DataFrame
    if styled:
        styled_df = dff.sort_values(sort_by_col, ascending=False).style.background_gradient('gray_r').format('{:.2f}')
        return styled_df
    else:
        return w_names


#-----------------------------------------------------------

def fmpw_returnsD(
    syms=['XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLC', 'XLRE', 'XLU', 'XLB', 'XLK', 'SPY'], 
    sort=True
    ):
    '''    ****TOO SLOW****
input:  syms as string = a list of symbol(s)
        sort as bool.  True sors by returns descending,  False keeps in the origninal
        syms list order for concatting with another df
        
returns: a series or returns today to date        
    '''

    lst=[]
    for i in syms:

        lst.append(fmpw_rt(i)['changesPercentage'])
    return pd.Series(lst, index=syms).sort_values(ascending=False)  
       
#---------------------------------------------------------------------    

def fmpw_returnsSetDF(syms=['XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLC', 'XLRE', 'XLU', 'XLB', 'XLK', 'SPY'], 
                      sortby='1D'
                     ):
    '''
    Input: a list of syms
    Parameter:  sortby= '1D',	'5D',	'1M',	'3M',	'6M',	'ytd',	'1Y','3Y', '5Y'
    Returns: companyName industry	mktCap	cik	1D	5D	1M	3M	6M	ytd	1Y 3Y 5Y
             in the form of a dataframe  '''
    names=fmp_prof(syms, facs=['companyName', 'industry',	'mktCap'])

    symsurl=tuple(syms)
    symsurl=','.join(syms)
    url='https://financialmodelingprep.com/api/v3/stock-price-change/'+symsurl+'?apikey='+apikey
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    px=pd.DataFrame(json.loads(data), columns=['symbol','1D',	'5D',	'1M',	'3M',	'6M',	'ytd',	'1Y', '3Y', '5Y']  ) 
    px.set_index('symbol', inplace=True)
    return pd.concat([names,px], axis=1).sort_values(sortby, ascending=False)
 #--------------------------------------------------------------------------------
    
def fmpw_secWeights(sym='SPY'):
    '''
    input: etf symbol as string
    output: dataframe with 11 sector weightings
    '''
    url='https://financialmodelingprep.com/api/v3/etf-sector-weightings/'+sym+'?apikey='+apikey
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data) 
    df= pd.DataFrame(stuff) 
    df.set_index('sector', inplace=True)
    return df



def fmpw_plotBarRetw(df,  title=None, save=False): 
    
    '''
input: pandas Series where symbols are the index  
output:  mpl bar graph object with green and red bars
x = symbols and y = returns
    '''
    df=df.sort_values(ascending=False)   
    colors = ['g' if value >= 0 else 'r' for value in df]


    plt.figure(figsize=(10,6))
    plt.grid(True)

    plt.bar(df.index, df, color=colors)
    plt.xticks(rotation=45)


    plt.title(title)

    if save:
        plt.savefig(title+'.png', bbox_inches='tight')    
    else:    
        plt.show()
        
#-----------------------------------------------------------------

# def fmpw_shorted():
#     '''
# returns a list of symbols from yahoo's most shorted stocks from "https://www.highshortinterest.com"
# no params needed
#     '''
#     lst = pd.read_html('https://www.highshortinterest.com/', header=None)[2].iloc[1:,0].tolist()
#     return [x for x in lst if '<' not in x]  ###removes the formating line from the list

#-----------------------------------------------------------------

def fmpw_mostShorted(_list = False):
    if _list:
        return pd.read_html('https://finance.yahoo.com/screener/predefined/most_shorted_stocks')[0].loc[:,'Symbol'].tolist()
           
    else:    
        return pd.read_html('https://finance.yahoo.com/screener/predefined/most_shorted_stocks', index_col=0)[0]

#---------------------------------------------------------------

# def fmpw_dci(sym='SPY', length=42):
#     df=fmp_price(sym, facs=['close'], start=utils.ddelt(length+5))

#     _max = df.close.rolling(length).max()[-1]
#     _min = df.close.rolling(length).min()[-1]

#     return np.round((df['close'][-1] - _min) / (_max - _min)*100,1)





#----------------------------------------------------------------

def fmpw_yield(sym):    
    d=fmp_profF(sym)
    return np.round(d['lastDiv']/d['price']*100,2)

#---------------------------------------------------------------

def fmpw_earnSym(sym):
    '''
    input: symbol as string
    returns:  next earnings date and time as a list ['02/01/2024', 'amc']
    '''
    url= f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{sym}?apikey={apikey}"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data) 
    df = pd.DataFrame(stuff)
    df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to datetime
    df.set_index('date', inplace=True)  # Set 'date' as the index

    today = pd.Timestamp.today().normalize()
    next_date = df.index[df.index > today][-1]
    next_time = df.loc[next_date, 'time']

    formatted_date = next_date.strftime('%m/%d/%Y')
    result_list = [formatted_date, next_time]
    return result_list

#---------------------------------------------------------------

def fmpw_bal(symbols, facs=None, period='quarter'):
    """
    Fetches balance sheet data for multiple symbols from Financial Modeling Prep (FMP) API
    and reshapes the data so that financial factors (facs) are the index and symbols are the columns.
    Only data from the most recent quarter is returned.

    Args:
        symbols (list of str): A list of stock symbols for which to fetch balance sheet data.
        facs (list of str, optional): A list of financial factors to include. 
        If None, a default list of factors is used.

                'date', 'symbol', 'reportedCurrency', 
                'period', 'cashAndCashEquivalents', 'shortTermInvestments',
                'cashAndShortTermInvestments', 'netReceivables', 'inventory',
                'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet',
                'goodwill', 'intangibleAssets', 
                'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets',
                'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
                'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue',
                'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
                'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
                'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities',
                'otherLiabilities', 'totalLiabilities', 'commonStock',
                'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss',
                'othertotalStockholdersEquity', 'totalStockholdersEquity',
                'totalLiabilitiesAndStockholdersEquity', 'totalInvestments',
                'totalDebt', 'netDebt', 'link', 'finalLink'  
                
        period (str, optional): The period for the data ('quarter' or 'year'). Default is 'quarter'.
    
    Returns:
        pd.DataFrame: A DataFrame where the index consists of financial factors (facs), and each 
                      column corresponds to a stock symbol. The data reflects only the most recent quarter.
    
    Example:
        >>> fmp_balts(['AAPL', 'MSFT', 'GOOGL'])
        
        This would return a DataFrame with facs (e.g., 'totalAssets', 'totalLiabilities') 
        as the index and symbols ('AAPL', 'MSFT', 'GOOGL') as the columns.
    """
    
    if facs is None:
        facs = ['date', 'symbol', 'reportedCurrency', 
                'period', 'cashAndCashEquivalents', 'shortTermInvestments',
                'cashAndShortTermInvestments', 'netReceivables', 'inventory',
                'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet',
                'goodwill', 'intangibleAssets',
                'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets',
                'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
                'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue',
                'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
                'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
                'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities',
                'otherLiabilities', 'totalLiabilities', 'commonStock',
                'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss',
                'othertotalStockholdersEquity', 'totalStockholdersEquity',
                'totalLiabilitiesAndStockholdersEquity', 'totalInvestments',
                'totalDebt', 'netDebt']

    combined_df = pd.DataFrame()

    for sym in symbols:
        sym = sym.upper()
        url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}?period={period}&limit=400&apikey={apikey}'
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff = json.loads(data)
        
        # Extract the most recent quarter's data
        most_recent_data = stuff[0]  # Assuming the first item is the most recent
        idx = facs  # facs become the index

        # Create DataFrame with facs as index and symbol as column
        df = pd.DataFrame([[most_recent_data.get(k) for k in facs]], columns=facs).T
        df.columns = [sym]  # Set the column to the symbol name
        df.index = idx  # Set facs as the index

        # Combine into the main DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df

#--------------------------------------------------------------------

def fmpw_inc(symbols, facs=None, period='quarter'):
    """
    Fetches income data for multiple symbols from Financial Modeling Prep (FMP) API
    and reshapes the data so that financial factors (facs) are the index and symbols are the columns.
    Only data from the most recent quarter/year is returned.

    Args:
        symbols (list of str): A list of stock symbols for which to fetch balance sheet data.
        facs (list of str, optional): A list of financial factors to include. 
        If None, a default list of factors is used.

        'date', 'symbol', 'reportedCurrency', 'fillingDate', 'acceptedDate',
       'period', 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
       'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
       'sellingAndMarketingExpenses',
       'sellingGeneralAndAdministrativeExpenses', 'otherExpenses',
       'operatingExpenses', 'costAndExpenses', 'interestExpense',
       'depreciationAndAmortization', 'ebitda', 'ebitdaratio',
       'operatingIncome', 'operatingIncomeRatio',
       'totalOtherIncomeExpensesNet', 'incomeBeforeTax',
       'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome',
       'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut',
       'weightedAverageShsOutDil'
                
        period (str, optional): The period for the data ('quarter' or 'year'). Default is 'quarter'.
    
    Returns:
        pd.DataFrame: A DataFrame where the index consists of financial factors (facs), and each 
                      column corresponds to a stock symbol. The data reflects only the most recent quarter.
    
    Example:
        >>> fmpw_inc(['AAPL', 'MSFT', 'GOOGL'], facs=['costAndExpenses', 'interestExpense'])
        
        This would return a DataFrame with facs (e.g., 'costAndExpenses', 'interestExpense') 
        as the index and symbols ('AAPL', 'MSFT', 'GOOGL') as the columns.
    """
    
    if facs==None:
        full=['date', 'symbol', 'reportedCurrency',
       'period', 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
       'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
       'sellingAndMarketingExpenses',
       'sellingGeneralAndAdministrativeExpenses', 'otherExpenses',
       'operatingExpenses', 'costAndExpenses', 'interestExpense',
       'depreciationAndAmortization', 'ebitda', 'ebitdaratio',
       'operatingIncome', 'operatingIncomeRatio',
       'totalOtherIncomeExpensesNet', 'incomeBeforeTax',
       'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome',
       'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut',
       'weightedAverageShsOutDil']	
        facs=full
 

    combined_df = pd.DataFrame()

    for sym in symbols:
        sym = sym.upper()
        url='https://financialmodelingprep.com/api/v3/income-statement/'+sym+'?period='+period+'&limit=400&apikey='+apikey
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff = json.loads(data)
        
        # Extract the most recent quarter's data
        most_recent_data = stuff[0]  # Assuming the first item is the most recent
        idx = facs  # facs become the index

        # Create DataFrame with facs as index and symbol as column
        df = pd.DataFrame([[most_recent_data.get(k) for k in facs]], columns=facs).T
        df.columns = [sym]  # Set the column to the symbol name
        df.index = idx  # Set facs as the index

        # Combine into the main DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df

#--------------------------------------------------------------------

def fmpw_dci(sym='SPY', length=42):
    try:
        df = fmp_price(sym, facs=['close'], start=utils.ddelt(length+5))
    except KeyError:
        return 'Na'

    _max = df.close.rolling(length).max()[-1]
    _min = df.close.rolling(length).min()[-1]

    return np.round((df['close'][-1] - _min) / (_max - _min) * 100, 1)

#---------------------------------------------------------------------------

def fmpw_mcap(symbol):
    try:
        url = f"https://financialmodelingprep.com/api/v3/market-capitalization/{symbol}?apikey={apikey}"
        response = urlopen(url, context=ssl_context)
        data = json.loads(response.read().decode("utf-8"))
        
        # Ensure the response contains valid data
        if data and len(data) > 0:
            return data[0]['marketCap']
        else:
            return None  # No data for the given symbol
    except Exception as e:
        print(f"Error fetching market cap for {symbol}: {e}")
        return None  # Return None for any error

#---------------------------------------------------------------------------------

def fmpw_keyMetricsttm(symbol, facs=['revenuePerShareTTM','netIncomePerShareTTM','operatingCashFlowPerShareTTM',
           'freeCashFlowPerShareTTM','cashPerShareTTM','bookValuePerShareTTM',
           'tangibleBookValuePerShareTTM','shareholdersEquityPerShareTTM','interestDebtPerShareTTM',
           'marketCapTTM', 'enterpriseValueTTM','peRatioTTM','priceToSalesRatioTTM',
           'pocfratioTTM','pfcfRatioTTM','pbRatioTTM','ptbRatioTTM','evToSalesTTM',
           'enterpriseValueOverEBITDATTM', 'evToOperatingCashFlowTTM', 'evToFreeCashFlowTTM',
           'earningsYieldTTM', 'freeCashFlowYieldTTM', 'debtToEquityTTM','debtToAssetsTTM',
           'netDebtToEBITDATTM', 'currentRatioTTM', 'interestCoverageTTM','incomeQualityTTM',
           'dividendYieldTTM','dividendYieldPercentageTTM','payoutRatioTTM',
           'salesGeneralAndAdministrativeToRevenueTTM','researchAndDevelopementToRevenueTTM',
           'intangiblesToTotalAssetsTTM','capexToOperatingCashFlowTTM',
           'capexToRevenueTTM','capexToDepreciationTTM','stockBasedCompensationToRevenueTTM',
           'grahamNumberTTM','roicTTM','returnOnTangibleAssetsTTM','grahamNetNetTTM',
           'workingCapitalTTM','tangibleAssetValueTTM','netCurrentAssetValueTTM',
           'investedCapitalTTM','averageReceivablesTTM','averagePayablesTTM',
           'averageInventoryTTM','daysSalesOutstandingTTM','daysPayablesOutstandingTTM',
           'daysOfInventoryOnHandTTM','receivablesTurnoverTTM','payablesTurnoverTTM',
           'inventoryTurnoverTTM','roeTTM','capexPerShareTTM','dividendPerShareTTM',
           'debtToMarketCapTTM']):
    '''
    Returns most recent ttmValues of the following metrics for a single symbol... 
    
    facs=[['revenuePerShareTTM','netIncomePerShareTTM','operatingCashFlowPerShareTTM',
           'freeCashFlowPerShareTTM','cashPerShareTTM','bookValuePerShareTTM',
           'tangibleBookValuePerShareTTM','shareholdersEquityPerShareTTM','interestDebtPerShareTTM',
           'marketCapTTM', 'enterpriseValueTTM','peRatioTTM','priceToSalesRatioTTM',
           'pocfratioTTM','pfcfRatioTTM','pbRatioTTM','ptbRatioTTM','evToSalesTTM',
           'enterpriseValueOverEBITDATTM', 'evToOperatingCashFlowTTM', 'evToFreeCashFlowTTM',
           'earningsYieldTTM', 'freeCashFlowYieldTTM', 'debtToEquityTTM','debtToAssetsTTM',
           'netDebtToEBITDATTM', 'currentRatioTTM', 'interestCoverageTTM','incomeQualityTTM',
           'dividendYieldTTM','dividendYieldPercentageTTM','payoutRatioTTM',
           'salesGeneralAndAdministrativeToRevenueTTM','researchAndDevelopementToRevenueTTM',
           'intangiblesToTotalAssetsTTM','capexToOperatingCashFlowTTM',
           'capexToRevenueTTM','capexToDepreciationTTM','stockBasedCompensationToRevenueTTM',
           'grahamNumberTTM','roicTTM','returnOnTangibleAssetsTTM','grahamNetNetTTM',
           'workingCapitalTTM','tangibleAssetValueTTM','netCurrentAssetValueTTM',
           'investedCapitalTTM','averageReceivablesTTM','averagePayablesTTM',
           'averageInventoryTTM','daysSalesOutstandingTTM','daysPayablesOutstandingTTM',
           'daysOfInventoryOnHandTTM','receivablesTurnoverTTM','payablesTurnoverTTM',
           'inventoryTurnoverTTM','roeTTM','capexPerShareTTM','dividendPerShareTTM',
           'debtToMarketCapTTM']
    '''
    
  
    
    url=f'https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?period=quarter&apikey={apikey}'
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    stuff = stuff[0]
    
    
    return pd.Series({key: value for key, value in stuff.items() if key in facs}).T
#-------------------------------------------------------------------------------------------

def fmpw_etfExposure(sym):
    """
    Fetches ETF exposure for a stock symbol.

    Parameters:
    sym (str): Stock symbol (e.g., "KNSL").

    Returns:
    pd.DataFrame: Processed DataFrame with ETF exposure details.
    """
    sym = sym.upper()
    url = f'https://financialmodelingprep.com/api/v3/etf-stock-exposure/{sym}?apikey=' + apikey
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)

    df = pd.DataFrame(stuff)  # Convert API response to DataFrame
    df = df.drop_duplicates(subset=['sharesNumber'], keep='first')

    # ✅ Fetch ETF profile data
    etf_profiles = fmp_prof(df['etfSymbol'].tolist(), facs=['companyName'])

    if isinstance(etf_profiles, pd.DataFrame) and not etf_profiles.empty:
        etf_profiles = etf_profiles.reset_index().rename(columns={'index': 'etfSymbol'})
        df = df.merge(etf_profiles[['etfSymbol', 'companyName']], on='etfSymbol', how='left')
        df.rename(columns={'companyName': 'name'}, inplace=True)
    else:
        df['name'] = np.nan  # If API fails, set all names to NaN

    # ✅ Get Market Cap safely
    mCap = fmp_profF(sym).get('mktCap', None)
    
    if mCap and mCap > 0:  # ✅ Prevents division by zero or NoneType errors
        df['mCapPct'] = np.round(df['marketValue'] / mCap * 100, 2)
        instHold = df['mCapPct'].sum()
       
        print(f'Institutional Holding for {sym}: {np.round(instHold, 1)}% of {int(round(mCap / 1_000_000, 0)):,}M market cap')
   
        # ✅ Filter and sort data
        df = df[df['mCapPct'] > 0.25]  # Correct column name
        df = df.sort_values('mCapPct', ascending=False)
        df.set_index('etfSymbol', inplace=True)
        
        return df.loc[:, ['name', 'weightPercentage', 'marketValue', 'mCapPct']]
    
    else:
        print(f"⚠️ Warning: Market Cap for {sym} not available or invalid.")
        return pd.DataFrame()  # Return empty DataFrame if mCap is invalid

#-----------------------------------------------------------------------------------------

def fmpw_cik(ciknum):

    """
    Retrieves company name and stock symbol associated with a given CIK number from Financial Modeling Prep API.

    Parameters:
    ciknum (str): The CIK number of the company (can be shorter than 10 digits, will be zero-padded).
  

    Returns:
    dict: A dictionary containing:
        - 'CIK': The zero-padded CIK number.
        - 'Name': The company name associated with the CIK.
        - 'Symbol': The stock symbol of the company, or 'N/A' if ambiguous or unavailable.
    """
    ciknum = ciknum.zfill(10)
    url = f'https://financialmodelingprep.com/api/v3/cik/{ciknum}?apikey=' + apikey
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)

    if not stuff:
        return {"error": "CIK not found"}

    company_name = stuff[0]['name']

    # Use company name to fetch associated stock symbols
    df = fmp_search(urllib.parse.quote(company_name))

    # Remove rows where index has punctuation
    df = df[~df.index.str.contains(f"[{string.punctuation}]", regex=True)]

    # If no valid symbols remain or multiple remain, set symbol to "N/A"
    if df.empty or len(df) > 1:
        symbol = "N/A"
    else:
        symbol = df.index[0]  # Assign the first (and only) valid symbol

    return {"CIK": ciknum, "Name": company_name, "Symbol": symbol}
    
#--------------------------------------------------
# MOD 1/20/26 10:54 AM
from fmp import *

def fmpw_ev(sym, share_type='current'):
    """
    Calculates the 'Live' Enterprise Value (EV) with a toggle for share count methodology.
    
    This function reconciles real-time price data with either the latest disclosed 
    outstanding shares (to match TradingView's Quote Summary) or the weighted 
    average shares from the most recent income statement.

    Formula:
        EV = (Live Price * Share Count) + Total Debt + Minority Interest - Cash & Equivalents

    Args:
        sym (str): The equity ticker symbol (e.g., 'ASTH').
        share_type (str): 
            - 'current': (Default) Uses fmp_shares(sym) to get the most recent 
              shares outstanding from filing cover pages.
            - 'weighted': Uses 'weightedAverageShsOut' from the latest quarterly 
              income statement.

    Returns:
        float: The calculated Enterprise Value.
    """
    sym = sym.upper()
    
    # 1. Get Live Price
    quote_url = f'https://financialmodelingprep.com/api/v3/quote/{sym}?apikey={apikey}'
    with urllib.request.urlopen(quote_url, context=ssl_context) as response:
        price_data = json.loads(response.read().decode("utf-8"))
        if not price_data:
            print(f"Error: No price data for {sym}")
            return None
        live_price = price_data[0]['price']
    
    # 2. Select Share Count Methodology
    if share_type == 'current':
        # Grab latest record from fmp_shares and force to scalar numeric
        raw_shares = fmp_shares(sym)
        method_label = "Current (Filing Cover)"
        if isinstance(raw_shares, (pd.DataFrame, pd.Series)):
            share_count = pd.to_numeric(raw_shares.iloc[-1], errors='coerce')
            # If it's a DataFrame, iloc[-1] might still be a Series; take first element
            if hasattr(share_count, 'iloc'): share_count = share_count.iloc[0]
        else:
            share_count = pd.to_numeric(raw_shares, errors='coerce')
    else:
        # Pull weighted average from most recent Income Statement
        inct = fmp_incts(sym, period='quarter', limit=1, facs=['weightedAverageShsOut'])
        share_count = pd.to_numeric(inct['weightedAverageShsOut'].iloc[0], errors='coerce')
        method_label = "Weighted Avg (Basic)"
    
    # 3. Get Latest Balance Sheet for Net Debt
    bs = fmp_balts(sym, period='quarter', limit=1, 
                   facs=['totalDebt', 'minorityInterest', 'cashAndShortTermInvestments'])
    
    # Force numeric to prevent sequence/type errors
    bs_numeric = bs.apply(pd.to_numeric, errors='coerce')
    
    net_debt = (bs_numeric['totalDebt'].iloc[0] + 
                bs_numeric['minorityInterest'].iloc[0] - 
                bs_numeric['cashAndShortTermInvestments'].iloc[0])
    
    # 4. Calculate Live EV
    market_cap = live_price * share_count
    enterprise_value = market_cap + net_debt
    
    # Print results formatted for easy TV comparison
    print(f"--- TV LIVE RECONCILIATION: {sym} ---")
    print(f"Methodology:    {method_label}")
    print(f"Live Price:     ${live_price:,.2f}")
    print(f"Shares Used:    {share_count/1e6:,.3f}M")
    print(f"Market Cap:     ${market_cap/1e9:,.3f}B")
    print(f"Net Debt:       ${net_debt/1e9:,.3f}B")
    print(f"-----------------------------")
    print(f"Enterprise Val: ${enterprise_value/1e9:,.3f}B")
    
    return enterprise_value
#---------------------------------------------------
# from IPython.display import display, Markdown
# from fmp import fmp_profF
# from fmpw import fmpw_inc, fmpw_entmulti

# def fmpw_descr(syms):
#     try:
#         inc_df = fmpw_inc(syms, period='year')
#     except Exception as e:
#         print(f"Error in fmpw_inc for {syms}: {e}")
#         inc_df = None

#     try:
#         ent_df = fmpw_entmulti(syms)
#     except Exception as e:
#         print(f"Error in fmpw_entmulti for {syms}: {e}")
#         ent_df = None

#     # Collect values for averages
#     evs = []
#     ev_revs = []
#     debt_caps = []
#     cash_caps = []

#     for i in syms:
#         sym = i.upper()
#         data = fmp_profF(sym)

#         # Add revenue
#         if sym in inc_df.columns and 'revenue' in inc_df.index:
#             data['revenue'] = inc_df.loc['revenue', sym]
#         else:
#             data['revenue'] = None

#         # Add enterprise-related values
#         for field in ['enterpriseValue', 'marketCapitalization', 'addTotalDebt', 'minusCashAndCashEquivalents']:
#             if sym in ent_df.index and field in ent_df.columns:
#                 data[field] = ent_df.loc[sym, field]
#             else:
#                 data[field] = None

#         # Collect values for averages
#         ev = data.get('enterpriseValue')
#         rev = data.get('revenue')
#         mcap = data.get('marketCapitalization')
#         debt = data.get('addTotalDebt')
#         cash = data.get('minusCashAndCashEquivalents')

#         if ev is not None:
#             evs.append(ev)
#         if ev and rev:
#             ev_revs.append(ev / rev)
#         if debt and mcap:
#             debt_caps.append(debt / mcap)
#         if cash and mcap:
#             cash_caps.append(cash / mcap)

#         # --- Header line 1 ---
#         mktcap_str = f"{data['marketCapitalization'] / 1_000_000_000:.3f} bil." if data.get('mktCap') else "N/A"
#         spacer = "&nbsp;" * 5
#         header1 = (
#             f"**{data['symbol']}**{spacer}"
#             f"**{data['companyName']}**{spacer}"
#             f"**{data['industry']}**{spacer}"
#             f"**MktCap: ${mktcap_str}**"
#         )

#         # --- Header line 2 ---
#         ent_val_str = f"{ev / 1_000_000_000:.3f} bil." if ev else "N/A"
#         ent_rev_str = f"{ev / rev:.2f}" if ev and rev else "N/A"
#         debt_str = f"{debt / mcap:.2f}" if debt and mcap else "N/A"
#         cash_str = f"{cash / mcap:.2f}" if cash and mcap else "N/A"

#         header2 = (
#             f"**EV: {ent_val_str}**{spacer}"
#             f"**EV/Rev: {ent_rev_str}**{spacer}"
#             f"**Debt/Equity: {debt_str}**{spacer}"
#             f"**Cash/Equity: {cash_str}**"
#         )

#         # --- Description ---
#         descr = data['description']
#         md = f"{header1}\n\n{header2}\n\n```\n{descr}\n```"
#         display(Markdown(md))

#     # --- Averages ---
#     avg_ev = sum(evs) / len(evs) if evs else None
#     avg_evrev = sum(ev_revs) / len(ev_revs) if ev_revs else None
#     avg_debt = sum(debt_caps) / len(debt_caps) if debt_caps else None
#     avg_cash = sum(cash_caps) / len(cash_caps) if cash_caps else None

#     # Format and display average header
#     avg_spacer = "&nbsp;" * 3
#     avg_ev_str = f"{avg_ev / 1_000_000_000:.3f} bil." if avg_ev else "N/A"
#     avg_evrev_str = f"{avg_evrev:.2f}" if avg_evrev else "N/A"
#     avg_debt_str = f"{avg_debt:.2f}" if avg_debt else "N/A"
#     avg_cash_str = f"{avg_cash:.2f}" if avg_cash else "N/A"

#     avg_header = (
#         f"**EV: {avg_ev_str}**{avg_spacer}"
#         f"**EV/Rev: {avg_evrev_str}**{avg_spacer}"
#         f"**Debt/Cap: {avg_debt_str}**{avg_spacer}"
#         f"**Cash/Cap: {avg_cash_str}**"
#     )

#     display(Markdown("**Averages:**"))
#     display(Markdown(avg_header))

