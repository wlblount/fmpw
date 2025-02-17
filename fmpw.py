#     https://financialmodelingprep.com/developer/docs
from pandas.tseries.offsets import *
from fmp import *
import utils 
import time
import certifi
import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import json
import re
import logging
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
                
       all other available facs are: ["dayLow", "dayHigh", "yearHigh", "yearLow", "marketCap",
       "priceAvg50", "priceAvg200", "exchange", "open", "previousClose", "eps", "pe",
       "sharesOutstanding"]
          
       .T for features in the index (1 symbol)
       could also modify to add: 'eps', 'pe', 'sharesOutstanding' which are available on the endpoint         
    '''
    if isinstance(syms, str):
        syms = syms
    else:    
        syms = tuple(syms)
        syms = ','.join(syms)

    urlf = 'https://financialmodelingprep.com/api/v3/quote/' + syms + '?apikey=' + apikey
    response = urlopen(urlf, cafile=certifi.where())
    data = response.read().decode("utf-8")
    px = pd.DataFrame(json.loads(data))
    px = px.set_index('symbol')

    # Handle errors gracefully by replacing problematic values with 'Na'
    px['timestamp'] = px['timestamp'].apply(lambda x: 'Na' if pd.isna(x) else dt.datetime.fromtimestamp(x) - dt.timedelta(hours=4))
    px['earningsAnnouncement'] = px['earningsAnnouncement'].apply(lambda x: 'Na' if pd.isna(x) else x[:-12])
    px['changesPercentage'] = px['changesPercentage'].apply(lambda x: 'Na' if pd.isna(x) else round(x, 2))
    
    # Calculate 'vol_pct' and handle division by zero
    px['vol_pct'] = px.apply(lambda row: 'Na' if pd.isna(row['volume']) or pd.isna(row['avgVolume']) or row['avgVolume'] == 0
                             else round(row['volume'] / row['avgVolume'] * 100, 2), axis=1)

    # Calculate 'marketCap' in millions and handle missing values
    px['marketCap'] = px['marketCap'].apply(lambda x: 'Na' if pd.isna(x) else round(x / 1000000, 0))

    # Rename columns
    px.rename(columns={'changesPercentage': 'pctChng', 'earningsAnnouncement': 'earnings', 'marketCap': 'mcap(Mil)'}, inplace=True)
    px.sort_values('pctChng', ascending=False, inplace=True)

    return px[facs]

#--------------------------------------------------------------------------------------------

def fmpw_rt(sym):
    from fmp import fmp_close
    #sym=sym.upper()
    rturl='https://financialmodelingprep.com/api/v3/quote-short/'+sym+'?apikey='+apikey
    response = urlopen(rturl, cafile=certifi.where())
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    #return stuff
    try:
        p = stuff[0]['price']
        ###when the market is trading prev day close is lbk=2
        pc=fmp_close(sym, 2)[0]['close']
        return {'price':p, 'chg':np.round(p-pc,2), 'ret':np.round(((p/pc)-1)*100, 2)}	
        
    except IndexError:   
        print(sym,':  No Price Data Available')
        #continue        
    return  {'price':None, 'chg':None, 'ret'  : None}  
	
#-----------------------------------------------------------

#----------------------------------------------------------------------------------------------
def fmpw_lbkClose(sym, lbk=15):

    if not isinstance(sym, str):
        sym = sym[0]
      
    try:
        date=utils.ddelt(lbk)

        urlpc='https://financialmodelingprep.com/api/v3/historical-price-full/'+sym+'?from='+date+'&to='+date+'&serietype=line&apikey=deb84eb89cd5f862f8f3216ea4d44719'    
        url = urlpc
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        stuff=json.loads(data)   

        [l]=stuff['historical']   #for single symbol
        
    except KeyError:
        date=utils.ddelt(lbk+1)

        urlpc='https://financialmodelingprep.com/api/v3/historical-price-full/'+sym+'?from='+date+'&to='+date+'&serietype=line&apikey=deb84eb89cd5f862f8f3216ea4d44719'    
        url = urlpc
        response = urlopen(url, cafile=certifi.where())
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
    returns a dataframe of returns

    inputs: symbol(s): as a list of strings
            days: as a list of ints and/or 'YTD' to get lookbacks for returns
            by: as an element of the list days for sorting purposes.
            styled:  bool.  True returns a styled dataframe with color gradient
            supress:  bool.  False creates a progress bar prints the symbols as the prices are fetched.
                      helpful for trouble shooting key (symbols) errors
    outputs: a styled dataframe when the index are the symbols and the columns are the lookback periods.
            with 1 day or intraday automatically returned

    note: dataframe is styled object with gradients based on values per column. to do calculation on dataframe... "df.data"
    '''
    import numpy as np
    from datetime import datetime, time
    import utils
    from fmp import fmp_priceLoop

    if sort_by not in days:
        sort_by = days[0]

    # Replace 'YTD' with number of days since the start of the year up to today's date
    ndays = [utils.ytd() if x == 'YTD' else x for x in days]
    ndays = [int(x) for x in ndays]  # convert days to integers
    ndays.sort()  # sort in case the user enters unsorted

#put 'YTD' back in columns for display
    cols = ['YTD' if x not in days else x for x in ndays]    

    df = fmp_priceLoop(syms, start=utils.ddelt(ndays[-1]+2), fac='close', supress=supress)
    dff = pd.DataFrame([np.round((df.iloc[-1, :] / df.iloc[-d - 1, :] - 1) * 100, 2) for d in ndays], index=cols).T
    
    names= fmp_prof(syms)
    names['mktCap'] = names['mktCap'].map('{:,.0f}'.format)  #',.0f'
    
    
    w_names = pd.concat([names, dff], axis=1)
    
    w_names= w_names.sort_values(sort_by, ascending=False)
    
    
    # Display the styled dataframe
    if styled:
        dff=dff.sort_values(sort_by, ascending=False)
        styled_df = dff.style.background_gradient('gray_r').format('{:.2f}')
        return styled_df
    
    else:
        return w_names
        
    

#-----------------------------------------------------------

def fmpw_returnsD(syms, sort=True):
    '''
input:  syms as string = a list of symbol(s)
        sort as bool.  True sors by returns descending,  False keeps in the origninal
        syms list order for concatting with another df
        
returns: a series or returns today to date        
    '''

    lst=[]
    for i in syms:

        lst.append(fmpw_rt(i)['ret'])
    return pd.Series(lst, index=syms).sort_values(ascending=False)  
       
#---------------------------------------------------------------------    

def fmpw_setReturnsDF(syms, sortby='1D'):
    '''
    Input: a list of syms
    Parameter:  sortby= '1D',	'5D',	'1M',	'3M',	'6M',	'ytd',	'1Y'
    Returns: companyName industry	mktCap	cik	1D	5D	1M	3M	6M	ytd	1Y
             in the form of a dataframe  '''
    names=fmp_prof(syms, facs=['companyName', 'industry',	'mktCap'])

    symsurl=tuple(syms)
    symsurl=','.join(syms)
    url='https://financialmodelingprep.com/api/v3/stock-price-change/'+symsurl+'?apikey='+apikey
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    px=pd.DataFrame(json.loads(data), columns=['symbol','1D',	'5D',	'1M',	'3M',	'6M',	'ytd',	'1Y']  ) 
    px.set_index('symbol', inplace=True)
    return pd.concat([names,px], axis=1).sort_values(sortby, ascending=False)
 #--------------------------------------------------------------------------------
    
def fmpw_secWeights(sym='SPY'):
    '''
    input: etf symbol as string
    output: dataframe with 11 sector weightings
    '''
    url='https://financialmodelingprep.com/api/v3/etf-sector-weightings/'+sym+'?apikey='+apikey
    response = urlopen(url, cafile=certifi.where())
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
    response = urlopen(url, cafile=certifi.where())
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
                'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets',
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
                'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets',
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
        response = urlopen(url, cafile=certifi.where())
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
        response = urlopen(url, cafile=certifi.where())
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
        response = urlopen(url, cafile=certifi.where())
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
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    stuff = stuff[0]
    
    
    return pd.Series({key: value for key, value in stuff.items() if key in facs}).T

