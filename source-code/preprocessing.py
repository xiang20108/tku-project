# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib as ta


print('Read order book csv file ... ', end='')
df_order = pd.read_csv('csv/LegacyMarketDepthTXH1.csv', index_col='Date-Time')
df_order.index = pd.to_datetime(df_order.index)


print('Done\nSelect unwanted columns ... ', end='')
removal = set(df_order.columns)
removal -= set(['Date-Time'] + [f'L{n}-{name}' for name in ['BidPrice', 'BidSize', 'AskPrice', 'AskSize'] for n in [1, 2]])


print('Done\nRemove unwanted columns ... ', end='')
df_order.drop(list(removal), axis=1, inplace=True)


print('Done\nDrop NA ... ', end='')
df_order.dropna(inplace=True)


print('Done\nRead trade book csv file ... ', end='')
df_trade = pd.read_csv('csv/TimeAndSalesReportTXH1.csv', index_col = 'Date-Time')
df_trade.index = pd.to_datetime(df_trade.index)


print('Done\nSelect unwanted columns ... ', end='')
removal = set(df_trade.columns)
removal -= set(["Date-Time"] + ['Price', 'Volume'])


print('Done\nRemove unwanted columns ... ', end='')
df_trade.drop(list(removal), axis=1, inplace=True)
del removal


print('Done\nDrop NA ... ', end='')
df_trade.dropna(inplace=True)


print('Done\nMerge two dataframe ... ', end='')
df_merge = pd.merge_asof(df_order, df_trade, left_index=True, right_index=True, tolerance=pd.Timedelta('5s'))


print('Done\nResample dataframe ... ', end='')
df_merge.insert(10, column='Open', value = df_merge['Price'])
df_merge.insert(11, column='High', value = df_merge['Price'])
df_merge.insert(12, column='Low', value = df_merge['Price'])
df_merge.insert(13, column='Close', value = df_merge['Price'])
df_resample = df_merge.resample('1T').agg({
    'L1-BidPrice': 'last',
    'L1-BidSize': 'sum',
    'L1-AskPrice': 'last',
    'L1-AskSize': 'sum',
    'L2-BidPrice': 'last',
    'L2-BidSize': 'sum',
    'L2-AskPrice': 'last',
    'L2-AskSize': 'sum',
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
})


print('Done\nDrop NA ... ', end='')
df_resample.dropna(inplace = True)


print('Done\nAdd new features ... ', end='')
df_resample.insert(13, column="Bid-Ask-Spread", value = df_resample.apply(lambda x: x['L1-AskPrice'] - x['L1-BidPrice'], axis=1))
df_resample.insert(14, column="Size-Difference", value = df_resample.apply(lambda x: x['L1-BidSize'] - x['L1-AskSize'], axis=1))
df_resample.insert(15, column="slow_K", value = np.nan)
df_resample.insert(16, column="slow_D", value = np.nan)
df_resample['slow_K'], df_resample['slow_D'] = ta.STOCH(df_resample['High'],df_resample['Low'],df_resample['Close'])
df_resample.insert(17, 'sma_bidprice_10', ta.SMA(df_resample['L1-BidPrice'], 10))
df_resample.insert(18, 'sma_askprice_10', ta.SMA(df_resample['L1-AskPrice'],10))
df_resample.insert(19, 'sma_close_10', ta.SMA(df_resample['Close'],10))


print('Done\nDrop NA ... ', end='')
df_resample.dropna(inplace = True)


print('Done\nCreate label...', end='')
df_resample.insert(0, 'label', np.nan)
df_resample['label'] = np.where(df_resample['L1-BidPrice'].shift(-9) > df_resample['L1-AskPrice'],
                    1,np.where(df_resample['L1-BidPrice'].shift(-9) < df_resample['L1-AskPrice'], -1, 0))


df_resample.insert(1, 'label_sma', np.nan)
df_resample['label_sma'] = np.where(df_resample['sma_bidprice_10'].shift(-9) > df_resample['L1-AskPrice'],
                    1,np.where(df_resample['sma_bidprice_10'].shift(-9) < df_resample['L1-AskPrice'], -1, 0)) 


print('Done\nDrop tail 9 ... ', end='')
df_resample.drop(df_resample.tail(9).index, inplace = True)


print('Done\nOutput to csv file ...', end='')
df_resample.to_csv('output/training_data.csv')


print('Done\n\nAll finished.')