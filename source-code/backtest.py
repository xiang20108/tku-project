# -*- coding: utf-8 -*-

"""
# Portfolio value
# .portfolio value = cash + position value
#     .Position value:
#         .long position : bidPrice1 * money_per_point 
#         .short postion : askPrice1 * (-money_per_point) 

#     .cash flows:(initially = initial cash)
#         .empty to long: previous askPrice1 * (-money_per_point)     0 to  1
#         .long to empty: previous bidPrice1 * money_per_point        1 to  0
#         .empty to short: previous bidPrice1 * money_per_point       1 to -1
#         .short to empty: previous askPrice1 * (-money_per_point)   -1 to  0
#         .long to short: previous bidPrice1 * money_per_point * 2    1 to -1
#         short to long: previous askPrice1 * (-money_per_point) * 2 -1 to  1
        
# .Additional fee of position change:
#     .Fixed slippage(-money_per_point)
#     .Tax + transaction fee($)
    
# .Initial cash: XXXXX NTD

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


class Backtest():
    def __init__(self, df, init_cash = 10000000):
        df = pd.DataFrame({
            'position': df['signal'][:], #label欄位
            'previous_position': df['signal'][:].shift(1), #label欄位下移一位
            'bidPrice': df['L1-BidPrice'][:],
            'askPrice': df['L1-AskPrice'][:],
            'next_bidPrice':df['L1-BidPrice'][:].shift(-1),
            'next_askPrice':df['L1-AskPrice'][:].shift(-1),       
        })
        df['previous_position'][0] = 0
        df.drop(df.tail(1).index,inplace=True)
        self.df = df
        self.init_cash = init_cash if init_cash > 10000000 else 10000000
        self.balance = self.init_cash
        self.money_per_point = 200
        self.tax = 0.00002
        self.hf = 80
        self.cost = 0
        self.number_of_transactions = 0
        self.transactions_records = []
        self.date_list = list(df.iloc[:, 0].resample('D').last().dropna().index.to_series().apply(lambda x: str(x.date())))
        
    #建立表
    def computing(self):
        self.df['position_value'] = self.df.apply(self.position_value, axis = 1)
        self.df['cashflow'] = self.df.apply(self.cashflow, axis = 1)
        self.df['cash'] = self.df.apply(self.cash, axis = 1)
        self.df['portfolio_value'] = self.df.apply(self.portfolio_value, axis = 1)
        self.df['ROI'] = self.df.apply(self.ROI, axis = 1)
        self.df['number_of_transactions'] = self.df.apply(self.NOT, axis = 1)
        
        self.Sharpe_Ratio()
        
    def show_info(self):
        for i in range(len(self.transactions_records)):
            r = self.transactions_records[i]
            print(f'[transactions {i+1}]')
            print(f'date : {r["date"]}')
            print(f'position : {int(r["previous_position"])} to {int(r["position"])}')
            print(f'bidPrice : {r["bidPrice"]}, next bidPrice : {r["next_bidPrice"]}')
            print(f'askPrice : {r["askPrice"]}, next askPrice : {r["next_askPrice"]}')
            print(f'position value : {r["position_value"]}')
            print(f'cashflow : {r["cashflow"]}')
            print(f'cash : {r["cash"]}')
            print(f'portfolio_value : {r["portfolio_value"]}')
            print(f'ROI : {r["ROI"]}', end = '\n\n')
        
        print(f'\nNumber of transactions : {self.number_of_transactions}')     
        print(f'Initial cash : {self.init_cash}')
        print(f'Final portfolio value : {self.df["portfolio_value"][-1]}')
        print(f'Sharpe Ratio : {self.sharpe_ratio}')
        
        
        
    #部位計算
    def position_value(self, df):
        if(df['position'] == 1):
            return df['bidPrice']*200
        elif(df['position'] == 0):
            return 0
        elif(df['position'] == -1):
            return df['askPrice']*-200
    #現金流    
    def cashflow(self, df):
        # empty to long: previous askPrice1 * (-money_per_point)    0 to  1
        if(df['previous_position'] == 0 and df['position'] == 1):
            return (df['next_askPrice'] * -200) * (1 + self.tax) - self.hf
        
        # long to empty: previous bidPrice1 * money_per_point       1 to  0
        elif(df['previous_position'] == 1 and df['position'] == 0):
            return (df['next_bidPrice'] * 200) * (1 - self.tax) - self.hf
        
        # empty to short: previous bidPrice1 * money_per_point      0 to -1
        elif(df['previous_position'] == 0 and df['position'] == -1):
            return (df['next_bidPrice'] * 200) * (1 - self.tax) - self.hf

        # short to empty: previous askPrice1 * (-money_per_point)   -1 to  0
        elif(df['previous_position'] == -1 and df['position'] == 0):
            return (df['next_askPrice'] * -200) * (1 + self.tax) - self.hf
        
        # long to short: previous bidPrice1 * money_per_point * 2    1 to -1
        elif(df['previous_position'] == 1 and df['position'] == -1):
            return (df['next_bidPrice'] * 200 * 2) * (1 - self.tax) - self.hf * 2
        
        # short to long: previous askPrice1 * (-money_per_point) * 2 -1 to  1 
        elif(df['previous_position'] == -1 and df['position'] == 1):
            return (df['next_askPrice'] *-200 * 2) * (1 + self.tax) - self.hf * 2
        else:
            return 0
    #投資報酬率    
    def ROI(self, df):
        # empty to long: previous askPrice1 * (-money_per_point)    0 to  1
        if(df['previous_position'] == 0 and df['position'] == 1):
            self.cost = df['cashflow']
            return 0
        
        # long to empty: previous bidPrice1 * money_per_point       1 to  0
        elif(df['previous_position'] == 1 and df['position'] == 0):
            a = (df['cashflow'] + self.cost)*100/abs(self.cost)
            self.cost = 0
            return a
        
        # empty to short: previous bidPrice1 * money_per_point      0 to -1
        elif(df['previous_position'] == 0 and df['position'] == -1):
            self.cost = df['cashflow']
            return 0

        # short to empty: previous askPrice1 * (-money_per_point)   -1 to  0
        elif(df['previous_position'] == -1 and df['position'] == 0):
            a = (df['cashflow'] + self.cost)*100/abs(self.cost)
            self.cost = 0
            return a
        
        # long to short: previous bidPrice1 * money_per_point * 2    1 to -1
        elif(df['previous_position'] == 1 and df['position'] == -1):
            a = (df['cashflow']/2 + self.cost)*100/abs(self.cost)
            self.cost = df['cashflow']/2
            return a
        
        # short to long: previous askPrice1 * (-money_per_point) * 2 -1 to  1 
        elif(df['previous_position'] == -1 and df['position'] == 1):
            a = (df['cashflow']/2 + self.cost)*100/abs(self.cost)
            self.cost = df['cashflow']/2
            return a
        else:
            return 0
    #交易次數
    def NOT(self, df):
        if(df['previous_position'] !=  df['position']):
            self.number_of_transactions = self.number_of_transactions+1
            self.transactions_records.append({
                'date': str(df.name),
                'previous_position': df['previous_position'],
                'position': df['position'],
                'bidPrice': df['bidPrice'],
                'next_bidPrice': df['next_bidPrice'],
                'askPrice': df['askPrice'],
                'next_askPrice': df['next_askPrice'],
                'position_value': df['position_value'],
                'cashflow': df['cashflow'],
                'cash': df['cash'],
                'portfolio_value': df['portfolio_value'],
                'ROI': df['ROI']
            })     
            # })
        return self.number_of_transactions
        
    #手頭現金計算
    def cash(self, df):
        self.balance += df['cashflow']
        return self.balance
    #總財產(現金+部位) 
    def portfolio_value(self, df):
        return df['cash'] + df['position_value']
    
   #夏普值 
    def Sharpe_Ratio(self):
        df = self.df
        df_resample = df.resample('1D').agg({
            'portfolio_value':'first',
        })
        dl_resample = df.resample('1D').agg({
            'portfolio_value':'last',
        })
        d = (dl_resample - df_resample)/df_resample
        sharpe_ratio = d.mean() / np.std(d)
        self.sharpe_ratio = float(f'{sharpe_ratio.values[0]: .4f}')
       
        
    def draw(self):
        data = pd.DataFrame(self.df['portfolio_value']).resample('5T').last().dropna()['portfolio_value'].to_list()
        plt.figure(figsize=(16, 9))
        i = np.argmax(np.maximum.accumulate(data) - data) # end of the period
        j = np.argmax(data[:i]) # start of period
        plt.plot(data, color = 'red', fillstyle='none', label = 'Portfolio Value')
        plt.plot([i, j], [data[i], data[j]], 'o', color='blue', markersize=10, fillstyle='full', label = 'Max drawdown')
        plt.ticklabel_format(style='plain')
        plt.title('Portfolio Value')
        plt.xlabel('times')
        plt.ylabel('values')
        plt.legend(loc='upper left')
        plt.grid()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        plt.savefig('output/backtest.png', format = 'png')
        
        
    
    #動態圖表-總財產
    def draw_ani(self):
        st = time.time()
        data = pd.DataFrame(self.df['portfolio_value']).resample('5T').last().dropna()['portfolio_value']
        
        fig = plt.figure(figsize=(16, 9), tight_layout=True)
        ax = plt.axes(xlim=(0, len(data)), ylim=(data.min(), data.max( )))
        line, = ax.plot([], [], lw=2, color = 'red')
        line_2, = ax.plot([], [], lw=2, color='blue', markersize=10, fillstyle='full', label = 'Max drawdown')
        ax.axes.xaxis.set_visible(False)
        
        i = np.argmax(np.maximum.accumulate(data) - data) # end of the period
        j = np.argmax(data[:i]) # start of period
        #初始化 line
        def init():
            line.set_data([], [])
            line_2.set_data([], [])
            return line,
        #給予 line 內容
        # print(a)
        def animate(fps):
            x = np.linspace(0, 0+fps, fps+1)
            y = data[:fps+1].values
                        
            line.set_data(x, y)
            line_2.set_data([i, j], [data[i], data[j]])
            return line,
        ani = FuncAnimation(fig, animate, init_func=init, frames=len(data),  
                            interval=10, blit=True,repeat = None)
        ani.save('output/backtest.gif')
        print(time.time() - st)
        

   

input_df = pd.read_csv("output/predict_result.csv", index_col = 'Date-Time')
input_df.index = pd.to_datetime(input_df.index)

backtest = Backtest(input_df)
backtest.computing()
backtest.draw()
backtest.show_info()
