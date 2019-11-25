#coding=utf-8
from WindPy import *
import pandas as pd
import numpy as np
import random
import matplotlib as plt
from scipy.stats import kstest
import bkt
import statsmodels.api as sm

plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize'] = (16.0, 8.0)

class spec:
    def __init__(self):
        self.data = pd.read_csv('C:/Users/cheer/Desktop/huicedata.csv',parse_dates=True,index_col=0)
        self.initial_balance=1000000
        self.holding_value=0
        self.position_df=pd.DataFrame(columns=['持仓日期','持仓量','持仓总市值','股票价格'])
        self.capital_df=pd.DataFrame(columns=['总资产','持仓价值','可用资金'])
        self.output_df=pd.DataFrame()
        self.currentbar=1
        self.risk_free_rate=0.03
        self.order_df=pd.DataFrame()
        self.trade_info=pd.DataFrame(columns=['日期','交易方向','交易标的','交易量','成交价格','交易状态'])
        self.showplot=True


#output_data.plot()

def find_cointegrated_pairs(dataframe):
    # 得到DataFrame长度
    n = dataframe.shape[1]
    # 初始化p值矩阵
    pvalue_matrix = np.ones((n, n))
    # 抽取列的名称
    keys = dataframe.keys()
    # 初始化强协整组
    pairs = []
    # 对于每一个i
    for i in range(n):
        # 对于大于i的j
        for j in range(i+1, n):
            # 获取相应的两只股票的价格Series
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            # 分析它们的协整关系
            result = sm.tsa.stattools.coint(stock1[:120], stock2[:120])
            # 取出并记录p值
            pvalue = result[1] 
            corr=stock1[:120].corr(stock2[:120])
            pvalue_matrix[i, j] = pvalue
            # 如果p值小于0.05
            if pvalue < 0.05 and abs(corr)>0.8:
                # 记录股票对和相应的p值
                pairs.append((keys[i], keys[j], pvalue,corr))
    # 返回结果
    return pvalue_matrix, pairs

def backtest(initialize,context,strategy):
    initialize(context)
    context.current_capital_df=pd.DataFrame({'总资产': context.initial_balance,'持仓价值':0,'可用资金':context.initial_balance},index=[context.date[0]])
    for t in range(0,len(context.date)):
        context.currentbar=t
#        context.current_capital_df=pd.DataFrame(context.capital_df,columns=context.capital_df.columns,index=[context.capital_df.index[t-1]])
        strategy(context)

        
        for i in range(0,len(context.position_df.index)):
            code=context.position_df.index[i]
            context.position_df.loc[code,'持仓总市值']=context.position_df.loc[code]['持仓量']*context.data[code][context.date[t]]
        holding_value=context.position_df['持仓总市值'].sum()
        available_fund=context.current_capital_df.iloc[0][2]
        total_assets=available_fund+holding_value
        context.current_capital_df=pd.DataFrame({'总资产':total_assets,'持仓价值':holding_value,'可用资金':available_fund},index=[context.date[t]])
        context.capital_df=context.capital_df.append(context.current_capital_df)
        
        

    context.outputdata=pd.DataFrame(index=context.date,dtype='float64')
    context.outputdata['基准']=context.ben.div(context.ben.ix[context.date[0],0])
    context.outputdata['策略']=context.capital_df['总资产']/context.initial_balance
    context.outputdata['基准']=context.outputdata['基准'].astype('float64')
    context.outputdata['策略']=context.outputdata['策略'].astype('float64')
    

    return bkt.summary(context,context.outputdata)



        
def order(context,code,amt,trade_side):
    t=context.currentbar
    available_fund=context.current_capital_df.iloc[0][2]
    price=context.data.loc[context.date[t],code]
    if not(price<=1000000):
        return
    else:
        volume=int((amt/price//100)*100)
    
    if trade_side=='buy':
        if available_fund>=amt:
            if code in list(context.position_df.index):
                context.position_df.loc[code, '持仓日期']=context.date[t]
                context.position_df.loc[code, '持仓量']=volume+context.position_df['持仓量'][code]
                context.position_df.loc[code, '持仓总市值']=price*context.position_df['持仓量'][code]
                context.position_df.loc[code, '股票价格']=price
                success=1
                context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'成功'}),ignore_index=True)
            elif code not in list(context.position_df.index):
                context.position_df=context.position_df.append(pd.DataFrame({'持仓日期': context.date[t],'持仓量': volume,'持仓总市值': context.data[code][context.date[t]]*volume,'股票价格':price},index=[code]))
                success=1
                context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'成功'}),ignore_index=True)
        else:
            success=0
            context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'资金不足'}),ignore_index=True)

            
    elif trade_side=='sell':
        if code in list(context.position_df.index):
            if context.position_df.loc[code,'持仓量']>=volume:
                context.position_df.loc[code, '持仓日期']=context.date[t]
                context.position_df.loc[code, '持仓量']=context.position_df['持仓量'][code]-volume
                context.position_df.loc[code, '持仓总市值']=price*context.position_df.loc[code]['持仓量']
                context.position_df.loc[code, '股票价格']=price
                success=1
                context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'成功'}),ignore_index=True)
            else:
                success=1
                volume=context.position_df.loc[code,'持仓量']
                context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'持仓数量不足，已卖出 '+code+' '+str(volume)}),ignore_index=True)
                context.position_df.loc[code, '持仓量']=int(0)
        else:
            success=0
            context.trade_info=context.trade_info.append(pd.Series({'日期':context.date[t],'交易方向':trade_side,'交易标的':code,'交易量':volume,'成交价格':price,'交易状态':'持仓数量不足'}),ignore_index=True)

    order_df=pd.DataFrame({'code':code,'amt':amt,'tradeside':trade_side,'success':success,'volume':volume,'price':price},index=[0])
    
    if order_df['success'][0]==1:
        if order_df['tradeside'][0]=='buy':
            available_fund=context.current_capital_df.iloc[0][2]-price*volume
        elif order_df['tradeside'][0]=='sell':
            available_fund=context.current_capital_df.iloc[0][2]+price*volume

    for i in range(0,len(context.position_df.index)):
        code=context.position_df.index[i]
        context.position_df.loc[code,'持仓总市值']=context.position_df.loc[code,'持仓量']*context.data.loc[context.date[t],code]

    rows=[x for i,x in enumerate(context.position_df.index) if context.position_df.iat[i,1]==0]
    context.position_df=context.position_df.drop(rows)


    
    holding_value=context.position_df['持仓总市值'].sum()
    total_assets=available_fund+holding_value

    context.order_df.iloc[0:]=0
    context.current_capital_df=pd.DataFrame({'总资产':total_assets,'持仓价值':holding_value,'可用资金':available_fund},index=[context.date[t]])
    
    available_fund=context.current_capital_df.iloc[0][2]


#回测指标

#绝对收益率(Absolute Return) 输入总资产变化
def ABS_R(strat):
    Abs_return=strat[-1]/strat[0]-1
    return Abs_return

#累计收益率(Cumulative Return)
def CUM_R(strat):
    Cum_return=(strat.diff().div(strat.shift(periods=1))+1).dropna().prod()-1
    return Cum_return

#年化收益率
def ANN_R(strat):
    Ann_return=(ABS_R(strat)+1)**(252/len(strat))-1
    return Ann_return

#相对收益率
def REL_R(strat,target):
    Rel_return=ABS_R(strat)-ABS_R(target)
    return Rel_return

#Beta值
def BETA(strat,target):
    Rpt=(strat.diff()).dropna()
    Rmt=(target.diff()).dropna()
    Beta=Rpt.cov(Rmt)/np.var(target.diff())
    return Beta

#Alpha值
def ALPHA(strat,target,Rf=0.03):
    Alpha=ANN_R(strat)-(Rf+BETA(strat,target)*(ANN_R(target)-Rf))
    return Alpha

#夏普比率（Sharp ratio）
def SHARPr(strat,Rf=0.03):
    Sharp_r=(ABS_R(strat)-Rf)/(np.std(strat))
    return Sharp_r

#收益波动率（Volatility）
def VOL(strat):
    VOL=(((strat.diff().div(strat.shift(periods=1))).dropna()-(strat.diff().div(strat.shift(periods=1))).dropna().mean()).pow(2).sum()*252/(len(strat)-1))**(1/2)
    return VOL

#信息比率（IR）
def IR(strat,target):
    Rpt=(strat.div(strat.shift(periods=1))-1).dropna()
    Rmt=(target.div(target.shift(periods=1))-1).dropna()
    I_r=(ANN_R(strat)-ANN_R(target))/(np.std(Rpt-Rmt)*np.sqrt(252))
    return I_r


#最大回撤（Max Drawdown）
def Drawdown(strat):
    j=0
    Drawdown=[]
    for i in range(0,len(strat)):
        if strat[j]>strat[i]:
            Drawdown.append((strat[j]-strat[i])/strat[j])
        else:
            Drawdown.append(0)
            j=i
    Drawdown=pd.Series(Drawdown)
    return Drawdown

def MAXDD(strat):
    return max(Drawdown(strat))

def summary(context,output_df):
    summarydata=pd.DataFrame(index=['值'])
    strat=output_df['策略']
    ben=output_df['基准']
    exceedreturn=pd.Series(name='超额收益率')
    exceedreturn=strat-ben
    summarydata['累计收益率']=CUM_R(strat)
    summarydata['年化收益率']=ANN_R(strat)
    summarydata['相对收益率']=REL_R(strat,ben)
    summarydata['Alpha']=ALPHA(strat,ben)
    summarydata['Beta']=BETA(strat,ben)
    summarydata['夏普比率']=SHARPr(strat)
    summarydata['收益波动率']=VOL(strat)
    summarydata['信息比率']=IR(strat,ben)
    summarydata['最大回撤']=MAXDD(strat)
    if context.showplot:
        outputplot=output_df.plot()
        exceed=outputplot.twinx()
        exceed.set_ylabel('超额收益率',color='green')
        exceed.plot(exceedreturn,color='green')
        plt.pyplot.show()
    return summarydata