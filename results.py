# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from utility import *
import matplotlib.pyplot as plt
from WindPy import *
import datetime

w.start()
path = 'D:\\citics\\Barra\\data\\'

Exposure = pd.read_csv(path + 'FactorExposure.csv',encoding='utf-8_sig')
FactorReturn = pd.read_csv(path + 'FactorReturn.csv', index_col = 0)
Specific =pd.read_csv(path+'specific_risk.csv') 
raw = pd.read_csv(path + 'dataall.csv')
tvalue = pd.read_csv(path + 'tvalue.csv',index_col = 0)
VIF = pd.read_csv(path + 'VIF.csv',index_col = 0)

#######因子收益率计算
def factor_cumulate_plot(FactorReturn, startdate = '2015-07-01', enddate = '2018-07-01'):
    FactorReturn =FactorReturn[startdate:enddate]
    FactorReturn.rename(columns = {'RV':'ResVol', 'BTP':'BTOP','EY':'EarnYild','NLS':'SizeNL'}, inplace = True)
    cumreturn = pd.DataFrame(index = pd.to_datetime(FactorReturn.index))
    for each in FactorReturn.columns.tolist()[:10]:
        cumreturn[each] = (np.array(FactorReturn[each]+1).cumprod() -1) *100
    split = [0,3,6,8,10]
    for i in np.arange(len(split)-1):
        cumreturn.iloc[:,split[i]:split[(i+1)]].plot(title = 'Cumulative Return (Percent)', figsize = (12,6))
        
    return(cumreturn)

c = factor_cumulate_plot(FactorReturn)

####因子检验
def factor_property(t, vif, FactorReturn,factors_ind = np.arange(10), startdate = '2015-07-01', enddate = '2018-07-01'):
    vif = vif[startdate:enddate].iloc[:,factors_ind]
    t = t[startdate:enddate].iloc[:,factors_ind]
    FactorReturn = FactorReturn[startdate:enddate].iloc[:,factors_ind]
    delta = datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.datetime.strptime(startdate, '%Y-%m-%d')
    factor_list = FactorReturn.columns.values
    result = pd.DataFrame()
    for each in factor_list:
        avg_t = np.mean(abs(t[each]))
        percent = np.mean(abs(t[each]) >2) *100
        annual_return = (np.exp(sum(FactorReturn[each])/(delta.days/365)) -1) *100
        annual_vol = np.std(FactorReturn[each] *100) * np.sqrt(252)
        avgvif = np.mean(vif[each])
        summary = pd.DataFrame({'Average_Absolute_t':avg_t,'Percent_t':percent,'Annual_Factor_Return':annual_return,'Annual_Factor_Volatility':annual_vol,'VIF':avgvif},
                               index = [each], columns =['Average_Absolute_t','Percent_t','Annual_Factor_Return','Annual_Factor_Volatility','VIF'])
        result = result.append(summary)
    return result


FactorProperty = factor_property(tvalue, VIF, FactorReturn)

########################################
######因子收益分解
def got_index_weight(index_code,tradedate = '2018-07-01'):
    if type(tradedate) == str:
        daily = w.wset("indexconstituent","date="+tradedate +";windcode=" + index_code)
        dfs = pd.DataFrame({'code':daily.Data[1],'w':daily.Data[3]}, index = np.tile(tradedate,len(daily.Data[1])))
    else:
        dfs = pd.DataFrame()
        for each in tradedate:
            daily = w.wset("indexconstituent","date="+each +";windcode=" + index_code)
            df = pd.DataFrame({'code':daily.Data[1],'w':daily.Data[3]}, index = np.tile(each,len(daily.Data[1])))
            dfs = dfs.append(df)
        dfs.reset_index(inplace = True)
        dfs.rename(columns = {'index':'datetime','code':'code','w':'w'}, inplace = True)
    return dfs


def portfolio_return(pofo, factor_return, factor):
    portfolio = pd.merge(pofo, factor, on = 'code', how = 'left')
    portfolio.loc[portfolio.isnull().any(axis = 1),'w'] = 0
    portfolio.fillna(0, inplace = True)
    w = np.array(portfolio.w) / sum(portfolio.w)
    X = np.array(portfolio.iloc[:,3:-2])
    expo = np.dot(w, X)
    return_dcp = np.dot(w,X) *factor_return
    return_dcp = pd.DataFrame(dict(return_dcp),index = [factor_return.name])
    expo = pd.DataFrame(dict(zip(portfolio.iloc[:,3:-2].columns.values,expo)),index = [factor_return.name])
    return return_dcp, expo

#收益分解作图
def return_dcp_plot(FactorReturn,code, Factors, startdate = '2015-07-01', enddate = '2018-07-01'):
    FactorReturn =FactorReturn[startdate:enddate]
    FactorReturn.rename(columns = {'RV':'ResVol', 'BTP':'BTOP','EY':'EarnYild','NLS':'SizeNL'}, inplace = True)
    cumreturn = pd.DataFrame(index = pd.to_datetime(FactorReturn.index))
    for each in FactorReturn.columns.tolist()[:10]:
        cumreturn[each] = (np.array(FactorReturn[each]+1).cumprod() -1) *100
    daily = w.wsd(code,'pct_chg',startdate,enddate,"")
    cumreturn[code] = ((np.array(daily.Data[0])/100+1).cumprod() -1) *100
#    split = [0,3,6,8,10]
#    for i in np.arange(len(split)-1):
#        col_ind = np.arange(split[i],split[(i+1)])
#        col_ind = np.append(col_ind, -1)
#        cumreturn.iloc[:,col_ind].plot(title = 'Cumulative Return (Percent)', figsize = (12,6))
    F = Factors+[code]
    cumreturn[F].plot(title = 'Return Attribution (Percent)', figsize = (12,6))
    return(cumreturn)
    
def portfolio_dcp(index, Factors, Exposure, FactorReturn,startdate = '2015-07-01', enddate = '2018-07-01'):
    portfolio = got_index_weight(index)
    Exposure = Exposure.loc[(Exposure.datetime>=startdate) & (Exposure.datetime< enddate),:]
    FactorReturn = FactorReturn.loc[startdate:enddate,:]
    portfolio_exposure = pd.DataFrame()
    portfolio_return_dcp = pd.DataFrame()
    for each in FactorReturn.index.values:
        pofo = portfolio.copy()
        factor = Exposure[Exposure.datetime == each]
        factor_return = FactorReturn.loc[each,:]
        pofo_dcp,expo = portfolio_return(pofo, factor_return, factor)
        portfolio_return_dcp = portfolio_return_dcp.append(pofo_dcp)
        portfolio_exposure = portfolio_exposure.append(expo)
    return_dcp_plot(portfolio_return_dcp, index, Factors, startdate, enddate)
    delta = datetime.datetime.strptime(enddate, '%Y-%m-%d') - datetime.datetime.strptime(startdate, '%Y-%m-%d')
    result = pd.DataFrame()
    result['average_exposure'] = portfolio_exposure.mean(axis = 0)
    result['cumulate_return'] = (portfolio_return_dcp+1).prod(axis = 0) - 1
    result['annualized_return'] = result['cumulate_return']/(delta.days/365)
    return  result.loc[Factors,:]


a = portfolio_dcp('000300.SH',['Size'], Exposure, FactorReturn,startdate = '2015-07-01', enddate = '2018-07-01')


#######################################
######################预测 实现风险比较
def forecast_risk(df, Factor, T= 90):
    risk = pd.Series(index = df.index)
    for i in np.arange(T,len(df)):
           risk.iloc[i] = cov_pair(df[Factor].iloc[(i-T):i], df[Factor].iloc[(i-T):i])
    risk = np.sqrt(risk)
    return risk

def realized_risk(df, Factor, T = 126):
    risk = pd.Series(index = df.index)
    for i in np.arange(len(df)-T):
        risk.iloc[i] = np.var(df[Factor].iloc[i:i+T])
    risk = np.sqrt(risk)
    return risk
def risk_compare(FactorReturn, Factor,startdate, enddate, TF= 90, TR= 126):
    realized = realized_risk(FactorReturn, 'Size', TR)  
    forecast = forecast_risk(FactorReturn, 'Size', TF)
    realized = realized[startdate:enddate] * np.sqrt(252)
    forecast = forecast[startdate:enddate] * np.sqrt(252)
    riskplot = pd.DataFrame({'Realized':realized,'Forecast':forecast})
    riskplot.index= pd.to_datetime(riskplot.index)
    riskplot.plot(title = 'Forecast and realized risk in '+Factor+' factor.', figsize = (12,6))
    return riskplot

startdate = '2016-01-01'
enddate = '2017-12-31'
risk_compare(FactorReturn, 'Size', startdate, enddate)


##########################################
'''
多因子有效性检验
多因子的有效性检验一般基于两个方法：
1、相关性检验：计算同一时刻的个股的指标值和未来一段时间收益的相关性，再进行显著性检验，相关性越强，说明选股能力越强；
2、单调性检验：按照指标值大小对股票进行分组，从时间序列的角度观察各组的历史累计收益、信息比率、最大回撤以及胜率等。各组表现的优势组的胜率越高，单调性越强，说明指标的区分能力和选股能力越强。
'''


###3#########根据方法1 计算spearman相关系数

def corrvalidate(Exposure, data, Factors, time, period):
    exposure = Exposure.copy()
    tradedate = sorted(exposure.datetime.unique())
    ind = np.where(np.array(tradedate) == time)[0][0]
    start = tradedate[ind]
    end = tradedate[ind+period]
    close = data[['datetime','code','close']].copy()
    close['datetime'] =pd.to_datetime(close.datetime)
    close.set_index(['datetime'], inplace = True)
    exposure['datetime'] = pd.to_datetime(exposure['datetime'])
    exposure.set_index('datetime', inplace = True)
    exposurebefore = exposure.loc[time,['datetime', 'code']+ Factors]
    dfs = pd.merge(exposurebefore, close[start], on = ['code'], how = 'inner')
    dfs = pd.merge(dfs, close[end], on = ['code'], how = 'inner')
    dfs['return'] = np.log(dfs.iloc[:,-1]/ dfs.iloc[:,-2])
    dfs.drop(['close_x','close_y'], axis =1, inplace = True)
    corr, pvalue = stats.spearmanr(dfs.iloc[:,2:])
    result= pd.DataFrame({'corr':corr[-1,:-1], 'pvalue':pvalue[-1,:-1]}, index = Factors)
    return result

styles = FactorReturn.columns.tolist()[:10]

spmcorr = corrvalidate(Exposure, raw, styles, '2018-04-27', 21)

##############根据方法2 在每月的前一天，根据因子大小 做n个portfolio，在下一期对收益进行排序

def factorvalidate(exposure, close,F,startdate,enddate, n = 10):
    month = monthRange(startdate, enddate)
    returnbyexposure = pd.DataFrame()
    for each in month:
        date = exposure[each].index.unique()
        start = date[0].strftime('%Y-%m-%d')
        end = date[-1].strftime('%Y-%m-%d')
        exposurebefore = exposure.loc[start,['code',F]]
        interval = len(exposurebefore) / n
        exposurebefore['group'] = exposurebefore[F].rank()//interval
        factorselect = pd.merge(exposurebefore, close[start], on = 'code')
        factorselect.rename(columns = {'close':'begin'}, inplace = True)
        factorselect = pd.merge(factorselect, close[end],on = 'code')
        factorselect.rename(columns = {'close':'end'}, inplace = True)
        factorselect['monthreturn'] = np.log(factorselect.end/factorselect.begin)
        result = factorselect.groupby('group').mean()
        returnbyexposure[each] = result.monthreturn
    return returnbyexposure
 

def getmarket(code,startdate, enddate):
    daily = w.wsd(code,'pct_chg',startdate,enddate,"Period=M")
    month = monthRange(startdate, enddate)
    market = pd.Series(daily.Data[0], index = month)
    market /= 100
    market.name = code
    return market



def groupvalidate(exposure, data, F,startdate,enddate,rf, code, n = 10):
    close = data[['datetime','code','close']]
    close['datetime'] =pd.to_datetime(close.datetime)
    close.set_index(['datetime'], inplace = True)
    exposure['datetime'] = pd.to_datetime(exposure.datetime)
    exposure.set_index('datetime', inplace = True)
    
    validate = factorvalidate(exposure, close,F, startdate,enddate, n)
    
    validate = validate.stack().unstack(0)
    
    validate[code] = getmarket(code,startdate, enddate)
    result = pd.DataFrame()
    
    result['Cumulate_Return'] = (validate +1).prod(axis =0) - 1
    
    result['Volatility'] = validate.std(axis = 0) * np.sqrt(12)
    
    result['Sharp_Ratio'] = (result['Cumulate_Return'] - rf)/result['Volatility']
    
    return result
 
    
group = groupvalidate(Exposure, raw, F = 'Size', startdate = '2017-01-01',enddate = '2017-12-31', rf = 0.03, code =  '000016.SH', n = 10)



#######################预测投资组合风险

def predict_portfolio_risk(pofo, risk_matrix, factor, specific):
    w = np.mat(pofo).T
    F = np.mat(risk_matrix)

    X = np.mat(factor)
    dta = np.asmatrix(np.diag(specific**2))
    sig = np.asarray(w.T*X*F*X.T*w + w.T*dta*w)[0][0]
    return sig


pofo = got_index_weight('000300.SH')

def portfolio_forecast_perform(exposure, return_matrix,specific_risk, pofo, startdate,enddate):
    return_matrix = return_matrix[startdate:enddate]
    datetime =return_matrix.index.values
    portfolio_risk = pd.DataFrame(index = datetime)
    portfolio_risk['predict'] = np.nan
    portfolio_risk['r'] = np.nan
    for each in datetime:
        risk_matrix = cov_mat(return_matrix,each)
        factor = exposure[exposure.datetime == each]
        specific = specific_risk[specific_risk.datetime == each]
        dfs = pd.merge(pofo, factor, on = 'code', how = 'inner')
        dfs = pd.merge(dfs, specific, on = ['datetime','code'], how = 'inner')
        dfs['w'] = dfs['w']/ sum(dfs['w'])
        portfolio_risk.predict.loc[each] = predict_portfolio_risk(dfs['w'], risk_matrix, dfs[return_matrix.columns.tolist()], dfs['sigma'])
        portfolio_risk.r.loc[each] = sum(dfs['w']* dfs['return'])
    portfolio_risk['predict'] = np.sqrt(portfolio_risk['predict'])
    portfolio_risk['bias_stat'] = abs(portfolio_risk['r'])/portfolio_risk['predict']
    return portfolio_risk

startdate = '2017-01-01'
enddate = '2018-01-01'
forecast = portfolio_forecast_perform(Exposure, FactorReturn,Specific, pofo, startdate,enddate)

forecast.reset_index(inplace = True)
forecast['datetime'] = pd.to_datetime(forecast['index'])

plt.figure(figsize=(16,8))          #设置绘图对象的宽度和高度

plt.plot(forecast.datetime[100:],forecast.bias_stat[100:])
plt.axhline(1+np.sqrt(2), c= 'r')
plt.title('Bias Statistics of CSI 300',{'size' : 23} )
plt.show()









