# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:55:46 2019

@author: CS18083101
"""

import os.path
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
#from sklearn.model_selection import train_test_split
from scipy.stats import norm
import datetime
import time
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from matplotlib.backends.backend_pdf import PdfPages
import math

# Load production values (clients)
data = pd.read_csv('prodVals01.csv',
                   delimiter=';',
                   parse_dates = ['Datetime'], 
                   index_col='Datetime')

#data01 = pd.read_csv('prodVals01.csv',
#                   delimiter=';',
#                   parse_dates = ['Datetime'], 
#                   index_col='Datetime')
#
#data02 = pd.read_csv('prodVals02.csv',
#                   delimiter=';',
#                   parse_dates = ['Datetime'], 
#                   index_col='Datetime')
#
#data = pd.concat([data01,data02])

data = data.rename(columns={'ProdValue': 'Production'})

# Load optimum efficiency (model)
PVSystemModel_Lisbon_20160801 = pd.read_csv('PVSystemModel_20160801_deltaUTC_1.csv')

# Compute number of entries per client per day
perday = data.groupby(['ClientID', data.index.date]).count()
# Pick only days without missing values
desired_rows = perday[perday.Production==96]
# Pick only one week without missing values
desired_rows_weeks=pd.DataFrame(columns=['Production'])
#clientlist=[]
for i in range(0, len(desired_rows)-7):
    #if desired_rows.index.get_level_values(1)[i].weekday()==0:
    if desired_rows.index.get_level_values(1)[i]==datetime.date(2016, 8, 1):
        if desired_rows.index.get_level_values(1)[i+7]-desired_rows.index.get_level_values(1)[i]==datetime.timedelta(days=7):
            desired_rows_weeks=desired_rows_weeks.append(desired_rows[i:i+7])
            #pd.concat([desired_rows_weeks,desired_rows[i:i+7]])
            #clientlist.append(desired_rows.index.get_level_values(0)[i])
# Filter data according to desired rows (weeks)
data1=data
data1['ID_Date_tuple'] = list(zip(data1['ClientID'], data1.index.date))
data1 = data1[data1['ID_Date_tuple'].isin(desired_rows_weeks.index)]
data1['dateindex'] = data1.index
data1 = data1.sort_values(['ClientID', 'dateindex'], ascending=[True, True])
data1 = data1.drop(['dateindex', 'ID_Date_tuple'], axis=1)   

#round_per_client = pd.read_csv('power_hand_checked.csv')
#round_per_client = round_per_client.set_index('ClientID')

data1.set_index(['ClientID', data1.index])

# Reshape in order to have time in columns
data2 = data1.assign(ClientID=data1.ClientID, Date=data1.index.date, Time=data1.index.time) \
                       .pivot_table(index=['ClientID','Date'], columns=['Time'], values='Production', fill_value=0) \
                       .rename_axis(None, 1)
                                        
# Remove clients with com. problems
client_with_issues = pd.read_csv('PossibleComIssues.csv')
data2 = data2[~data2.index.get_level_values(0).isin(client_with_issues.ClientID)]

# Pick only clients with same location
#location=pd.read_csv('subsCity.csv',delimiter=';',encoding='latin-1')
#location=location.set_index('ClientID','City')
#clientlist=[]
#for i in range(0, len(data2)):
#    if ((location[location.index.get_level_values(0)==data2.index.get_level_values(0)[i]]=='Cascais') | \
#        (location[location.index.get_level_values(0)==data2.index.get_level_values(0)[i]]=='Sintra ') | \
#        (location[location.index.get_level_values(0)==data2.index.get_level_values(0)[i]]=='Lisboa')).bool()==False:
#        clientlist.append(data2.index.get_level_values(0)[i])
#        #data2=data2.drop(data2.index.get_level_values(0)[i])
#data2=data2.drop(clientlist)

# Remove clients with nighttime production
ClientsNightProd=[]
for i in range(0, len(data2), 7):
    if max(data2.values[i:i+7,0:4*4].flatten()) >= 1e-3:
#    or max(data2.values[i:i+7,22*4:23.75*4].flatten()) >= 1e-3:
        ClientsNightProd.append(data2.index.get_level_values(0)[i])
data2=data2.drop(ClientsNightProd)
   
#data2 = data2[data2.index.get_level_values(0).isin(round_per_client.index)]

#data3 = cp.copy(data2)
#data3['round'] = round_per_client.loc[data3.index.get_level_values(0)].values
#data3 = data3.div(data3['round'].values, axis=0)
#data3 = data3.drop('round', axis=1)

#plt.figure()
# Median filter
#data4 = data3.rolling(window=3, axis=1).median().fillna(0)
data4 = data2.rolling(window=3, axis=1).median().fillna(0)

# Mean filter
data5 = data4.rolling(window=3, axis=1).mean().fillna(0)

## Calculate reference values of maximum production
#MaxProdRef=pd.DataFrame(columns=['ClientID','MaxProdAbs'])
#for i in range(0, len(data['ClientID'].drop_duplicates())):
#    if np.isin(data['ClientID'].drop_duplicates()[i],data5.index.get_level_values(0)):
#        MaxProdRef.loc[i,'ClientID']=data['ClientID'].drop_duplicates()[i]
#        if data.loc[(data['ClientID']==data['ClientID'].drop_duplicates()[i]),'Production'] \
#        .nlargest(25).median()*4 <= 2.5: 
#            MaxProdRef.loc[i,'MaxProdAbs']=data.loc[(data['ClientID']==data['ClientID'] \
#                                           .drop_duplicates()[i]),'Production'].nlargest(25).median()
#            print(data['ClientID'].drop_duplicates()[i])
#        else:
#            MaxProdRef.loc[i,'MaxProdAbs']=2.5/4
#            print(data['ClientID'].drop_duplicates()[i])
#
#MaxProdRef=MaxProdRef.reset_index(drop=True)
#MaxProdRef['MaxProdRef']=''
#for i in range(0, len(MaxProdRef)):
#    if MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 0.25:
#        MaxProdRef.loc[i,'MaxProdRef']=0.25/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 0.25 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 0.5:
#        MaxProdRef.loc[i,'MaxProdRef']=0.5/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 0.5 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 0.75:
#        MaxProdRef.loc[i,'MaxProdRef']=0.75/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 0.75 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 1:
#        MaxProdRef.loc[i,'MaxProdRef']=1/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 1 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 1.25:
#        MaxProdRef.loc[i,'MaxProdRef']=1.25/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 1.25 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 1.5:
#        MaxProdRef.loc[i,'MaxProdRef']=1.5/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 1.5 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 1.75:
#        MaxProdRef.loc[i,'MaxProdRef']=1.75/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 1.75 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 2:
#        MaxProdRef.loc[i,'MaxProdRef']=2/4
#    elif MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 > 2 and \
#    MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdRef['ClientID'][i]),'MaxProdAbs'].values[0]*4 <= 2.25:
#        MaxProdRef.loc[i,'MaxProdRef']=2.25/4
#    else:
#        MaxProdRef.loc[i,'MaxProdRef']=2.5/4
#
## Save reference values of maximum production
#MaxProdRef.to_csv('MaxProdRef_20161118-20161124.txt', index=None, sep='\t')

# Load reference values of maximum production
MaxProdRef = pd.read_csv('input/20160801-20160807/MaxProdRef_20160801-20160807.txt', sep='\t')

# Group clients without production
ZeroProd=pd.DataFrame()
ClientsZeroProd=[]
ProdMin = 1e-3
for i in range(0, len(data5), 7):
    if all(data5.values[i:i+7].flatten() <= ProdMin):
        ZeroProd=ZeroProd.append(data5[i:i+7])
        ClientsZeroProd.append(data5.index.get_level_values(0)[i])
        
# Calculate sunrise/sunset time
## Sun declination angle (radians)
def delta(n):
    return 23.45*math.pi/180*math.sin(2*math.pi*(284+n)/365.25)

## Hour angle of sunrise/sunset (radians)
def omegaS(phi,n):
    if -math.tan(phi)*math.tan(delta(n)) >= 1:
        return 1e-6
    elif -math.tan(phi)*math.tan(delta(n)) <= -1:
        return np.pi
    else:
        return math.acos(-math.tan(phi)*math.tan(delta(n)))

## Local solar time of sunrise (hours)
def LSTsunrise(phi,n):
    return 12 - 180/math.pi*omegaS(phi,n)/15

## Local solar time of sunset (hours)
def LSTsunset(phi,n):
    return 12 + 180/math.pi*omegaS(phi,n)/15

## Local standard time meridian (radians)
def LSTM(delta_UTC):
    return 15*math.pi/180*delta_UTC

## ? (radians)
def alpha(n):
    return math.pi/180*360/365*n*(-81)

## Equation of time (minutes)
def EOT(n):
    return 9.87*math.sin(2*alpha(n)) - 7.53*math.cos(alpha(n)) - 1.5*math.sin(alpha(n))

## Time correction factor (minutes)
def TC(long,delta_UTC,n):
    return 4*180/math.pi*(long - LSTM(delta_UTC)) + EOT(n)

## Local sunrise time (hours)
def LTsunrise(phi,n,long,delta_UTC):
    return LSTsunrise(phi,n) - TC(long,delta_UTC,n)/60

## Local sunset time (hours)
def LTsunset(phi,n,long,delta_UTC):
    return LSTsunset(phi,n) - TC(long,delta_UTC,n)/60

# Calculate starttime/endtime (15-minute intervals)
phi = 40*math.pi/180
n = 213
long = -9*math.pi/180
delta_UTC = 1
ProdOffset = 2.5

SunriseTime = round(LTsunrise(phi,n,long,delta_UTC)*4)
SunsetTime = round(LTsunset(phi,n,long,delta_UTC)*4)
starttime = round((LTsunrise(phi,n,long,delta_UTC) + ProdOffset)*4)
endtime = round((LTsunset(phi,n,long,delta_UTC) - ProdOffset)*4)           

# Group clients with daytime zero-production
#starttime=10*4
#endtime=16*4
ZeroProdDay=pd.DataFrame()
ClientsZeroProdDay = []
for i in range(0, len(data5), 7):
    if any(data5.values[i:i+7,starttime:endtime].flatten() <= ProdMin):
        ZeroProdDay=ZeroProdDay.append(data5[i:i+7])
        ClientsZeroProdDay.append(data5.index.get_level_values(0)[i])        
                
# Group clients with sustained daytime zero-production
ZeroProdSust=pd.DataFrame()
ClientsZeroProdSust = []
for i in range(0, len(ZeroProdDay), 7):
    if max(ZeroProdDay.values[i,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+1,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+2,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+3,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+4,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+5,starttime:endtime].flatten()) <= ProdMin or \
       max(ZeroProdDay.values[i+6,starttime:endtime].flatten()) <= ProdMin: #or \
#      max(ZeroProd.values[i+7,9*4:17*4].flatten()) <= 1e-3:
            ZeroProdSust=ZeroProdSust.append(ZeroProdDay[i:i+7])
            ClientsZeroProdSust.append(ZeroProdDay.index.get_level_values(0)[i])
        
# Group clients with brief daytime zero-production        
ZeroProdBrief=pd.DataFrame()
ClientsZeroProdBrief = []
for i in range(0, len(ZeroProdDay), 7):
    if max(ZeroProdDay.values[i,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+1,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+2,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+3,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+4,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+5,starttime:endtime].flatten()) >= ProdMin and \
       max(ZeroProdDay.values[i+6,starttime:endtime].flatten()) >= ProdMin:
           ZeroProdBrief=ZeroProdBrief.append(ZeroProdDay[i:i+7])
           ClientsZeroProdBrief.append(ZeroProdDay.index.get_level_values(0)[i])

# Group clients without daytime zero-production
NoZeroProdDay=pd.DataFrame()
for i in range(0, len(data5), 7):
    if all(data5.values[i:i+7,starttime:endtime].flatten() > ProdMin):
        NoZeroProdDay=NoZeroProdDay.append(data5[i:i+7])

# Group clients with local minima
LocalMin=pd.DataFrame()
ClientsLocalMin=[]
LocalMinThreshold=1.01
for i in range(0, len(data5), 7):
    for j in range(starttime, endtime):
        if (data5.values[i,j].flatten() >= ProdMin and \
            ((data5.values[i,j-1].flatten() > LocalMinThreshold*data5.values[i,j].flatten() and \
            data5.values[i,j+1].flatten() > LocalMinThreshold*data5.values[i,j].flatten()) or \
            (data5.values[i,j-2].flatten() > LocalMinThreshold*data5.values[i,j].flatten() and \
            data5.values[i,j+2].flatten() > LocalMinThreshold*data5.values[i,j].flatten()))) or \
            (data5.values[i+1,j].flatten() >= ProdMin and \
            ((data5.values[i+1,j-1].flatten() > LocalMinThreshold*data5.values[i+1,j].flatten() and \
            data5.values[i+1,j+1].flatten() > LocalMinThreshold*data5.values[i+1,j].flatten()) or \
            (data5.values[i+1,j-2].flatten() > LocalMinThreshold*data5.values[i+1,j].flatten() and \
            data5.values[i+1,j+2].flatten() > LocalMinThreshold*data5.values[i+1,j].flatten()))) or \
            (data5.values[i+2,j].flatten() >= ProdMin and \
            ((data5.values[i+2,j-1].flatten() > LocalMinThreshold*data5.values[i+2,j].flatten() and \
            data5.values[i+2,j+1].flatten() > LocalMinThreshold*data5.values[i+2,j].flatten()) or \
            (data5.values[i+2,j-2].flatten() > LocalMinThreshold*data5.values[i+2,j].flatten() and \
            data5.values[i+2,j+2].flatten() > LocalMinThreshold*data5.values[i+2,j].flatten()))) or \
            (data5.values[i+3,j].flatten() >= ProdMin and \
            ((data5.values[i+3,j-1].flatten() > LocalMinThreshold*data5.values[i+3,j].flatten() and \
            data5.values[i+3,j+1].flatten() > LocalMinThreshold*data5.values[i+3,j].flatten()) or \
            (data5.values[i+3,j-2].flatten() > LocalMinThreshold*data5.values[i+3,j].flatten() and \
            data5.values[i+3,j+2].flatten() > LocalMinThreshold*data5.values[i+3,j].flatten()))) or \
            (data5.values[i+4,j].flatten() >= ProdMin and \
            ((data5.values[i+4,j-1].flatten() > LocalMinThreshold*data5.values[i+4,j].flatten() and \
            data5.values[i+4,j+1].flatten() > LocalMinThreshold*data5.values[i+4,j].flatten()) or \
            (data5.values[i+4,j-2].flatten() > LocalMinThreshold*data5.values[i+4,j].flatten() and \
            data5.values[i+4,j+2].flatten() > LocalMinThreshold*data5.values[i+4,j].flatten()))) or \
            (data5.values[i+5,j].flatten() >= ProdMin and \
            ((data5.values[i+5,j-1].flatten() > LocalMinThreshold*data5.values[i+5,j].flatten() and \
            data5.values[i+5,j+1].flatten() > LocalMinThreshold*data5.values[i+5,j].flatten()) or \
            (data5.values[i+5,j-2].flatten() > LocalMinThreshold*data5.values[i+5,j].flatten() and \
            data5.values[i+5,j+2].flatten() > LocalMinThreshold*data5.values[i+5,j].flatten()))) or \
            (data5.values[i+6,j].flatten() >= ProdMin and \
            ((data5.values[i+6,j-1].flatten() > LocalMinThreshold*data5.values[i+6,j].flatten() and \
            data5.values[i+6,j+1].flatten() > LocalMinThreshold*data5.values[i+6,j].flatten()) or \
            (data5.values[i+6,j-2].flatten() > LocalMinThreshold*data5.values[i+6,j].flatten() and \
            data5.values[i+6,j+2].flatten() > LocalMinThreshold*data5.values[i+6,j].flatten()))):
                LocalMin=LocalMin.append(data5[i:i+7])
                ClientsLocalMin.append(data5.index.get_level_values(0)[i])
                break

# Group clients with regular local minima
LocalMinReg=pd.DataFrame()
ClientsLocalMinReg=[]
for i in range(0, len(LocalMin), 7):
    for j in range(starttime, endtime):
        if ((LocalMin.values[i,j].flatten() >= ProdMin and \
           ((LocalMin.values[i,j-1].flatten() > LocalMinThreshold*LocalMin.values[i,j].flatten() and \
            LocalMin.values[i,j+1].flatten() > LocalMinThreshold*LocalMin.values[i,j].flatten()) or \
            (LocalMin.values[i,j-2].flatten() > LocalMinThreshold*LocalMin.values[i,j].flatten() and \
            LocalMin.values[i,j+2].flatten() > LocalMinThreshold*LocalMin.values[i,j].flatten()))),
            (LocalMin.values[i+1,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+1,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+1,j].flatten() and \
            LocalMin.values[i+1,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+1,j].flatten()) or \
            (LocalMin.values[i+1,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+1,j].flatten() and \
            LocalMin.values[i+1,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+1,j].flatten()))),
            (LocalMin.values[i+2,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+2,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+2,j].flatten() and \
            LocalMin.values[i+2,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+2,j].flatten()) or \
            (LocalMin.values[i+2,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+2,j].flatten() and \
            LocalMin.values[i+2,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+2,j].flatten()))),
            (LocalMin.values[i+3,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+3,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+3,j].flatten() and \
            LocalMin.values[i+3,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+3,j].flatten()) or \
            (LocalMin.values[i+3,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+3,j].flatten() and \
            LocalMin.values[i+3,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+3,j].flatten()))),
            (LocalMin.values[i+4,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+4,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+4,j].flatten() and \
            LocalMin.values[i+4,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+4,j].flatten()) or \
            (LocalMin.values[i+4,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+4,j].flatten() and \
            LocalMin.values[i+4,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+4,j].flatten()))),
            (LocalMin.values[i+5,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+5,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+5,j].flatten() and \
            LocalMin.values[i+5,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+5,j].flatten()) or \
            (LocalMin.values[i+5,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+5,j].flatten() and \
            LocalMin.values[i+5,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+5,j].flatten()))),
            (LocalMin.values[i+6,j].flatten() >= ProdMin and \
            ((LocalMin.values[i+6,j-1].flatten() > LocalMinThreshold*LocalMin.values[i+6,j].flatten() and \
            LocalMin.values[i+6,j+1].flatten() > LocalMinThreshold*LocalMin.values[i+6,j].flatten()) or \
            (LocalMin.values[i+6,j-2].flatten() > LocalMinThreshold*LocalMin.values[i+6,j].flatten() and \
            LocalMin.values[i+6,j+2].flatten() > LocalMinThreshold*LocalMin.values[i+6,j].flatten())))) \
            .count(True)>=4: 
                LocalMinReg=LocalMinReg.append(LocalMin[i:i+7])
                ClientsLocalMinReg.append(LocalMin.index.get_level_values(0)[i])
                break
                
# Group clients with irregular local minima
LocalMinIrreg=LocalMin.drop(ClientsLocalMinReg)

# Group clients without local minima
NoLocalMin=data5.drop(ClientsLocalMin)
NoLocalMin=NoLocalMin.drop(ClientsZeroProd)

# Calculate maximum production
MaxProdWeek=[]
for i in range(0, len(data5), 7):
    MaxProdWeek.append([data5.index.get_level_values(0)[i],data5.values[i:i+7].flatten().max()])
MaxProdWeek=pd.DataFrame(MaxProdWeek,columns=['ClientID','MaxProdWeek'])

# Group clients with low maximum production
ClientsLowMaxProd=[]
ProdThreshold=0.85
for i in range(0, len(MaxProdWeek)):
    if MaxProdWeek['MaxProdWeek'][i] >= ProdMin and \
    MaxProdWeek['MaxProdWeek'][i] <= ProdThreshold*MaxProdRef.loc[(MaxProdRef['ClientID']==MaxProdWeek['ClientID'][i]),'MaxProdRef'].values[0]: 
        ClientsLowMaxProd.append(MaxProdWeek['ClientID'][i])
LowMaxProd=data5[data5.index.get_level_values(0).isin(ClientsLowMaxProd)]

# Group clients with high maximum production
HighMaxProd = data5.drop(ClientsLowMaxProd)
HighMaxProd = HighMaxProd.drop(ClientsZeroProd)

## Calculate % of correctly identified clients with brief daytime zero-production
ZeroProdBriefAnnotation=pd.read_csv('input/20160801-20160807/ZeroProdBrief01_20160801-20160807.csv')
ZeroProdBriefCorrect=np.sum(ZeroProdBriefAnnotation.isin(ZeroProdBrief.index.get_level_values(0)).values.flatten())/ \
    len(ZeroProdBriefAnnotation)*100
ZeroProdBriefFalsePositives=(len(ZeroProdBrief.index.get_level_values(0).drop_duplicates())- \
    np.sum(ZeroProdBriefAnnotation.isin(ZeroProdBrief.index.get_level_values(0)) \
    .values.flatten()))/len(ZeroProdBrief.index.get_level_values(0).drop_duplicates())*100

## Calculate % of correctly identified clients with sustained daytime zero-production
ZeroProdSustAnnotation=pd.read_csv('input/20160801-20160807/ZeroProdSust01_20160801-20160807.csv')
ZeroProdSustCorrect=np.sum(ZeroProdSustAnnotation.isin(ZeroProdSust.index.get_level_values(0)).values.flatten())/ \
    len(ZeroProdSustAnnotation)*100
ZeroProdSustFalsePositives=(len(ZeroProdSust.index.get_level_values(0).drop_duplicates())- \
    np.sum(ZeroProdSustAnnotation.isin(ZeroProdSust.index.get_level_values(0)) \
    .values.flatten()))/len(ZeroProdSust.index.get_level_values(0).drop_duplicates())*100

## Calculate % of correctly identified clients with regular local minima
LocalMinRegAnnotation=pd.read_csv('input/20160801-20160807/LocalMinReg01_20160801-20160807.csv')
LocalMinRegCorrect=np.sum(LocalMinRegAnnotation.isin(LocalMinReg.index.get_level_values(0)).values.flatten())/ \
    len(LocalMinRegAnnotation)*100
LocalMinRegFalsePositives=(len(LocalMinReg.index.get_level_values(0).drop_duplicates())- \
    np.sum(LocalMinRegAnnotation.isin(LocalMinReg.index.get_level_values(0)) \
    .values.flatten()))/len(LocalMinReg.index.get_level_values(0).drop_duplicates())*100

# Anomaly summary per day
SummaryDay=pd.DataFrame(columns=['ClientID','Date','ZeroProdDay'])
for i in range(0, len(data5), 7):
    for j in range(0, 7):
        if any(data5.values[i+j,starttime:endtime] <= ProdMin):
            SummaryDay.loc[i+j]=[data5.index.get_level_values(0)[i+j], \
            data5.index.get_level_values(1)[i+j],'Yes']
        else:
            SummaryDay.loc[i+j]=[data5.index.get_level_values(0)[i+j], \
            data5.index.get_level_values(1)[i+j],'No']

SummaryDay['ZeroProdSust']=''
for i in range(0, len(data5), 7):
    for j in range(0, 7):
        if SummaryDay['ZeroProdDay'][i+j]=='No':
            SummaryDay.loc[i+j,'ZeroProdSust']='N/A'
        elif SummaryDay['ZeroProdDay'][i+j]=='Yes' and \
        max(data5.values[i+j,starttime:endtime]) <= ProdMin:
            SummaryDay.loc[i+j,'ZeroProdSust']='Yes'
        else:
            SummaryDay.loc[i+j,'ZeroProdSust']='No'

SummaryDay['ZeroProdBrief']=''
for i in range(0, len(data5), 7):
    for j in range(0, 7):
        if SummaryDay['ZeroProdDay'][i+j]=='No':
            SummaryDay.loc[i+j,'ZeroProdBrief']='N/A'
        elif SummaryDay['ZeroProdDay'][i+j]=='Yes' and \
        max(data5.values[i+j,starttime:endtime]) >= ProdMin:
            SummaryDay.loc[i+j,'ZeroProdBrief']='Yes'
        else:
            SummaryDay.loc[i+j,'ZeroProdBrief']='No'
            
SummaryDay['LocalMin']=''
for i in range(0, len(data5), 7):
    for j in range(0, 7):
        for k in range(starttime, endtime):
            if data5.values[i+j,k] >= ProdMin and \
            ((data5.values[i+j,k-1] > LocalMinThreshold*data5.values[i+j,k] and \
            data5.values[i+j,k+1] > LocalMinThreshold*data5.values[i+j,k]) or \
            (data5.values[i+j,k-2] > LocalMinThreshold*data5.values[i+j,k] and \
            data5.values[i+j,k+2] > LocalMinThreshold*data5.values[i+j,k])):
                SummaryDay.loc[i+j,'LocalMin']='Yes'
                break
            else:
                SummaryDay.loc[i+j,'LocalMin']='No'

SummaryDay['LowMaxProd']=''
for i in range(0, len(data5), 7):
    for j in range(0, 7):
        if SummaryDay['ZeroProdSust'][i+j]=='Yes':
            SummaryDay.loc[i+j,'LowMaxProd']='N/A'
        elif max(data5.values[i+j,starttime:endtime]) >= ProdMin and \
        max(data5.values[i+j,starttime:endtime]) <= ProdThreshold*MaxProdRef \
        .loc[(MaxProdRef['ClientID']==SummaryDay['ClientID'][i]),'MaxProdRef'].values[0]:
            SummaryDay.loc[i+j,'LowMaxProd']='Yes'
        else:
            SummaryDay.loc[i+j,'LowMaxProd']='No'

# Anomaly summary per week
SummaryWeek=pd.DataFrame(columns=['ClientID','ZeroProdSustTotal'])
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i]=[data5.index.get_level_values(0)[i], \
    SummaryDay['ZeroProdSust'][i:i+7].eq('Yes').sum()]

SummaryWeek['ZeroProdSustLast']=''
for i in range(0, len(data5), 7):
    if SummaryWeek['ZeroProdSustTotal'][i]==0:
        SummaryWeek.loc[i,'ZeroProdSustLast']='N/A'
    elif SummaryDay['ZeroProdSust'][i+6]=='Yes':
        SummaryWeek.loc[i,'ZeroProdSustLast']='Yes'
    else:
        SummaryWeek.loc[i,'ZeroProdSustLast']='No'
        
SummaryWeek['ZeroProdBriefTotal']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'ZeroProdBriefTotal']=SummaryDay['ZeroProdBrief'][i:i+7].eq('Yes').sum()
    
SummaryWeek['LocalMinReg']=''
for i in range(0, len(data5), 7):
    for j in range(starttime, endtime):
        if ((data5.values[i,j] >= ProdMin and \
           ((data5.values[i,j-1] > LocalMinThreshold*data5.values[i,j] and \
            data5.values[i,j+1] > LocalMinThreshold*data5.values[i,j]) or \
            (data5.values[i,j-2] > LocalMinThreshold*data5.values[i,j] and \
            data5.values[i,j+2] > LocalMinThreshold*data5.values[i,j]))),
            (data5.values[i+1,j] >= ProdMin and \
            ((data5.values[i+1,j-1] > LocalMinThreshold*data5.values[i+1,j] and \
            data5.values[i+1,j+1] > LocalMinThreshold*data5.values[i+1,j]) or \
            (data5.values[i+1,j-2] > LocalMinThreshold*data5.values[i+1,j] and \
            data5.values[i+1,j+2] > LocalMinThreshold*data5.values[i+1,j]))),
            (data5.values[i+2,j] >= ProdMin and \
            ((data5.values[i+2,j-1] > LocalMinThreshold*data5.values[i+2,j] and \
            data5.values[i+2,j+1] > LocalMinThreshold*data5.values[i+2,j]) or \
            (data5.values[i+2,j-2] > LocalMinThreshold*data5.values[i+2,j] and \
            data5.values[i+2,j+2] > LocalMinThreshold*data5.values[i+2,j]))),
            (data5.values[i+3,j] >= ProdMin and \
            ((data5.values[i+3,j-1] > LocalMinThreshold*data5.values[i+3,j] and \
            data5.values[i+3,j+1] > LocalMinThreshold*data5.values[i+3,j]) or \
            (data5.values[i+3,j-2] > LocalMinThreshold*data5.values[i+3,j] and \
            data5.values[i+3,j+2] > LocalMinThreshold*data5.values[i+3,j]))),
            (data5.values[i+4,j] >= ProdMin and \
            ((data5.values[i+4,j-1] > LocalMinThreshold*data5.values[i+4,j] and \
            data5.values[i+4,j+1] > LocalMinThreshold*data5.values[i+4,j]) or \
            (data5.values[i+4,j-2] > LocalMinThreshold*data5.values[i+4,j] and \
            data5.values[i+4,j+2] > LocalMinThreshold*data5.values[i+4,j]))),
            (data5.values[i+5,j] >= ProdMin and \
            ((data5.values[i+5,j-1] > LocalMinThreshold*data5.values[i+5,j] and \
            data5.values[i+5,j+1] > LocalMinThreshold*data5.values[i+5,j]) or \
            (data5.values[i+5,j-2] > LocalMinThreshold*data5.values[i+5,j] and \
            data5.values[i+5,j+2] > LocalMinThreshold*data5.values[i+5,j]))),
            (data5.values[i+6,j] >= ProdMin and \
            ((data5.values[i+6,j-1] > LocalMinThreshold*data5.values[i+6,j] and \
            data5.values[i+6,j+1] > LocalMinThreshold*data5.values[i+6,j]) or \
            (data5.values[i+6,j-2] > LocalMinThreshold*data5.values[i+6,j] and \
            data5.values[i+6,j+2] > LocalMinThreshold*data5.values[i+6,j])))).count(True)>=4:
               SummaryWeek.loc[i,'LocalMinReg']='Yes'
               break
        else:
            SummaryWeek.loc[i,'LocalMinReg']='No'
        
SummaryWeek['MeanProd']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MeanProd']=np.mean(data5.values[i:i+7,starttime:endtime].flatten())
    
SummaryWeek['MaxProd']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MaxProd']=max(data5.values[i:i+7,starttime:endtime].flatten())

SummaryWeek['MaxProdDate']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MaxProdDate']=data5.iloc[i:i+7,starttime:endtime].max(axis=1).idxmax()[1]
    
SummaryWeek['MaxProdTime']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MaxProdTime']=data5.iloc[i:i+7,starttime:endtime].max(axis=0).idxmax()

SummaryWeek['HighMaxProdTotal']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'HighMaxProdTotal']=SummaryDay['LowMaxProd'][i:i+7].eq('No').sum()
    
SummaryWeek['LocalMinRegTime']=''
for i in range(0, len(data5), 7):
    for j in range(starttime, endtime):
        if ((data5.values[i,j] >= ProdMin and \
           ((data5.values[i,j-1] > LocalMinThreshold*data5.values[i,j] and \
            data5.values[i,j+1] > LocalMinThreshold*data5.values[i,j]) or \
            (data5.values[i,j-2] > LocalMinThreshold*data5.values[i,j] and \
            data5.values[i,j+2] > LocalMinThreshold*data5.values[i,j]))),
            (data5.values[i+1,j] >= ProdMin and \
            ((data5.values[i+1,j-1] > LocalMinThreshold*data5.values[i+1,j] and \
            data5.values[i+1,j+1] > LocalMinThreshold*data5.values[i+1,j]) or \
            (data5.values[i+1,j-2] > LocalMinThreshold*data5.values[i+1,j] and \
            data5.values[i+1,j+2] > LocalMinThreshold*data5.values[i+1,j]))),
            (data5.values[i+2,j] >= ProdMin and \
            ((data5.values[i+2,j-1] > LocalMinThreshold*data5.values[i+2,j] and \
            data5.values[i+2,j+1] > LocalMinThreshold*data5.values[i+2,j]) or \
            (data5.values[i+2,j-2] > LocalMinThreshold*data5.values[i+2,j] and \
            data5.values[i+2,j+2] > LocalMinThreshold*data5.values[i+2,j]))),
            (data5.values[i+3,j] >= ProdMin and \
            ((data5.values[i+3,j-1] > LocalMinThreshold*data5.values[i+3,j] and \
            data5.values[i+3,j+1] > LocalMinThreshold*data5.values[i+3,j]) or \
            (data5.values[i+3,j-2] > LocalMinThreshold*data5.values[i+3,j] and \
            data5.values[i+3,j+2] > LocalMinThreshold*data5.values[i+3,j]))),
            (data5.values[i+4,j] >= ProdMin and \
            ((data5.values[i+4,j-1] > LocalMinThreshold*data5.values[i+4,j] and \
            data5.values[i+4,j+1] > LocalMinThreshold*data5.values[i+4,j]) or \
            (data5.values[i+4,j-2] > LocalMinThreshold*data5.values[i+4,j] and \
            data5.values[i+4,j+2] > LocalMinThreshold*data5.values[i+4,j]))),
            (data5.values[i+5,j] >= ProdMin and \
            ((data5.values[i+5,j-1] > LocalMinThreshold*data5.values[i+5,j] and \
            data5.values[i+5,j+1] > LocalMinThreshold*data5.values[i+5,j]) or \
            (data5.values[i+5,j-2] > LocalMinThreshold*data5.values[i+5,j] and \
            data5.values[i+5,j+2] > LocalMinThreshold*data5.values[i+5,j]))),
            (data5.values[i+6,j] >= ProdMin and \
            ((data5.values[i+6,j-1] > LocalMinThreshold*data5.values[i+6,j] and \
            data5.values[i+6,j+1] > LocalMinThreshold*data5.values[i+6,j]) or \
            (data5.values[i+6,j-2] > LocalMinThreshold*data5.values[i+6,j] and \
            data5.values[i+6,j+2] > LocalMinThreshold*data5.values[i+6,j])))).count(True)>=4:
               SummaryWeek.loc[i,'LocalMinRegTime']=data5.columns[j]
               break
        else:
            SummaryWeek.loc[i,'LocalMinRegTime']='N/A'           

SummaryWeek['MaxProdRef']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MaxProdRef']=MaxProdRef \
    .loc[(MaxProdRef['ClientID']==data5.index.get_level_values(0)[i]),'MaxProdRef'].values[0]
    
SummaryWeek['LowMaxProdTotal']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'LowMaxProdTotal']=SummaryDay['LowMaxProd'][i:i+7].eq('Yes').sum()

SummaryWeek['MaxEfficiency']=''
for i in range(0, len(data5), 7):
    SummaryWeek.loc[i,'MaxEfficiency']=SummaryWeek['MaxProd'][i]/SummaryWeek['MaxProdRef'][i]*100
    

clients = data5.index.get_level_values(0)
data_labels=data5.values

# Save production of all clients
ProdData=data5.stack().rename_axis(['ClientID','Date','Time']).reset_index(name='Production')

#ProdData['Efficiency']=''
#for i in range(0, len(ProdData)):
#    ProdData.loc[i, 'Efficiency']=ProdData['Production'][i]/ \
#        (MaxProdRef[MaxProdRef['ClientID']==ProdData['ClientID'][i]]['MaxProdRef']).values*100
#    print(ProdData['ClientID'][i])

ProdData.to_csv('ProdData.txt', sep='\t', index=False)

## Calculate one-week mean curves   
#MeanCurves=pd.DataFrame()
#MeanCurves['ClientID']=''
#for i in range(0, len(ProdData), 96*7):
#    for j in range(0, 96):
#        MeanCurves.loc[i+j,'ClientID']=ProdData['ClientID'][i]
#        
#MeanCurves['Time']=''
#for i in range(0, len(ProdData), 96*7):
#    for j in range(0, 96):
#        MeanCurves.loc[i+j,'Time']=(datetime.datetime(100,1,1,0,0,0)+datetime.timedelta(minutes=15*j)) \
#        .strftime('%H:%M:%S')
#
#MeanCurves['MeanProd']=''
#for i in range(0, len(ProdData), 96*7):
#    for j in range(0, 96):
#        MeanCurves.loc[i+j,'MeanProd']=np.mean([ProdData['Production'][i+j+96*0], \
#        ProdData['Production'][i+j+96*1],
#        ProdData['Production'][i+j+96*2],        
#        ProdData['Production'][i+j+96*3],
#        ProdData['Production'][i+j+96*4],        
#        ProdData['Production'][i+j+96*5],
#        ProdData['Production'][i+j+96*6]])
#
#MeanCurves['MeanEfficiency']=''
#for i in range(0, len(ProdData), 96*7):
#    for j in range(0, 96):
#        MeanCurves.loc[i+j,'MeanEfficiency']=MeanCurves['MeanProd'][i+j]/ \
#        MaxProdRef['MaxProdRef'][i/96/7]*100
#
#MeanCurves['MeanSlope']=''
#for i in range(0, len(ProdData), 96*7):
#    for j in range(1, 95):
#        MeanCurves.loc[i+j,'MeanSlope']=(MeanCurves['MeanEfficiency'][i+j+1]-MeanCurves['MeanEfficiency'][i+j-1])/ \
#        (2*15)
#MeanCurves['MeanSlope'].replace(r'^\s*$', np.nan, regex=True, inplace = True)
#        
## Save one-week mean curves   
#MeanCurves.to_csv('MeanCurves_20161118-20161124.txt', index=None, sep='\t')

# Load one-week mean curves
MeanCurves = pd.read_csv('input/20160801-20160807/MeanCurves_20160801-20160807.txt', sep='\t')

# Calculate shading amplitude and length   
SummaryWeek['PreLocalMax']=''
SummaryWeek['PostLocalMax']=''
SummaryWeek['ExpectedEfficiency']=''
SummaryWeek['ShadingAmplitude']=''
for i in range(0, int(len(MeanCurves)/96)):
    if SummaryWeek.loc[i*7,'LocalMinRegTime'] == 'N/A':        
        SummaryWeek.loc[i*7,'PreLocalMax'] = 'N/A'
        SummaryWeek.loc[i*7,'PostLocalMax'] = 'N/A'
        SummaryWeek.loc[i*7,'ExpectedEfficiency'] = 'N/A'
        SummaryWeek.loc[i*7,'ShadingAmplitude'] = 'N/A'      
    else:
        for j in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
        (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'].strftime("%H:%M:%S"))].index.values.item(), \
        MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']].iloc[[-1]].index.values.item()): 
            if MeanCurves['MeanEfficiency'][j] > MeanCurves['MeanEfficiency'][j-1] and \
            MeanCurves['MeanEfficiency'][j] > MeanCurves['MeanEfficiency'][j+1]:
                SummaryWeek.loc[i*7,'PostLocalMax']=MeanCurves['MeanEfficiency'][j]
                x1 = j
                y1 = MeanCurves['MeanEfficiency'][j]

        for k in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
        (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'].strftime("%H:%M:%S"))].index.values.item(), \
        MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']].iloc[[0]].index.values.item(),-1):
            if MeanCurves['MeanEfficiency'][k] > MeanCurves['MeanEfficiency'][k-1] and \
            MeanCurves['MeanEfficiency'][k] > MeanCurves['MeanEfficiency'][k+1]:
                SummaryWeek.loc[i*7,'PreLocalMax']=MeanCurves['MeanEfficiency'][k]
                x2 = k
                y2 = MeanCurves['MeanEfficiency'][k]
                Coefficients = np.polyfit([x1,x2], [y1,y2], 1)
                SummaryWeek.loc[i*7,'ExpectedEfficiency'] = \
                    Coefficients[0]*MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) \
                    & (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'] \
                    .strftime("%H:%M:%S"))].index.values.item() + Coefficients[1]
                SummaryWeek.loc[i*7,'ShadingAmplitude'] = (SummaryWeek['ExpectedEfficiency'][i*7] - \
                    MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) \
                    & (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'] \
                    .strftime("%H:%M:%S"))]['MeanEfficiency'].values)/ \
                    SummaryWeek['ExpectedEfficiency'][i*7]

## Fill in zero-valued shading amplitudes                      
SummaryWeek['ShadingAmplitude']=SummaryWeek['ShadingAmplitude'].str.get(0)
SummaryWeek['ShadingAmplitude']=SummaryWeek['ShadingAmplitude'].replace('N',0)

## Replace negative shading amplitudes by zero
SummaryWeek['ShadingAmplitude'][SummaryWeek['ShadingAmplitude'] < 0] = 0 

SummaryWeek['ShadingLength']=''
for i in range(0, int(len(MeanCurves)/96)):
    if SummaryWeek.loc[i*7, 'LocalMinRegTime'] == 'N/A':        
        SummaryWeek.loc[i*7,'ShadingLength'] = 'N/A'
    else:
        for j in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
        (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'].strftime("%H:%M:%S"))].index.values.item(), \
        MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']].iloc[[-1]].index.values.item()): 
            if MeanCurves['MeanEfficiency'][j] > MeanCurves['MeanEfficiency'][j-1] and \
            MeanCurves['MeanEfficiency'][j] > MeanCurves['MeanEfficiency'][j+1]:
                PostLocalMax = MeanCurves['MeanEfficiency'][j]
                PostLocalMaxTime = j
 
        for k in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
        (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'].strftime("%H:%M:%S"))].index.values.item(), \
        MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']].iloc[[0]].index.values.item(),-1):
            if MeanCurves['MeanEfficiency'][k] > MeanCurves['MeanEfficiency'][k-1] and \
            MeanCurves['MeanEfficiency'][k] > MeanCurves['MeanEfficiency'][k+1]:
                PreLocalMax = MeanCurves['MeanEfficiency'][k]
                PreLocalMaxTime = k
                Coefficients = np.polyfit([PostLocalMaxTime,PreLocalMaxTime], [PostLocalMax,PreLocalMax], 1)
                ExpectedEfficiency = Coefficients[0]*MeanCurves[(MeanCurves['ClientID']== \
                    SummaryWeek.loc[i*7,'ClientID']) & (MeanCurves['Time']== \
                    SummaryWeek.loc[i*7,'LocalMinRegTime'].strftime("%H:%M:%S"))] \
                    .index.values.item() + Coefficients[1]
                
                if PreLocalMax < PostLocalMax:
                    for l in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
                    (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'] \
                     .strftime("%H:%M:%S"))].index.values.item(), MeanCurves[MeanCurves['ClientID']== \
                     SummaryWeek.loc[i*7,'ClientID']].iloc[[-1]].index.values.item()):
                        if MeanCurves['MeanEfficiency'][l] > ExpectedEfficiency:
                            SummaryWeek.loc[i*7, 'ShadingLength'] = (l - PreLocalMaxTime)*15/60
                            break
                elif PreLocalMax > PostLocalMax:
                    for m in range(MeanCurves[(MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']) & \
                    (MeanCurves['Time']==SummaryWeek.loc[i*7,'LocalMinRegTime'] \
                     .strftime("%H:%M:%S"))].index.values.item(), MeanCurves[MeanCurves['ClientID']== \
                     SummaryWeek.loc[i*7,'ClientID']].iloc[[0]].index.values.item(),-1):
                        if MeanCurves['MeanEfficiency'][m] > ExpectedEfficiency:
                            SummaryWeek.loc[i*7, 'ShadingLength'] = (PostLocalMaxTime - m)*15/60
                            break

# Calculate shading severity
SummaryWeek['ShadingSeverity']=''
for i in range(0, int(len(MeanCurves)/96)):
    if SummaryWeek.loc[i*7, 'LocalMinRegTime'] == 'N/A':        
        SummaryWeek.loc[i*7,'ShadingSeverity'] = 'N/A'
    
    elif SummaryWeek.loc[i*7,'ShadingAmplitude'] <= 0.15 and SummaryWeek.loc[i*7, 'ShadingLength'] <= 1.5:
        SummaryWeek.loc[i*7,'ShadingSeverity'] = 'Mild'
    
    elif SummaryWeek.loc[i*7,'ShadingAmplitude'] >= 0.3 and SummaryWeek.loc[i*7, 'ShadingLength'] >= 3:
        SummaryWeek.loc[i*7,'ShadingSeverity'] = 'Severe'
    
    else:
        SummaryWeek.loc[i*7,'ShadingSeverity'] = 'Moderate'

# Calculate shading at sunrise/sunset
## Calculate optimum sunrise/sunset slopes
SunriseSlopeOpt = (PVSystemModel_Lisbon_20160801['Performance'][int(SunriseTime + ProdOffset*4)] - \
    PVSystemModel_Lisbon_20160801['Performance'][int(SunriseTime)])/(int(SunriseTime + ProdOffset*4) - int(SunriseTime))
SunsetSlopeOpt = (PVSystemModel_Lisbon_20160801['Performance'][int(SunsetTime)] - \
    PVSystemModel_Lisbon_20160801['Performance'][int(SunsetTime - ProdOffset*4)])/(int(SunsetTime) - int(SunsetTime - ProdOffset*4)) 

## Calculate client sunrise/sunset slopes
SummaryWeek['SunriseSlope']=''
for i in range(0, int(len(MeanCurves)/96)):                             
    SummaryWeek.loc[i*7,'SunriseSlope'] = (MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency'] \
    [i*96 + int(SunriseTime + ProdOffset*4)] - MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency'] \
    [i*96 + int(SunriseTime)])/(int(SunriseTime + ProdOffset*4) - int(SunriseTime))
    
SummaryWeek['SunsetSlope']=''
for i in range(0, int(len(MeanCurves)/96)):                             
    SummaryWeek.loc[i*7,'SunsetSlope'] = (MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency'] \
    [i*96 + int(SunsetTime)] - MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency'] \
    [i*96 + int(SunsetTime - ProdOffset*4)])/(int(SunsetTime) - int(SunsetTime - ProdOffset*4))
    
## Determine sunrise/sunset shading
SlopeThreshold = 0.4
SummaryWeek['SunriseShading']=''  
for i in range(0, int(len(MeanCurves)/96)):
    if np.max(MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']) <= 25:
        SummaryWeek.loc[i*7,'SunriseShading'] = 'N/A'
    elif abs(SummaryWeek.loc[i*7,'SunriseSlope']) <= abs(SlopeThreshold*SunriseSlopeOpt):
        SummaryWeek.loc[i*7,'SunriseShading'] = 'Yes'
    else:
        SummaryWeek.loc[i*7,'SunriseShading'] = 'No'
        
SummaryWeek['SunsetShading']=''  
for i in range(0, int(len(MeanCurves)/96)):
    if np.max(MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']) <= 25:
        SummaryWeek.loc[i*7,'SunsetShading'] = 'N/A'
    elif abs(SummaryWeek.loc[i*7,'SunsetSlope']) <= abs(SlopeThreshold*SunsetSlopeOpt):
        SummaryWeek.loc[i*7,'SunsetShading'] = 'Yes'
    else:
        SummaryWeek.loc[i*7,'SunsetShading'] = 'No'        
                   
## Calculate % of correctly identified clients with sunrise shading
SunriseShadingAnnotation=pd.read_csv('input/20160801-20160807/SunriseShading01_20160801-20160807.csv')
SunriseShadingCorrect=np.sum(SunriseShadingAnnotation.isin(np.asarray(SummaryWeek[SummaryWeek['SunriseShading']=='Yes']['ClientID']))).iat[0]/ \
    len(SunriseShadingAnnotation)*100
SunriseShadingFalsePositives=(len(SummaryWeek[SummaryWeek['SunriseShading']=='Yes'])- \
    np.sum(SunriseShadingAnnotation.isin(np.asarray(SummaryWeek[SummaryWeek['SunriseShading']=='Yes'] \
    ['ClientID']))).iat[0])/len(SummaryWeek[SummaryWeek['SunriseShading']=='Yes'])*100        

SunsetShadingAnnotation=pd.read_csv('input/20160801-20160807/SunsetShading01_20160801-20160807.csv')
SunsetShadingCorrect=np.sum(SunsetShadingAnnotation.isin(np.asarray(SummaryWeek[SummaryWeek['SunsetShading']=='Yes']['ClientID']))).iat[0]/ \
    len(SunsetShadingAnnotation)*100
SunsetShadingFalsePositives=(len(SummaryWeek[SummaryWeek['SunsetShading']=='Yes'])- \
    np.sum(SunsetShadingAnnotation.isin(np.asarray(SummaryWeek[SummaryWeek['SunsetShading']=='Yes'] \
    ['ClientID']))).iat[0])/len(SummaryWeek[SummaryWeek['SunsetShading']=='Yes'])*100                               
                                 
# Calculate orientation index
EffThreshold = 0.1   
SummaryWeek['SunriseOrientation']=''
for i in range(0, int(len(MeanCurves)/96)):
    if np.max(MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']) < 10:
        SummaryWeek.loc[i*7,'SunriseOrientation'] = 'N/A'
        
    else:
        x = PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance']/PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance'].max()>EffThreshold
        y = MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']/ \
            PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance'].max()>EffThreshold
        SummaryWeek.loc[i*7,'SunriseOrientation'] = (datetime.datetime.strptime(PVSystemModel_Lisbon_20160801['Time'] \
            [x[x].index[0]], '%H:%M:%S') - datetime.datetime.strptime(MeanCurves['Time'][y[y].index[0]], \
            '%H:%M:%S')).total_seconds()/60/60

SummaryWeek['SunsetOrientation']=''
for i in range(0, int(len(MeanCurves)/96)):
    if np.max(MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']) < 10:
        SummaryWeek.loc[i*7,'SunsetOrientation'] = 'N/A'
        
    else:
        x = PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance']/PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance'].max()>EffThreshold
        y = MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']/ \
            PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
            ['Performance'].max()>EffThreshold
        SummaryWeek.loc[i*7,'SunsetOrientation'] = (datetime.datetime.strptime(PVSystemModel_Lisbon_20160801['Time'] \
            [x[x].index[-1]], '%H:%M:%S') - datetime.datetime.strptime(MeanCurves['Time'][y[y].index[-1]], \
            '%H:%M:%S')).total_seconds()/60/60

SummaryWeek['OrientationIndex']=''
for i in range(0, int(len(MeanCurves)/96)):
    if np.max(MeanCurves[MeanCurves['ClientID']==SummaryWeek.loc[i*7,'ClientID']]['MeanEfficiency']) < 10:
        SummaryWeek.loc[i*7,'OrientationIndex'] = 'N/A'
    
    else:
        SummaryWeek.loc[i*7,'OrientationIndex'] = (SummaryWeek.loc[i*7,'SunriseOrientation'] + SummaryWeek.loc[i*7,'SunsetOrientation'])/2

# Calculate orientation severity
SummaryWeek['OrientationSeverity']=''
for i in range(0, int(len(MeanCurves)/96)):
    if SummaryWeek.loc[i*7,'OrientationIndex'] != 'N/A' and abs(SummaryWeek.loc[i*7,'OrientationIndex']) > 0 and \
    abs(SummaryWeek.loc[i*7,'OrientationIndex']) <= 1:
        SummaryWeek.loc[i*7,'OrientationSeverity'] = 'Mild'
    
    elif SummaryWeek.loc[i*7,'OrientationIndex'] != 'N/A' and abs(SummaryWeek.loc[i*7,'OrientationIndex']) > 1 and \
    abs(SummaryWeek.loc[i*7,'OrientationIndex']) <= 2:
        SummaryWeek.loc[i*7,'OrientationSeverity'] = 'Moderate'

    elif SummaryWeek.loc[i*7,'OrientationIndex'] != 'N/A' and abs(SummaryWeek.loc[i*7,'OrientationIndex']) > 2:
        SummaryWeek.loc[i*7,'OrientationSeverity'] = 'Severe'
    
    else:
        SummaryWeek.loc[i*7,'OrientationSeverity'] = 'N/A'
    
# Save summary dataframes
os.makedirs('summaries')
SummaryWeek.to_csv('summaries/SummaryWeek.txt', index=None, sep='\t')
SummaryDay.to_csv('summaries/SummaryDay.txt', index=None, sep='\t')

# Plot all curves
os.makedirs('plots')

f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(data5.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), data5.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/AllClients.pdf", bbox_inches='tight')

with PdfPages('plots/AllClients_separate.pdf') as pdf:
    for i in range(0, int(len(data5.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(data5.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), data5.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()

# Plot curves with sustained daytime zero production
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(ZeroProdSust.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), ZeroProdSust.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/ZeroProdSust_all.pdf", bbox_inches='tight')

with PdfPages('plots/ZeroProdSust_separate.pdf') as pdf:
    for i in range(0, int(len(ZeroProdSust.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(ZeroProdSust.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), ZeroProdSust.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()

# Plot curves with brief daytime zero production
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(ZeroProdBrief.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), ZeroProdBrief.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/ZeroProdBrief_all.pdf", bbox_inches='tight')

with PdfPages('plots/ZeroProdBrief_separate.pdf') as pdf:
    for i in range(0, int(len(ZeroProdBrief.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(ZeroProdBrief.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), ZeroProdBrief.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()
        
# Plot curves with local minima
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(LocalMin.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), LocalMin.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/LocalMin_all.pdf", bbox_inches='tight')

with PdfPages('plots/LocalMin_separate.pdf') as pdf:
    for i in range(0, int(len(LocalMin.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(LocalMin.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), LocalMin.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()

# Plot curves with regular local minima
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(LocalMinReg.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), LocalMinReg.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/LocalMinReg_all.pdf", bbox_inches='tight')

with PdfPages('plots/LocalMinReg_separate.pdf') as pdf:
    for i in range(0, int(len(LocalMinReg.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(LocalMinReg.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), LocalMinReg.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()

# Plot curves with irregular local minima
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(LocalMinIrreg.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), LocalMinIrreg.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/LocalMinIrreg_all.pdf", bbox_inches='tight')

with PdfPages('plots/LocalMinIrreg_separate.pdf') as pdf:
    for i in range(0, int(len(LocalMinIrreg.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(LocalMinIrreg.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), LocalMinIrreg.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()
        
# Plot curves without local minima
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(NoLocalMin.values.flatten())/96), 7):
    plt.plot(np.arange(96*7), NoLocalMin.values[i:i+7,:].flatten(), linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/NoLocalMin_all.pdf", bbox_inches='tight')

with PdfPages('plots/NoLocalMin_separate.pdf') as pdf:
    for i in range(0, int(len(NoLocalMin.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(NoLocalMin.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), NoLocalMin.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()
        
# Plot curves with low maximum production
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(LowMaxProd.values.flatten())/96), 7):
    plt.plot(np.arange(96*7),LowMaxProd.values[i:i+7,:].flatten(),linewidth=1.0)
    plt.plot(np.arange(96*7),[0.25/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[0.5/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[0.75/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[1/4]*96*7,c='k',ls=':',linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/LowMaxProd_all.pdf", bbox_inches='tight')

with PdfPages('plots/LowMaxProd_separate.pdf') as pdf:
    for i in range(0, int(len(LowMaxProd.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(LowMaxProd.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), LowMaxProd.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()
        
# Plot curves with high maximum production
f = plt.figure(figsize=(4,3))
plt.xlabel('Date',fontweight='bold')
plt.ylabel('Production (kWh)',fontweight='bold')
#clientlist=[]
for i in range(0, int(len(HighMaxProd.values.flatten())/96), 7):
    plt.plot(np.arange(96*7),HighMaxProd.values[i:i+7,:].flatten(),linewidth=1.0)
    plt.plot(np.arange(96*7),[0.25/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[0.5/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[0.75/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[1/4]*96*7,c='k',ls=':',linewidth=1.0)
    plt.plot(np.arange(96*7),[1.25/4]*96*7,c='k',ls=':',linewidth=1.0)
plt.xlim([0, 96*7])
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
f.savefig("plots/HighMaxProd_all.pdf", bbox_inches='tight')

with PdfPages('plots/HighMaxProd_separate.pdf') as pdf:
    for i in range(0, int(len(HighMaxProd.values.flatten())/96), 7):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(HighMaxProd.index.get_level_values(0)[i]))
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96*7), HighMaxProd.values[i:i+7,:].flatten(), linewidth=1.0)
        plt.xlim([0, 96*7])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 96*8, 96),pd.date_range(pd.datetime(2016, 8, 1),periods=8).date,rotation=45)
        plt.tight_layout()
        pdf.savefig()
        
# Plot one-week mean production curves
with PdfPages('plots/MeanProd_separate.pdf') as pdf:
    for i in range(0, int(len(MeanCurves)/96)):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(MeanCurves.reset_index()['ClientID'][96*i]))
        plt.xlabel('Time',fontweight='bold')
        plt.ylabel('Production (kWh)',fontweight='bold')
        plt.plot(np.arange(96), MeanCurves['MeanProd'][96*i:96*(i+1)], linewidth=1.0)
        plt.xlim([0, 96])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(0, 97, 96/4),pd.date_range("2016-08-01 0:00", "2016-08-02 00:00", freq="360min"). \
           strftime('%H:%M'))
        plt.tight_layout()
        pdf.savefig()

# Plot one-week mean efficiency curves (with optimum curve)
with PdfPages('plots/MeanEfficiency_separate.pdf') as pdf:
    for i in range(0, int(len(MeanCurves)/96)):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(MeanCurves.reset_index()['ClientID'][96*i]))
        plt.xlabel('Time',fontweight='bold')
        plt.ylabel('Efficiency (%)',fontweight='bold')
        plt.plot(np.arange(96), MeanCurves['MeanEfficiency'][96*i:96*(i+1)], linewidth=1.0)
        plt.plot(np.arange(96), PVSystemModel_Lisbon_20160801[PVSystemModel_Lisbon_20160801['Date']=='2016-08-01'] \
                 ['Performance'][0:96], c='k', ls='--', linewidth=1.0)
        plt.xlim([0, 96])
        plt.ylim([0, 100])
        plt.xticks(np.arange(0, 97, 96/4),pd.date_range("2016-08-01 0:00", "2016-08-02 00:00", freq="360min"). \
           strftime('%H:%M'))
        plt.tight_layout()
        pdf.savefig()
    
# Plot one-week mean efficiency curves (without optimum curve)
with PdfPages('plots/MeanEfficiency_separate_no optimum.pdf') as pdf:
    for i in range(0, int(len(MeanCurves)/96)):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(MeanCurves.reset_index()['ClientID'][96*i]))
        plt.xlabel('Time',fontweight='bold')
        plt.ylabel('Efficiency (%)',fontweight='bold')
        plt.plot(np.arange(96), MeanCurves['MeanEfficiency'][96*i:96*(i+1)], linewidth=1.0)
        plt.xlim([0, 96])
        plt.ylim([0, 100])
        plt.xticks(np.arange(0, 97, 96/4),pd.date_range("2016-08-01 0:00", "2016-08-02 00:00", freq="360min"). \
           strftime('%H:%M'))
        plt.tight_layout()
        pdf.savefig()    
        
# Plot one-week mean slope curves
with PdfPages('plots/MeanSlope_separate.pdf') as pdf:
    for i in range(0, int(len(MeanCurves)/96)):
        f = plt.figure(figsize=(4,3))
        plt.title('{}'.format(MeanCurves.reset_index()['ClientID'][96*i]))
        plt.xlabel('Time',fontweight='bold')
        plt.ylabel('Slope ($\mathregular{min^{-1}}$)',fontweight='bold')
        plt.plot(np.arange(96), MeanCurves['MeanSlope'][96*i:96*(i+1)], linewidth=1.0)
        plt.xlim([0, 96])
        plt.ylim([-1, 1])
        plt.xticks(np.arange(0, 97, 96/4),pd.date_range("2016-08-01 0:00", "2016-08-02 00:00", freq="360min"). \
           strftime('%H:%M'))
        plt.tight_layout()
        pdf.savefig()