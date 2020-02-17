#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:52:26 2020

@author: youwu
"""

import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
#jupyter notebook: %matplotlib inline

import pymysql
from sqlalchemy import create_engine

#Numpy
arr1 = np.array([[1,2,3],[2,3,4]])
arr2 = np.arange(1,10,2)
arr3 = np.linspace(1,20,10)
arr4 = np.linspace(1,20,10, endpoint = False) #default endpoint = True
arr0 = np.zeros([4,5,2], dtype = int)
arrone = np.ones([2,3]) + 1
dim0 = arr0.ndim # 3
shape0 = arr0.shape # (4,5,2)
size0 = arr0.size # 40
type0 = arr0.dtype

data2 = ((1,4,2,8),(5,-3,4,1),(3,8,4,6),(2,5,6,3))
arr = np.array(data2)

s = np.array([1,5,2,8,2,4,5,9,5,7,8,0,12])
s1 = np.sort(s)
sr = np.array(sorted(s, reverse = True))
s_pos = np.argsort(s)


np.sort(arr) #default axis = 1
np.sort(arr, axis = 0) #axis=0 按行操作， axis=1按列操作

np.where(s > 3, 1, 0)
np.where(arr > 3, arr, 0)
np.extract(s>3, arr) 


# Series
series1 = pd.Series([2.8, 3.01, 8.99, 5.85, 5.18])
series2 = pd.Series([2.8, 3.01, 8.99, 5.85, 5.18], index = ['a','b','c', 'd','e'], name = 'test series')
series3 = pd.Series({'beijing':2.8, 'shanghai':3.01, 'wuhan':8.99, 'hefei':5.85, 'uppsala':5.18}, name = 'city series')

series3['beijing':'wuhan']
series3.values
series3.index

#Dataframe

list1 = [['wuyou',23,'male'], ['xiajing',24,'female'], ['jarmo',53,'male']]
df1 = pd.DataFrame(list1, columns = ['Name','Age','Sex'])
df1.head(2)

df2 = pd.DataFrame({'Name':['you','xiajing','jarmo'], 'Age':[23,24,53], 'Sex':['male', 'female', 'male']})

array1 = np.array([['wuyou',23,'male'], ['xiajing',24,'female'], ['jarmo',53,'male']])
df3 = pd.DataFrame(array1, columns = ['Name','Age','Sex'], index = ['a','b','c'])

df3.values
df3.index
df3.columns.tolist()

df3.dtypes #from array
df2.dtypes #from list


###Files handling
'''
Encoding: utf-8, gbk, gbk2313, gb18030
'''
baby0 = pd.read_csv('Desktop/python_data_cleasing/notebook_data/sam_tianchi_mum_baby.csv', encoding = 'utf-8') #default=utf-8
baby0.head()
order = pd.read_csv('Desktop/python_data_cleasing/notebook_data/meal_order_info.csv', 
                    encoding = 'gbk', dtype={'info_id':str, 'emp_id':str})
order.head(10)
baby = pd.read_csv('Desktop/python_data_cleasing/notebook_data/baby_trade_history.csv', dtype = {'user_id':str})#nrows = 100)
baby.info()
# pd.set_option('display.max_columns',20)
#pd.set_option('display.max_rows',30) probably used in notebook
#baby.to_csv('path', encoding = 'gbk'or'utf-8', index = False) #Write to a new csv file
baby.iloc[0]

meal_order = pd.read_excel('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx', 
                           encoding = 'utf-8', sheet_name = 'meal_order_detail1')
#meal_order.to_excel('path',index = False, sheet_name = 'foo')


#SQL databese if only a database exists
connection = create_engine('mysql+pymysql://root:youwu@localhost:3306/test1')

order.to_sql('testdf', con=connection, index=False, if_exists='replace') #replace, append, fail
df_sql = pd.read_sql('select * from meal_order_info', con = connection)

def query(table):
    '''
    Query a table in a sql database.
    '''
    host = 'localhost'
    user = 'root'
    password = 'pwd'
    database = 'db_name'
    port = 3306
    conn = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(user, password, host, port, database))
    #SQL语句，可以定制，实现灵活查询
    sql = 'select * from ' + table  #选择数据库中表名称    
    # 使用pandas 的read_sql函数，可以直接将数据存放在dataframe中
    results = pd.read_sql(sql,conn)
    return results
#query(meal_order_info)

baby.columns
baby[['user_id','cat_id']]
baby[['user_id','cat_id']][1:5]
#df.loc 选择标签 para1:row index, para2:columns
baby.loc[1:3]
baby.loc[1:3, ['user_id','cat_id']]
baby.loc[baby.user_id=='532110457', 'buy_mount']
baby.loc[(baby.user_id=='532110457') | (baby.buy_mount>100), ['user_id','buy_mount']]
baby.loc[(baby.buy_mount>100) & (baby.buy_mount<110), ['user_id','buy_mount']]
baby[baby.buy_mount.between(100,150)]
#df.iloc 选择位置
baby.iloc[1:3]

baby['buy_degree'] = np.where(baby.buy_mount>3,'High','Low') #inserting a column 

cat_id = baby.cat_id
del baby['cat_id'] #deleting a column
baby.insert(0,'cat_id_new',cat_id) #insert a column by (position, name, data)

#Deleting. axis=0: row-wise; inplace=False: create copy not change original dataframe
#           default axis=0, inplace=False
baby.drop(labels=['buy_degree'], axis=1, inplace=True) 
baby.drop(labels=[0,1], axis=0, inplace=True)

#Modifying data in dataframe 
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/sam_tianchi_mum_baby.csv', encoding = 'utf-8', dtype = str)
df.loc[df.gender=='0','gender'] = 'female'
df.loc[df.gender=='1','gender'] = 'male'
df.loc[df.gender=='2','gender'] = 'unknown'
df.rename(columns={'user_id':'User_id', 'birthday':'Birthday', 'gender':'Gender'}, inplace=True)
df.rename(index={0:'a',1:'b'},inplace=True) #rename row index
df.reset_index(drop=True, inplace=True) #reset row index to 0 ~ num_row-1

#selecting data
baby[baby.buy_mount.between(100,150,inclusive=True)]
baby[(baby['buy_mount']>1000) & (baby['day']>20141023)]
baby[baby.day.isin([20131011,20141011])]

df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/baby_trade_history.csv', dtype = {'user_id':str})#nrows = 100)
df[~(df.buy_mount<1000)]

workbook= xlrd.open_workbook('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx')
sheet_names = workbook.sheet_names()

order1  = pd.read_excel('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx', sheet_name = sheet_names[0])
order2  = pd.read_excel('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx', sheet_name = sheet_names[1])
order3  = pd.read_excel('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx', sheet_name = sheet_names[2])
order = pd.concat([order1,order2,order3],axis=0,ignore_index=True)
#OR better as below
order = pd.DataFrame()
for sheet_name in sheet_names:
    basic =  pd.read_excel('Desktop/python_data_cleasing/notebook_data/meal_order_detail.xlsx', sheet_name = sheet_name)
    order = pd.concat([order,basic],axis=0,ignore_index=True)
order.reset_index(inplace=True)

#Merge dataframe  
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/baby_trade_history.csv', dtype = {'user_id':str})#nrows = 100)
df1 = pd.read_csv('Desktop/python_data_cleasing/notebook_data/sam_tianchi_mum_baby.csv', dtype = {'user_id':str})
df2 = pd.merge(left=df,right=df1,how='inner',left_on='user_id',right_on='user_id')

#multi-indexes
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/baby_trade_history.csv', 
                 dtype = {'user_id':str}, index_col=[3,0]) #using the 4th and the 1st col as row index
df.loc[(28,[348660284,128447452]),]

#datetime
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/baby_trade_history.csv', dtype = {'user_id':str})#nrows = 100)
df.info()
df['buy_date'] = pd.to_datetime(df.day, format='%Y%m%d', errors='coerce')
df.buy_date.dt.year#year/month/day
df['diff_day'] = pd.datetime.now() - df.buy_date
df.diff_day.dt.days #days,seconds, microseconds
df['diff_time'] = df.diff_day / pd.Timedelta('1 D') #Year, Month, Day, Hour, Minute, Second, MicroSecond
df.diff_time.round(decimals=3)
df.diff_day.astype('timedelta64[D]')

#functional processing
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/sam_tianchi_mum_baby.csv', dtype = str)
def def_sex(x):
    if x == '0':
        return 'female'
    elif x == '1':
        return 'male'
    else:
        return 'unknown'

df['sex'] = df.gender.apply(def_sex)
del df['sex']
df['sex'] = df.gender.map({'0':'female', '1':'male','2':'unknown'})
del df['sex']
df['sex'] = df.gender.map(def_sex)

df['user'] = df.user_id.apply(lambda x: x.replace(x[1:3],'**'))
df.birthday.apply(lambda x: x[0:4])

#string
df1 = pd.read_csv('Desktop/python_data_cleasing/notebook_data/MotorcycleData.csv', encoding = 'gbk')
df1.Price
df1['my_price'] = df1.Price.str.strip('$')
df1['my_price'] = df1.my_price.str.replace(',','')
df1.Location.str.split(',').str[0]

#分组groupby/ group.sum()/max()/min()/mean()/median()/count()
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/online_order.csv', encoding = 'gbk')
df['customer'] = df['customer'].astype('str')
df['order'] = df['order'].astype('str')
group1 = df.groupby('weekday')
group1.mean()['total_items']
group2 = df.groupby(by =  ['customer','weekday'])
group2.sum()['total_items'] 
#聚合agg
group1.agg([np.mean,np.max,np.min])
group1.agg({'total_items':np.sum,'Food%':[np.mean,np.max]})
df[['total_items','Food%','Drinks%']].agg([np.sum,np.mean,np.max])
#groupby.apply()
group1.apply(np.mean)[['Food%', 'Fresh%']]
var = ['Food%', 'Fresh%', 'Drinks%', 'Home%', 'Beauty%', 'Health%', 'Baby%', 'Pets%']
df[var].apply(np.sum,axis=0)
df[var].apply(lambda x:x[0]-x[1], axis=1)
#table&graph pivot_table
pd.pivot_table(data=df,index='weekday',values='total_items',aggfunc=[np.sum,np.max],
               margins=True,margins_name='total') #or df.pivot_table(#without 'data=df')
pd.pivot_table(data=df,index='weekday',columns='customer',values='total_items',aggfunc=np.sum,
               margins=True,margins_name='total',fill_value=0) #or df.pivot_table(#without 'data=df')
pd.pivot_table? #Documentations

pd.crosstab(index=df.weekday, columns=df['discount%'])
pd.crosstab(index=df.weekday, columns=df['discount%'], margins=True, normalize='columns')#normalize=all/index/columns

#Cleansing
df = pd.read_csv('Desktop/python_data_cleasing/notebook_data/MotorcycleData.csv', encoding = 'gbk',
                 na_values='Na')
#重复
def f(x):
    if '$' in str(x):
        x = str(x).strip('$')
    x = str(x).replace(',','')
    return float(x)

df['Price'] = df.Price.apply(f)
df['Mileage'] = df.Mileage.apply(f)
df[df.duplicated()]
np.sum(df.duplicated())
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['Condition', 'Condition_Desc', 'Price', 'Location'],inplace=True)
#缺失
df.apply(lambda x: sum(x.isnull())/len(x), axis=0)
df.dropna(how='all',axis=0,inplace=True)
df.dropna(how='any',subset=['Condition','Price','Mileage'],axis=0,inplace=True)
df.Mileage.fillna(df.Mileage.mean(),inplace=True)
df['Exterior_Color'].mode()[0] #mode众数
df.Exterior_Color.fillna(df['Exterior_Color'].mode()[0],inplace=True)
df.fillna(value={'Exterior_Color':'Black','Mileage':df.Mileage.median()},inplace=True)
df['Exterior_Color'].fillna(method='ffill') #前向填补
df['Exterior_Color'].fillna(method='bfill') #后向填补
#异常
x_bar = df['Price'].mean()
x_std = df['Price'].std() #标准差
#异常判断标准：均值加减2(2.5/3)倍标准差
any(df['Price'] > x_bar+2*x_std)
any(df['Price'] < x_bar-2*x_std)
df['Price'].describe()
Q1 = df['Price'].quantile(q=0.25) #1/4分位数
Q3 = df['Price'].quantile(q=0.75) #3/4分位数
IQR = Q3 - Q1 #分位差
any(df['Price'] > Q3 + 1.5*IQR)
any(df['Price'] < Q1 - 1.5*IQR)
df[df['Price'] > Q3 + 1.5*IQR]['Price']
df['Price'].plot(kind='box') #箱线图

plt.style.use('seaborn')
df['Price'].plot(kind='hist',bins=30,density=True)
df['Price'].plot('kde') #曲线拟合
#盖帽法处理异常值
p99 = df['Price'].quantile(q=0.99)
p1 = df['Price'].quantile(q=0.01)
df['price_new'] = df['Price']
df.loc[df['Price']>p99,'price_new'] = p99
df.loc[df['Price']<p1,'price_new'] = p1
df[['Price','price_new']].describe()
df['price_new'].plot(kind='box')
#数据离散化/分箱 pd.cut
df['price_bin'] =  pd.cut(df['price_new'],5,labels=range(5)) #default 等宽分箱
df['price_bin'].hist()

w = [100,1000,5000,10000,20000,50000]
df['price_bin'] =  pd.cut(df['price_new'], bins =w,labels=['low','cheep','reasonable','medium','high'],right=False)
df['price_bin'].value_counts()
#等频分箱 pd.qcut
k = 5
w = [1.0*i/k for i in range(k+1)] 
df['price_bin'] = pd.qcut(df['price_new'],q=w,labels=range(k))
df['price_bin'].hist()
#OR
w1 = df['price_new'].quantile([1.0*i/k for i in range(k+1)])#先计算分位数,在进行分段
w1[0] = w1[0]* 0.95 # 最小值缩小一点
w1[-1] = w1[1]* 1.05 # 将最大值增大一点, 目的是为了确保数据在这个范围内
df['price_bin'] = pd.cut(df['price_new'],bins=w1,labels=range(k))
df['price_bin'].hist()





