import numpy as np
import pandas as pd
na_list=['?','notpresent']
df=pd.read_csv('data.csv',na_values=na_list) 
# print(df.head())
# print(df.size)
# missing_values=df.isnull()
# print(missing_values)
# print()
# missing_rows=df.isnull().sum()
# print(missing_rows)
# print()
# missing_columns=df.isnull().sum(axis=1)
# print(missing_columns)
mapping={'present':1 , 'notpresent':0}
df['ba']=df['ba'].replace(mapping)
# df.drop(columns=['Unnamed: 0.1'], inplace=True)
df.to_csv('data.csv',index=False)