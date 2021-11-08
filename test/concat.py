import pandas as pd
dic = {str(i):[i+20]*2 for i in range(1,5+1)}
df_dic = pd.DataFrame(dic)
df_dic = df_dic.T
df_dic.index = df_dic.index.astype(int)-1
df_dic = df_dic.sort_index(axis=0,ascending=True)
print(df_dic)
df_attr = pd.read_csv('attr.txt',sep=',', header=None)
print(df_attr)
df = pd.concat([df_dic,df_attr],axis=1)
print(df)

df.to_csv('feat.csv',header=None,index=None)
df.to_csv('feat.txt',header=None,index=None)