import pandas as pd
data = pd.read_csv('/Users/hhhwa/Desktop/oss/OSS기말과제/2019_kbo_for_kaggle_v2.csv')
data_1_1=data[data.p_year >= 2015]
data_1_1=data_1_1[data_1_1.p_year <= 2018]
data_1_1_H = data_1_1.loc[:,['H','batter_name','p_year']]
data_1_1_avg = data_1_1.loc[:,['avg','batter_name','p_year']] 
data_1_1_HR = data_1_1.loc[:,['HR', 'batter_name','p_year']]
data_1_1_OBP = data_1_1.loc[:,['OBP','batter_name','p_year']]

data_1_1_H=data_1_1_H.sort_values(by = 'H', ascending = False).head(10)
data_1_1_H['H']=data_1_1_H['H'].rank(method='max', ascending=False)
H_Ranking ={'H': data_1_1_H['H']}
data_1_1_H.drop(['H'],axis='columns',inplace=True)
data_1_1_H.index= H_Ranking['H']

data_1_1_avg=data_1_1_avg.sort_values(by = 'avg', ascending = False).head(10)
data_1_1_avg['avg']=data_1_1_avg['avg'].rank(method='max', ascending=False)
avg_Ranking ={'avg': data_1_1_avg['avg']}
data_1_1_avg.drop(['avg'],axis='columns',inplace=True)
data_1_1_avg.index= avg_Ranking['avg']

data_1_1_HR=data_1_1_HR.sort_values(by = 'HR', ascending = False).head(10)
data_1_1_HR['HR']=data_1_1_HR['HR'].rank(method='max',ascending=False)
HR_Ranking ={'HR': data_1_1_HR['HR']}
data_1_1_HR.drop(['HR'],axis='columns',inplace=True)
data_1_1_HR.index= HR_Ranking['HR']

data_1_1_OBP=data_1_1_OBP.sort_values(by = 'OBP', ascending = False).head(10)
data_1_1_OBP['OBP']=data_1_1_OBP['OBP'].rank(method='max', ascending=False)
OBP_Ranking ={'OBP': data_1_1_OBP['OBP']}
data_1_1_OBP.drop(['OBP'],axis='columns',inplace=True)
data_1_1_OBP.index= OBP_Ranking['OBP']

print(data_1_1_H, '\n')
print(data_1_1_avg, '\n')
print(data_1_1_HR, '\n')
print(data_1_1_OBP, '\n')

data_1_2=data[data.p_year == 2018]
data_1_2=data_1_2.sort_values('war', ascending= False)
data_1_2 = data_1_2.loc[:,['batter_name','cp','war','p_year']]
data_1_2=data_1_2.drop_duplicates(['cp'], keep='first')
idx =  data_1_2[data_1_2['cp'] == "지명타자"].index
data_1_2.drop(idx , inplace=True)
data_1_2 = data_1_2.reset_index(drop=True)
print(data_1_2, '\n')

data_1_3={'Field' : ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG'],
    'Correlation' : [data['salary'].corr(data['R']),
                    data['salary'].corr(data['H']),
                    data['salary'].corr(data['HR']),                 
                    data['salary'].corr(data['RBI']),                 
                    data['salary'].corr(data['SB']),                 
                    data['salary'].corr(data['war']),                 
                    data['salary'].corr(data['avg']),                
                    data['salary'].corr(data['OBP']),               
                    data['salary'].corr(data['SLG'])]}
frame=pd.DataFrame(data_1_3)
frame = frame.sort_values(by = 'Correlation', ascending = False)
frame = frame.reset_index(drop=True)
print(frame)
print('The highest correlation with salary is', frame['Field'][0], '\n')