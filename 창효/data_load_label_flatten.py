#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:53:31 2018

@author: choigirhyo
"""

import os
import csv
import pandas as pd
import numpy as np
path = '/Users/choigirhyo/Downloads/2018빅콘테스트/data/'
def read_csv(file, stop_col=None, mode='r'):
    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data_values, line_count = [], 0        
        for row in csv_reader:
            if line_count == 0:
                data_columns = row
            elif line_count == stop_col:
                break
            else:
                data_values.append(row)
            line_count += 1            
        data = pd.DataFrame(data_values, columns = data_columns)    
    return data
    
def read_data_folder(path, stop_col=None):
    data_list = {}
    for file in os.listdir(path):
        data_name = file.split('.')[0].split('_')[1]        
        if stop_col==None:
            data_list[data_name] = read_csv(path + '/%s' %(file))
        else:
            data_list[data_name] = read_csv(path + '/%s' %(file), stop_col=stop_col)    
    return data_list
            
            
train = read_data_folder(path + 'train', 10000)    
train['activity'].to_csv(path + 'activity.csv', index=None)
train['payment'].to_csv(path + 'payment.csv', index=None)
train['activity'] = pd.read_csv(path + 'activity.csv')
train['payment'] = pd.read_csv(path + 'payment.csv')
###############################################################################################################
# 사용자별 활동내역 추출
id_list = list(set(train['activity']['acc_id']))


label = read_csv(path+'train/train_label.csv')
from pandas import DataFrame
import pandas as pd
id_activity = {}
#########################만들때 label도 같이 추가
for acc_id in id_list:
    df = train['activity'][train['activity']['acc_id'] == acc_id]
    empty_df = DataFrame(index = range(len(df)),columns = ['label'])
    empty_df = label[label['acc_id'] == acc_id]['label']
    df = df.reset_index(drop = True)
    empty_df = empty_df.reset_index(drop = True)
    id_activity[acc_id] = pd.concat([df,empty_df],axis = 1)
    id_activity[acc_id] = id_activity[acc_id].fillna(method = 'ffill')
    


 # merge payment 
for acc_id in id_list:
    id_activity[acc_id] = pd.merge(id_activity[acc_id], train['payment'], how='left', left_on=['wk','acc_id'], right_on=['payment_week','acc_id'])
id_activity[id_list[0]].columns
###############################################################################################################
def make_index(id_activity):    
    sum_list = {'playing' : ['cnt_dt', 'play_time', 'game_combat_time'],
                'exp' : ['npc_exp', 'npc_hongmun', 'quest_exp', 'quest_hongmun', 'item_hongmun'],
                'inzone_cnt' : ['cnt_enter_inzone_solo', 'cnt_enter_inzone_light', 'cnt_enter_inzone_skilled', 'cnt_enter_inzone_normal'],
                'inzone_clear' : ['cnt_clear_inzone_solo', 'cnt_clear_inzone_light', 'cnt_clear_inzone_skilled', 'cnt_clear_inzone_normal'],
                'raid_cnt' : ['cnt_enter_raid', 'cnt_enter_raid_light', 'cnt_enter_bam'],
                'raid_clear' : ['cnt_clear_raid', 'cnt_clear_raid_light', 'cnt_clear_bam'],
                'chat' : ['normal_chat', 'whisper_chat', 'district_chat', 'party_chat', 'guild_chat', 'faction_chat'],
                'living' : ['gathering_cnt', 'making_cnt'],
                'payment' : ['payment_amount'],            
                }
    multi_list = {'duel' : ['duel_cnt', 'duel_win'],
                  'battle' : ['partybattle_cnt', 'partybattle_win'],
                  'inzone' : ['inzone_cnt_index', 'inzone_clear_index'],
                  'raid' : ['raid_cnt_index', 'raid_clear_index']}
    
    
    for acc_id in id_list:
        for col in sum_list.keys():
            id_activity[acc_id]['%s_index' %(col)] = np.sum(id_activity[acc_id][sum_list[col]], axis=1)
            
    for acc_id in id_list:
        for col in multi_list.keys():
            id_activity[acc_id]['%s_index' %(col)] = id_activity[acc_id][multi_list[col][0]] * id_activity[acc_id][multi_list[col][1]]
    
    
    
    id_summary = {}
    index_list = ['wk', 'playing_index', 'exp_index', 'inzone_index', 'raid_index',
                  'chat_index', 'living_index', 'payment_index','label']
    for acc_id in id_activity.keys():
        data = {}
        for index in index_list:
            data[index] = id_activity[acc_id][index]
        id_summary[acc_id] = pd.DataFrame(data)
        
        
    return id_summary
            
id_summary = make_index(id_activity)
###############################################################################################################
def scale(dataframe):
    for col in dataframe.columns:
        if col=='wk':
            pass
        elif col=='label':
            pass
        else:
            col_max = dataframe[col].max()
            col_min = dataframe[col].min()
            dataframe[col] = (dataframe[col] - col_min) / (col_max - col_min)
            dataframe = dataframe.fillna(0)
    return dataframe
    
for acc_id in id_summary.keys():
    id_summary[acc_id] = scale(id_summary[acc_id])
###############################################################################################################



def Flatten(id_summary):
    data = DataFrame()
    for _,v in enumerate(id_summary):
        ex = id_summary[v] #데이터
        value = ex.drop(['label','wk'],axis = 1) #값
        weight = np.transpose(DataFrame(index= ['Weight'],
                                columns = ex[ex['wk'] == 1].drop(['label','wk'],axis = 1).columns,
                                data = [[1,1,1,1,1,1,1]])) #가중치 바꾸고 싶으면 여기 값 변경
        final_value = DataFrame(np.matmul(value,weight)) #매트릭스 곱셈
        wk = ex['wk']
        df = pd.concat([final_value,wk],axis = 1)
        df.columns = [v,'wk']
        
        nan = float('Nan') #플레이 안했음을 뭐로 표현할지 (0으로 해도 상관없음) //-값을 줘도 괜찮을듯
        Non_play_dataframe = DataFrame(columns =[v,'wk'],data = [[nan,1],[nan,2],[nan,3],[nan,4],[nan,5],[nan,6],[nan,7],[nan,8]])


        final_df = pd.concat([df,Non_play_dataframe[-Non_play_dataframe['wk'].isin(df['wk'])]])#없는 wk 는 NA값으로 넣기
        final_df = final_df.sort_values(by =['wk']).reset_index(drop = True) #순서정렬
        final_df = np.transpose(final_df) #행렬 전치
        
        final_df[8] = [ex['label'][0],'label'] #맨 끝에 라벨 붙이기
        final_df.columns = final_df.iloc[1,:] #wk값을 column이름으로 바꾸기
        final_df = final_df.drop('wk') #wk행 삭제
        data = data.append(final_df)
    return data


data = Flatten(id_summary)


###데이터 살펴보기
retain = data[data['label'] == 'retained']
not_retain = data[~(data['label'] == 'retained')]
week = data[data['label'] == 'week']
month = data[data['label'] == 'month']
two_month = data[data['label'] == '2month']

###분류모델
from sklearn.model_selection import train_test_split
#일단은 Nan을 0으로
data = data.replace([float('Nan')],[0])
X = data.drop('label',axis = 1)
Y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#fit&predict
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
#결과
from sklearn.metrics import confusion_matrix
result = confusion_matrix(y_test,pred)
score = (result[0,0]+result[1,1]+result[2,2]+result[3,3])/sum(sum(result))








##loss function 짜보기
def Flatten(id_summary,weight_value):
    data = DataFrame()
    for _,v in enumerate(id_summary):
        ex = id_summary[v] #데이터
        value = ex.drop(['label','wk'],axis = 1) #값
        weight = np.transpose(DataFrame(index= ['Weight'],
                                columns = ex[ex['wk'] == 1].drop(['label','wk'],axis = 1).columns,
                                data = weight_value)) #가중치 바꾸고 싶으면 여기 값 변경
        final_value = DataFrame(np.matmul(value,weight)) #매트릭스 곱셈
        wk = ex['wk']
        df = pd.concat([final_value,wk],axis = 1)
        df.columns = [v,'wk']
        
        nan = float('Nan') #플레이 안했음을 뭐로 표현할지 (0으로 해도 상관없음) //-값을 줘도 괜찮을듯
        Non_play_dataframe = DataFrame(columns =[v,'wk'],data = [[nan,1],[nan,2],[nan,3],[nan,4],[nan,5],[nan,6],[nan,7],[nan,8]])


        final_df = pd.concat([df,Non_play_dataframe[-Non_play_dataframe['wk'].isin(df['wk'])]])#없는 wk 는 NA값으로 넣기
        final_df = final_df.sort_values(by =['wk']).reset_index(drop = True) #순서정렬
        final_df = np.transpose(final_df) #행렬 전치
        
        final_df[8] = [ex['label'][0],'label'] #맨 끝에 라벨 붙이기
        final_df.columns = final_df.iloc[1,:] #wk값을 column이름으로 바꾸기
        final_df = final_df.drop('wk') #wk행 삭제
        data = data.append(final_df)
    return data

def loss_function(weight_value):
    data = Flatten(id_summary,weight_value)
    data = data.replace([float('Nan')],[0])
    X = data.drop('label',axis = 1)
    Y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state = 42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    pred = gnb.predict(X_test)
    result = confusion_matrix(y_test,pred)
    score = (result[0,0]+result[1,1]+result[2,2]+result[3,3])/sum(sum(result))
    return 1-score

1.loss_function을 미분해서 최솟값을 찾는다 ->1차함수라서 안될 거 같다
2.강화학습으로 최적의 acc가 나올 때 까지x1,x2,...,x7의 값을 변화시킨다
    7개의 변수를 어떻게 세팅하느냐에 따라 다른 return값을 준다
    해당 return값이 최소가 되도록 하는 변수를 찾아나가자