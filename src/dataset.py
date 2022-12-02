import random
import numpy as np
import torch
import matplotlib.pyplot as plt
# multivariate data preparation
from numpy import array
from numpy import hstack
from tqdm import tqdm

from itertools import groupby
from operator import itemgetter
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
import time
from collections import Counter

import sys

# import EarlyStopping
from sklearn.metrics import mean_squared_error



def remove_problematic_sequnces(df, det_id, list_months):
    '''data curation based on visual analysis
    '''
    temp_df  = df[df.paris_id==det_id]
    temp_df = temp_df[~temp_df.month.isin(list_months)]
    filtered_df = pd.concat([df[df.paris_id!=det_id], temp_df], axis=0)
    return filtered_df

def split_sequences(sequences, static, id_det, n_steps, p_horizon, auto_regressive=False):
    '''
    split a multivariate sequence into samples
    '''
    W, X, y, z = list(), list(), list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix + p_horizon > len(sequences):
            break
        # gather input and output parts of the pattern
        if not auto_regressive:
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:end_ix+p_horizon, -1]
        else:
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:end_ix+p_horizon, -1]
        X.append(seq_x)
        y.append(seq_y)
        W.append(static)
        z.append(id_det)
    return array(W), array(X), array(y), array(z)

def format_data(df, 
                lookback_timesteps, 
                prediction_horizon,
                features_static, 
                features_dynamic,
                auto_regressive=False):
    '''
    new implementation considering only month as static
    '''
    W_list, X_list, y_list, z_list = list(), list(), list(), list()
    for i in tqdm(df.paris_id.unique()):
        temp = df[df.paris_id==i]
        for month in temp.month.unique():
            temp_m = temp[temp.month==month]
            
            # deactivating within day as a fixed feature
            # as it leads to loss of data for first few hours of 
            # each day            
            #######################################
            temp_m = temp_m.sort_values(by="time_idx")
            temp_m.index=temp_m.time_idx
            #######################################
            
#             for day in temp_m.day.unique():
#                 temp_md = temp_m[temp_m.day==day]
#                 temp_md = temp_md.sort_values(by="time_idx")
#                 print(temp.columns)
#                 temp_md.index=temp_md.time_idx
            #     for t in range(min(temp.time_idx), max(temp.time_idx)-lookback_timesteps):    
            #     time_indices = list(range(t, t+lookback_timesteps))
                
            w = np.array(temp_m[features_static].drop_duplicates())[0]
            for k, g in groupby(enumerate(list(temp_m.index)), lambda ix : ix[0] - ix[1]):
                temp_list =list(map(itemgetter(1), g)) 

                if len(temp_list)>lookback_timesteps:
                    temp_df = temp_m.loc[temp_list, features_dynamic]
                    W, X, y, z = split_sequences(np.array(temp_df), w, i, 
                                                 lookback_timesteps, prediction_horizon, auto_regressive)
                    W_list.extend(W)
                    X_list.extend(X)
                    y_list.extend(y)
                    z_list.extend(z)
    return W_list, X_list, y_list, z_list

def deprecated_format_data(df, 
                            lookback_timesteps, 
                            prediction_horizon,
                            features_static, 
                            features_dynamic):
    W_list, X_list, y_list, z_list = list(), list(), list(), list()
    for i in tqdm(df.paris_id.unique()):
        temp = df[df.paris_id==i]
        for month in temp.month.unique():
            temp_m = temp[temp.month==month]
            for day in temp_m.day.unique():
                temp_md = temp_m[temp_m.day==day]
                temp_md = temp_md.sort_values(by="time_idx")
                temp_md.index=temp_md.time_idx
                
                w = np.array(temp_md[features_static].drop_duplicates())[0]
                for k, g in groupby(enumerate(list(temp_md.index)), lambda ix : ix[0] - ix[1]):
                    temp_list =list(map(itemgetter(1), g)) 

                    if len(temp_list)>lookback_timesteps:
                        temp_df = temp_md.loc[temp_list, features_dynamic]
                        W, X, y, z = split_sequences(np.array(temp_df), w, i, lookback_timesteps, prediction_horizon)
                        W_list.extend(W)
                        X_list.extend(X)
                        y_list.extend(y)
                        z_list.extend(z)
    return W_list, X_list, y_list, z_list

def gather_time_indices(sequences, n_steps, p_horizon):
    '''
    split a multivariate sequence into samples
    '''
    T = list()
    for i, idx in enumerate(sequences.time_idx):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix + p_horizon > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_t = idx+n_steps
        T.append(seq_t)
    return array(T)

def get_list_time_indices(df, lookback_timesteps = 3, prediction_horizon=1 ):
    T_list = list()
    for i in tqdm(df.paris_id.unique()):
        temp = df[df.paris_id==i]
        for month in temp.month.unique():
            temp_m = temp[temp.month==month]
            
            # deactivating within day as a fixed feature
            # as it leads to loss of data for first few hours of 
            # each day            
            #######################################
            temp_m = temp_m.sort_values(by="time_idx")
            temp_m.index=temp_m.time_idx
            #######################################
            for k, g in groupby(enumerate(list(temp_m.index)), lambda ix : ix[0] - ix[1]):
                temp_list =list(map(itemgetter(1), g)) 

                if len(temp_list)>lookback_timesteps:
                    temp_df = temp_m.loc[temp_list, :]
                    T = gather_time_indices(temp_df, lookback_timesteps, prediction_horizon)
                    T_list.extend(T)
    return T_list


def get_time_idx(x):
    return pd.to_datetime(str(x[0])+"-"+str(x[1])+"-2019 "+ str(x[2])+":00", dayfirst=True)

def prepare_data(input_file, 
                link_types,
                seed=2 ):
    df_fusion = pd.read_csv(input_file)

    df_fusion.dropna(subset = ['maxspeed', 'lanes'], inplace=True)
    df_fusion = df_fusion.drop_duplicates(subset=['paris_id', 'hour', 'day', 'month','pp_from_x', 'pp_to_x', 'pp_osmstartnodeid', 'pp_osmendnodeid' ], keep='first').reset_index(drop=True)
    df_new = df_fusion.groupby(by=['paris_id', 'hour', 'day', 'month']).mean().reset_index()
    det_road_key = df_fusion[['paris_id', 'highway']].drop_duplicates(subset=['paris_id'], keep='first').reset_index(drop=True)
    det_road_key = det_road_key.sort_values(by='paris_id').reset_index(drop=True)
    df_new = pd.merge(left=df_new, right=det_road_key, left_on="paris_id", right_on = "paris_id")
    df_fusion = df_new

    X_filtered = df_fusion.drop(columns=[ 'pp_from_x', 'pp_to_x',
                                    'pp_osmstartnodeid',
                                    'pp_osmendnodeid',\
                                    ])

    det_type = list(Counter(X_filtered['highway'].astype(str) + "_" + X_filtered['paris_id'].astype(str)).keys())
    det_number = []
    det_types = []
    for i, val in enumerate(det_type):
        det_types.append(str(val.split("_")[0]))
        det_number.append(int(val.split("_")[-1]))
        
    set_ids  = []
    for i, j in enumerate(det_types):
        ### Using only trunk type detectors for the training the model
        ### as this is only one of three prominent detector categories
        if j in link_types:#,'primary', 'secondary', 'tertiary']:
            set_ids.append(det_number[i])

    X_filtered['date_time'] = Parallel(n_jobs=8, verbose=0)(delayed(get_time_idx)(i) 
                                                            for _, i in enumerate(zip(X_filtered.day, 
                                                                                    X_filtered.month,
                                                                                    X_filtered.hour)))

    X_filtered["time_idx"] = X_filtered["date_time"].dt.year * 12 + \
                            X_filtered["date_time"].dt.day_of_year *24 + \
                            X_filtered["date_time"].dt.hour
    X_filtered["time_idx"] -= X_filtered["time_idx"].min()
    X_filtered["time_idx"] = X_filtered["time_idx"].astype(int)
    X_filtered["month"] = X_filtered.date_time.dt.month.astype(str).astype("category")
    X_filtered["hour"] = X_filtered.date_time.dt.hour.astype(str).astype("category")
    X_filtered["day"] = X_filtered.date_time.dt.dayofweek.astype(str).astype("category")

    for det_id, months in [(5437, ['4', '5']),
                        (5370, ['8']),
                        (5362, ['1']),
                        (5367, ['3', '4']),
                        (5414, ['11', '12']),
                        (5343, ['8']),
                        (5322, ['8']),
                        (5440, ['4', '7', '8']),
                        ]:
        X_filtered = remove_problematic_sequnces(X_filtered, det_id, months)


    X_filtered.reset_index(drop=True, inplace=True)

    highway_importance = pd.get_dummies(X_filtered['highway'])
    
    highway_importance = highway_importance[link_types]

    highway_direction = pd.get_dummies(X_filtered['oneway'])

    X_formatted = pd.DataFrame(np.hstack([highway_importance.values, highway_direction.values]))
                                
    X_formatted = pd.concat([X_formatted, X_filtered[["time_idx", "month", "day", "hour", 
                                            'maxspeed', 'length', 'lanes', 'speed_kph_mean', 
                                            'speed_kph_stddev', 'paris_id', 'k', 'q']].reset_index(drop=True)], axis=1)

    feature_list = list(highway_importance.columns)
    feature_list.extend(['oneway'])
    feature_list.extend(["time_idx", "month", "day", "hour",
                        'maxspeed', 'length', 'lanes', 'speed_kph_mean', 
                        'speed_kph_stddev', 'paris_id', 'k', 'q'])

    X_formatted.columns = feature_list

    for col in link_types:
        X_formatted[col] = X_formatted[col].astype(str).astype("category")
    X_formatted["oneway"] = X_formatted["oneway"].astype(str).astype("category")

    ### rounding off the lanes to the nearest integer
    X_formatted.lanes = X_formatted.lanes.astype(int)
    X_formatted.maxspeed = np.round(X_formatted.maxspeed/10)*10
    X_formatted = X_formatted[X_formatted.lanes<6]
    X_formatted['month'] = X_formatted.month.astype(int) -1
    X_formatted["month"] = X_formatted["month"].astype(str).astype("category")

    # X_formatted = X_formatted[X_formatted.primary_link!=1]
    assert link_types[0]=="trunk"
    X_formatted = X_formatted[X_formatted[link_types[0]]=="1"]

    X_formatted.drop(columns="oneway", inplace=True) # only one unique value which does not makes sense here

    X_formatted = X_formatted.sort_values(by=["paris_id", "time_idx"])
    X_formatted = X_formatted[~X_formatted.paris_id.isin([5320, 5416,5378])]

    X_formatted.reset_index(drop=True, inplace=True)

    return X_formatted, set_ids

def split_data(df, det_ids, test_ids, val_ratio=0.15, seed=2):
    random.seed(seed)
    non_test_ids = []
    for i in det_ids:
        if i not in test_ids:
            non_test_ids.append(i)

    val_ids = list(random.sample(non_test_ids, int(val_ratio*len(det_ids))))
    new_train_ids = []
    for i in non_test_ids:
        if i not in val_ids:
            new_train_ids.append(i)
    train_ids= new_train_ids

    X_formatted_train = df[df.paris_id.isin(train_ids)]
    X_formatted_test = df[df.paris_id.isin(test_ids)]
    X_formatted_val = df[df.paris_id.isin(val_ids)]
    return X_formatted_train, X_formatted_test, X_formatted_val

def scaling_data(X_formatted_train, 
                X_formatted_test, 
                X_formatted_val,
                continous_feature_columns,
                categorical_features_columns,
                other_columns):

    scaler = StandardScaler().fit(X_formatted_train[continous_feature_columns].values)
    scaled_features_train = scaler.transform(X_formatted_train[continous_feature_columns].values)
    scaled_features_val = scaler.transform(X_formatted_val[continous_feature_columns].values)
    scaled_features_test = scaler.transform(X_formatted_test[continous_feature_columns].values)

    scaled_features_df_train = pd.DataFrame(scaled_features_train, index=X_formatted_train.index,
                                    columns=continous_feature_columns)

    scaled_features_df_val = pd.DataFrame(scaled_features_val, index=X_formatted_val.index,
                                    columns=continous_feature_columns)


    scaled_features_df_test = pd.DataFrame(scaled_features_test, index=X_formatted_test.index,
                                    columns=continous_feature_columns)

    scaled_features_df_train[categorical_features_columns] = X_formatted_train[categorical_features_columns]
    scaled_features_df_test[categorical_features_columns] = X_formatted_test[categorical_features_columns]
    scaled_features_df_val[categorical_features_columns] = X_formatted_val[categorical_features_columns]

    scaled_features_df_train[other_columns] = X_formatted_train[other_columns]
    scaled_features_df_val[other_columns] = X_formatted_val[other_columns]
    scaled_features_df_test[other_columns] = X_formatted_test[other_columns]

    if "q" not in continous_feature_columns:
        scaled_features_df_train['q'] = X_formatted_train['q']
        scaled_features_df_val['q'] = X_formatted_val['q']
        scaled_features_df_test['q'] = X_formatted_test['q']
    print("Scaler output columns")
    print(scaled_features_df_train.columns)

    return scaler, scaled_features_df_train, scaled_features_df_val, scaled_features_df_test


def pre_train_conversions(w_list, x_list, y_list, lookback_timesteps, dyn_to_static):
    X = np.array(x_list, dtype="float64")#[idx, :]
    y = np.array(y_list, dtype="float64")#[idx]
    W = np.array(w_list, dtype="float64")#[idx]
    X = torch.from_numpy(X)#.to(device)
    y = torch.from_numpy(y)#.to(device)
    W = torch.from_numpy(W)#.to(device)
    W = torch.cat([W, X[:, int(np.floor((lookback_timesteps)/2)), 0:len(dyn_to_static)].reshape(X.shape[0], -1)], axis=1)
    X = X[:,:,len(dyn_to_static):]
    W[:,-1] = W[:,-1].type(torch.int)
    W_cat_stat = W[:, 3:]
    W_cont_stat = W[:,:3]
    X_cat_dyn = X[:, :, :0]
    X_cont_dyn = X[:, :, 0:]
    return W_cat_stat, W_cont_stat, X_cat_dyn, X_cont_dyn, y


if __name__ == "__main__":

    data = "../data/traffic_data_2019.csv"
    
    features_dynamic = ["day", "hour", "speed_kph_mean", "speed_kph_stddev", "q"]
    # month and day are considered as static for the forecasting horizon, short-term forecasting
    features_static = ["lanes", "length", "maxspeed", "month"]

    continous_feature_columns = ["maxspeed", "lanes", "speed_kph_mean", "speed_kph_stddev", "length", "k"]
    categorical_features_columns = ["month", "day", "hour"]#, "oneway"]
    other_columns = ['time_idx', 'paris_id']

    dyn_to_static = ['day', 'hour']

    test_ids = [5169, 5183, 5266, 5273, 5277, 5299, 5301, 5318, 5325, 
                5330, 5354, 5360, 5361, 5378, 5380, 5382, 5384, 5425, 
                5432, 5446]

    lookback_timesteps = 3
    prediction_horizon = 1
    link_types = ["trunk"]

    X_formatted, det_ids = prepare_data(data, link_types)
    X_formatted_train, X_formatted_test, X_formatted_val = split_data(X_formatted, det_ids, test_ids)

    _, df_train, df_val, df_test = scaling_data(continous_feature_columns,
                                                categorical_features_columns,
                                                other_columns)


    W_list_train, X_list_train, y_list_train, z_list_train = format_data(df_train, lookback_timesteps, prediction_horizon, features_static, features_dynamic)
    W_list_val, X_list_val, y_list_val, z_list_val = format_data(df_val, lookback_timesteps, prediction_horizon, features_static, features_dynamic)
    W_list_test, X_list_test, y_list_test, z_list_test = format_data(df_test, lookback_timesteps, prediction_horizon, features_static, features_dynamic)

    W_train_cat_stat, W_train_cont_stat, X_train_cat_dyn, X_train_cont_dyn, y_train = pre_train_conversions(W_list_train, 
                                                                                                            X_list_train, 
                                                                                                            y_list_train,
                                                                                                            lookback_timesteps, 
                                                                                                            dyn_to_static)
    W_val_cat_stat, W_val_cont_stat, X_val_cat_dyn, X_val_cont_dyn, y_val = pre_train_conversions(W_list_val, 
                                                                                                    X_list_val, 
                                                                                                    y_list_val,
                                                                                                    lookback_timesteps, 
                                                                                                    dyn_to_static)
    W_test_cat_stat, W_test_cont_stat, X_test_cat_dyn, X_test_cont_dyn, y_test = pre_train_conversions(W_list_test, 
                                                                                                        X_list_test, 
                                                                                                        y_list_test,
                                                                                                        lookback_timesteps, 
                                                                                                        dyn_to_static)


    cat_dims_dyn = [int(X_formatted_train[col].nunique()) for col in []]
    cat_dims_stat = [int(X_formatted_train[col].nunique()) for col in ["month", "day", "hour"]]
    emb_dims_stat = [(x, min(10, (x + 1) // 2)) for x in cat_dims_stat]
    emb_dims_dyn = [(x, min(10, (x + 1) // 2)) for x in cat_dims_dyn]

    print("Done")