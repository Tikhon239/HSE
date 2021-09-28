import numpy as np
import pandas as pd
import sqlite3

from datetime import datetime

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.clustering import TimeSeriesKMeans


def get_data_1():
	con = sqlite3.connect('trade_info.sqlite3')
	data =  pd.read_sql(
	    """SELECT * FROM Chart_data C
	    JOIN Trading_session T ON C.session_id=T.id
	    WHERE T.trading_type = 'monthly'""",
	    con)
	con.close()

	data['all_date'] = pd.to_datetime(data.date + " " + data.time)
	data.sort_values(by='all_date', inplace=True)
	data.drop_duplicates(subset='deal_id', inplace=True)

	data.time = pd.to_datetime(data.time,  format="%H:%M:%S")
	#торги идут только час
	data.time = data.time.apply(lambda x: datetime(x.year, x.month, x.day, 0, x.minute, x.second))

	# может быть есть годовая или месячная или недельная сезонность
	data['year'] = data.all_date.apply(lambda x: x.date().year)
	data['month'] = data.all_date.apply(lambda x: x.date().month)
	data['week'] = data.all_date.apply(lambda x: x.weekday())
	data['hour'] = data.all_date.apply(lambda x: x.hour)

	data.drop(['id', 'deal_id', 'trading_type', 'date'], axis=1, inplace=True)
	data.reset_index(drop=True, inplace=True)
	return data


def get_data_2():
	con = sqlite3.connect('trade_info.sqlite3')
	data =  pd.read_sql(
	    """SELECT * FROM Chart_data C
	    JOIN Trading_session T ON C.session_id=T.id
	    WHERE T.trading_type = 'monthly'""",
	    con)
	con.close()

	data['all_date'] = pd.to_datetime(data.date + " " + data.time)
	data.sort_values(by='all_date', inplace=True)
	data.drop_duplicates(subset='deal_id', inplace=True)

	#торги идут только час
	data.all_date = data.all_date.apply(lambda x: datetime(x.year, x.month, x.day, 0, x.minute, x.second))

	data.drop(['id', 'deal_id', 'trading_type', 'lot_size', 'time', 'date'], axis=1, inplace=True)
	data.reset_index(drop=True, inplace=True)
	return data


def data_equalization(data):
	unique_session_id = data.session_id.unique()
	unique_session_id = np.hstack([unique_session_id[0], unique_session_id, unique_session_id[-1]])
	mean_norm_price = data.groupby('session_id')['norm_price'].mean()

	for prev_session_id, session_id, next_session_id in zip(unique_session_id, unique_session_id[1:], unique_session_id[2:]):
	    cur_session = data[data.session_id == session_id].iloc[0]
	    platform_id = cur_session.platform_id
	    date = cur_session.all_date
	    data = pd.concat([
	        data,
	        pd.DataFrame({
	            'session_id': session_id,
	            'platform_id': platform_id,
	            'all_date': [
	                datetime(date.year, date.month, date.day, 0, 0, 0),
	                datetime(date.year, date.month, date.day, 0, 59, 0)
	            ],
	            # берем среднее до нас и среднее после нас
	            'norm_price': [
	                mean_norm_price[mean_norm_price.index == prev_session_id].iloc[0],
	                mean_norm_price[mean_norm_price.index == next_session_id].iloc[0]
	            ]
	        })
	    ])

	data = data.groupby('session_id').resample("1T", on="all_date").mean().interpolate("linear")
	data.drop('session_id', axis=1, inplace=True)
	data.reset_index(inplace=True)
	return data


def get_time_series_dataset(data):
	time_series_dataset = list(map(lambda x: x[1].values, data.groupby('session_id')['norm_price']))
	time_series_dataset = to_time_series_dataset(time_series_dataset)
	time_series_dataset = TimeSeriesResampler(sz=time_series_dataset.shape[1]).fit_transform(time_series_dataset)
	return time_series_dataset


def t_s_k_means(time_series_dataset, n_clusters=5, n_init=10):
	model_1 = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean', n_init=n_init)
	model_2 = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', n_init=n_init)
	models = [model_1, model_2]

	y_pred_1 = model_1.fit_predict(time_series_dataset)
	y_pred_2 = model_2.fit_predict(time_series_dataset)
	y_preds = [y_pred_1, y_pred_2]

	plt.figure(figsize = (10, 10))
	for i in range(n_clusters):
	    for j in range(2):
	        plt.subplot(n_clusters, 2, 2 * i + j + 1)
	        for time_series in time_series_dataset[y_preds[j] == i]:
	            plt.plot(time_series.ravel(), "k-", alpha=0.2)
	        plt.plot(models[j].cluster_centers_[i].ravel(), "r-")
	plt.show()
