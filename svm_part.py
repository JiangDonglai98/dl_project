"""
author: Donglai Jiang
"""
# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import glob
import time
import random

random_num = 1
np.random.seed(random_num)
random.seed(random_num)

random.seed(100)
np.random.seed(100)

# %%
root = r'G:\Master\master_course\Deep Learning\project\dl_project_code'


def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
    file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
    return file_list


name_list = get_file_names(root, 'datasets', 'mini', 'npy')

## 二分类
# %%
# 分钟级别
test_X = np.load(name_list[0])
test_y = np.load(name_list[1])
train_X = np.load(name_list[2])
train_y = np.load(name_list[3])
train_y = train_y > 0
test_y = test_y > 0
train_y = train_y.flatten()
test_y = test_y.flatten()
total_dict = {'C': [], 'kernel': [], 'gamma': [], 'degree': [], 'accuracy': []}


# %%
def timer(function):
    """
     装饰器函数timer
     :param function:想要计时的函数
     :return:
     """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("【%s】运行时间：【%s】秒" % (function.__name__, cost_time))
        return res

    return wrapper


# # %%
# svm_clf = SVC(C=1, kernel='rbf', verbose=True)
# svm_clf.fit(train_X, train_y > 0)
#

# %%
@timer
def train_test_call(model, train_X, train_y, test_X, test_y, parms, name):
    print('-' * 30, parms, '-' * 30)
    clf = model(**parms)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    accu = sum(pred == test_y) / len(test_y)
    print(f'accuracy of {name}: ', accu)
    if parms.get('C', -1) != -1:
        total_dict['C'].append(parms['C'])
    else:
        total_dict['C'].append(np.nan)
    if parms.get('kernel', -1) != -1:
        total_dict['kernel'].append(parms['kernel'])
    else:
        total_dict['kernel'].append(np.nan)
    if parms.get('gamma', -1) != -1:
        total_dict['gamma'].append(parms['gamma'])
    else:
        total_dict['gamma'].append(np.nan)
    if parms.get('degree', -1) != -1:
        total_dict['degree'].append(parms['degree'])
    else:
        total_dict['degree'].append(np.nan)
    total_dict['accuracy'].append(accu)
    total_dict['name'].append(name)


# %%
for C in [1, 2, 5, 7, 10]:
    for gamma in ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100]:
        svm_dict = {'C': C, "kernel": 'rbf', "verbose": True, 'gamma': gamma}
        train_test_call(SVC, train_X, train_y, test_X, test_y, svm_dict, 'svm')

# %%
for C in [1, 2, 5, 7, 10]:
    for degree in [1, 2, 3, 5]:
        svm_dict = {'C': C, "kernel": 'poly', "verbose": True, 'degree': degree}
        train_test_call(SVC, train_X, train_y, test_X, test_y, svm_dict, 'svm')

# %%
df = pd.DataFrame(total_dict)
df.to_csv(os.path.join(root, 'minute_svm_search_2class.csv'))

# %%
daily_name_list = get_file_names(root, 'datasets', 'Bert(', 'npy')

# %%
# 日度级别
daily_test11_13_X = np.load(daily_name_list[0])
daily_test11_13_y = np.load(daily_name_list[1])
daily_train11_13_X = np.load(daily_name_list[2])
daily_train11_13_y = np.load(daily_name_list[3])
daily_train11_13_y = daily_train11_13_y > 0
daily_test11_13_y = daily_test11_13_y > 0
daily_train11_13_y = daily_train11_13_y.flatten()
daily_test11_13_y = daily_test11_13_y.flatten()

daily_test15_17_X = np.load(daily_name_list[4])
daily_test15_17_y = np.load(daily_name_list[5])
daily_train15_17_X = np.load(daily_name_list[6])
daily_train15_17_y = np.load(daily_name_list[7])
daily_train15_17_y = daily_train15_17_y > 0
daily_test15_17_y = daily_test15_17_y > 0
daily_train15_17_y = daily_train15_17_y.flatten()
daily_test15_17_y = daily_test15_17_y.flatten()

daily_test07_09_X = np.load(daily_name_list[8])
daily_test07_09_y = np.load(daily_name_list[9])
daily_train07_09_X = np.load(daily_name_list[10])
daily_train07_09_y = np.load(daily_name_list[11])
daily_train07_09_y = daily_train07_09_y > 0
daily_test07_09_y = daily_test07_09_y > 0
daily_train07_09_y = daily_train07_09_y.flatten()
daily_test07_09_y = daily_test07_09_y.flatten()


# %%
def rbf(train_X, train_y, test_X, test_y, name):
    for C in [1, 2, 5, 7, 10]:
        for gamma in ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100]:
            svm_dict = {'C': C, "kernel": 'rbf', "verbose": True, 'gamma': gamma}
            train_test_call(SVC, train_X, train_y, test_X, test_y, svm_dict, name)


# %%
def poly(train_X, train_y, test_X, test_y, name):
    for C in [1, 2, 5, 7, 10]:
        for degree in [1, 2, 3, 5]:
            svm_dict = {'C': C, "kernel": 'poly', "verbose": True, 'degree': degree}
            train_test_call(SVC, train_X, train_y, test_X, test_y, svm_dict, name)


# %%
total_dict = {'C': [], 'kernel': [], 'gamma': [], 'degree': [], 'accuracy': [], 'name': []}
daily_test_X_list = [daily_test11_13_X, daily_test15_17_X, daily_test07_09_X]
daily_train_X_list = [daily_train11_13_X, daily_train15_17_X, daily_train07_09_X]
daily_test_y_list = [daily_test11_13_y, daily_test15_17_y, daily_test07_09_y]
daily_train_y_list = [daily_train11_13_y, daily_train15_17_y, daily_train07_09_y]
daily_names = ['Bert(11,13)', "Bert(15,17)", "Bert(07,09)"]
for train_X, train_y, test_X, test_y, name in zip(daily_train_X_list, daily_train_y_list,
                                                  daily_test_X_list, daily_test_y_list,
                                                  daily_names):
    rbf(train_X, train_y, test_X, test_y, name)
    poly(train_X, train_y, test_X, test_y, name)

daily_df = pd.DataFrame(total_dict)
daily_df.to_csv(os.path.join(root, 'daily_svm_search_2class.csv'))

# %%
# the best 2 class SVM we select
bi_svm_clf = SVC(C=2, kernel='poly', gamma=5)
bi_svm_clf.fit(daily_train07_09_X, daily_train07_09_y)
pred = bi_svm_clf.predict(daily_test07_09_X)
accu = sum(pred == daily_test07_09_y) / len(daily_test07_09_y)
train_pred = bi_svm_clf.predict(daily_train07_09_X)
sentiment_X = pd.concat([pd.Series(train_pred), pd.Series(pred)])
sentiment_X = sentiment_X.map({True: 1, False: 0})
sentiment_time = pd.read_csv(get_file_names(root, 'datasets', 'sentiment', 'csv')[0])['date']
sentiment_time = pd.to_datetime(sentiment_time)
sentiment_time = sentiment_time[11:-12]
sentiment_time = sentiment_time.dt.date
sentiment_X.reset_index(inplace=True, drop=True)
sentiment_time.reset_index(inplace=True, drop=True)
bi_sentiment_df = pd.DataFrame({'date': sentiment_time, '0': sentiment_X})
bi_sentiment_df.to_csv(os.path.join(os.path.join(root, 'datasets'), 'daily_svm_sentiment_2class.csv'))


## 六分类
# %%
def rank_column(col: np.ndarray, rank_list) -> np.ndarray:
    temp_col = pd.Series(col)
    rank_dict = {}
    for index in range(len(rank_list[:-1])):
        rank_dict[index] = (rank_list[index], rank_list[index + 1])
        print(index, (rank_list[index], rank_list[index + 1]))

    def make_rank(x, r_dict: dict):
        for r, interval in r_dict.items():
            if interval[0] <= x < interval[1]:
                return int(r)

    temp_col = temp_col.apply(make_rank, args=(rank_dict,)).to_numpy()
    return temp_col


# %%
name_list = get_file_names(root, 'datasets', 'mini', 'npy')
test_X = np.load(name_list[0])
test_y = np.load(name_list[1])
train_X = np.load(name_list[2])
train_y = np.load(name_list[3])
train_y = train_y.flatten()
test_y = test_y.flatten()
train_y = rank_column(train_y, [-10, -1, -0.5, 0, 0.5, 1, 10])
test_y = rank_column(test_y, [-10, -1, -0.5, 0, 0.5, 1, 10])

# %%
mult_svm_clf = SVC(decision_function_shape='ovo')
mult_svm_clf.fit(train_X, train_y)
pred = mult_svm_clf.predict(test_X)
accu = sum(pred == test_y) / len(test_y)
print(f'accuracy of {"multiclass svm"}: ', accu)
# accuracy of multiclass svm:  0.4540540540540541
train_pred = mult_svm_clf.predict(train_X)
sentiment_X = pd.concat([pd.Series(train_pred), pd.Series(pred)])


# %%
def gen_sentiment_score(y: pd.Series, rank_list) -> pd.Series:
    rank_dict = {}
    for index in range(len(rank_list[:-1])):
        rank_dict[index] = (rank_list[index], rank_list[index + 1])
        print(index, (rank_list[index], rank_list[index + 1]))

    def get_score(x, r_dict: dict):
        for r, interval in r_dict.items():
            if r == x:
                return float((interval[0] + interval[1]) / 2)

    y = y.apply(get_score, args=(rank_dict,))
    return y


# %%
sentiment_X = gen_sentiment_score(sentiment_X, [-10, -1, -0.5, 0, 0.5, 1, 10])
sentiment_time = pd.read_csv(get_file_names(root, 'datasets', 'ShangZheng_index_30min', 'csv')[0])['trade_time']
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2019-12-31')
split_date = pd.to_datetime('2019-06-01')
sentiment_time = pd.to_datetime(sentiment_time)
sentiment_time = sentiment_time[(sentiment_time < end_date) & (sentiment_time > start_date)]
train_time = sentiment_time[sentiment_time < split_date]
test_time = sentiment_time[sentiment_time > split_date]
train_time = train_time[1:]
test_time = test_time[1:-1]
sentiment_time = pd.concat([train_time, test_time])
sentiment_X.reset_index(inplace=True, drop=True)
sentiment_time.reset_index(inplace=True, drop=True)
multi_sentiment_df = pd.DataFrame({'date': sentiment_time, '0': sentiment_X})
multi_sentiment_df['0'] = multi_sentiment_df['0'] + 1
multi_sentiment_df['date'] = multi_sentiment_df['date'].dt.date
multi_sentiment_df = multi_sentiment_df.groupby('date', as_index=False).prod()
multi_sentiment_df.to_csv(os.path.join(os.path.join(root, 'datasets'), 'minute_svm_sentiment_6class.csv'))


# %% daily 级别
daily_test07_09_X = np.load(daily_name_list[8])
daily_test07_09_y = np.load(daily_name_list[9])
daily_train07_09_X = np.load(daily_name_list[10])
daily_train07_09_y = np.load(daily_name_list[11])
daily_train07_09_y = daily_train07_09_y.flatten()
daily_test07_09_y = daily_test07_09_y.flatten()
daily_train07_09_y = rank_column(daily_train07_09_y, [-10, -1, -0.5, 0, 0.5, 1, 10])
daily_test07_09_y = rank_column(daily_test07_09_y, [-10, -1, -0.5, 0, 0.5, 1, 10])

daily_test11_13_X = np.load(daily_name_list[0])
daily_test11_13_y = np.load(daily_name_list[1])
daily_train11_13_X = np.load(daily_name_list[2])
daily_train11_13_y = np.load(daily_name_list[3])
daily_train11_13_y = daily_train11_13_y.flatten()
daily_test11_13_y = daily_test11_13_y.flatten()
daily_train11_13_y = rank_column(daily_train11_13_y, [-10, -1, -0.5, 0, 0.5, 1, 10])
daily_test11_13_y = rank_column(daily_test11_13_y, [-10, -1, -0.5, 0, 0.5, 1, 10])

daily_test15_17_X = np.load(daily_name_list[4])
daily_test15_17_y = np.load(daily_name_list[5])
daily_train15_17_X = np.load(daily_name_list[6])
daily_train15_17_y = np.load(daily_name_list[7])
daily_train15_17_y = daily_train15_17_y.flatten()
daily_test15_17_y = daily_test15_17_y.flatten()
daily_train15_17_y = rank_column(daily_train15_17_y, [-10, -1, -0.5, 0, 0.5, 1, 10])
daily_test15_17_y = rank_column(daily_test15_17_y, [-10, -1, -0.5, 0, 0.5, 1, 10])

# %%
mult_svm_clf = SVC(decision_function_shape='ovo')
mult_svm_clf.fit(daily_train07_09_X, daily_train07_09_y)
pred = mult_svm_clf.predict(daily_test07_09_X)
accu = sum(pred == daily_test07_09_y) / len(daily_test07_09_y)
print(f'accuracy of {"multiclass daily svm"}: ', accu)
# accuracy of multiclass daily 07-09 svm:  0.35
# %%
mult_svm_clf = SVC(decision_function_shape='ovo')
mult_svm_clf.fit(daily_train11_13_X, daily_train11_13_y)
pred = mult_svm_clf.predict(daily_test11_13_X)
accu = sum(pred == daily_test11_13_y) / len(daily_test11_13_y)
print(f'accuracy of {"multiclass daily svm"}: ', accu)
# accuracy of multiclass daily 11-13 svm:  0.34428571428571427

# %%
mult_svm_clf = SVC(decision_function_shape='ovo')
mult_svm_clf.fit(daily_train15_17_X, daily_train15_17_y)
pred = mult_svm_clf.predict(daily_test15_17_X)
accu = sum(pred == daily_test15_17_y) / len(daily_test15_17_y)
print(f'accuracy of {"multiclass daily svm"}: ', accu)
# accuracy of multiclass daily 15-17 svm:  0.32445922457976

# %%
train_pred = mult_svm_clf.predict(daily_train07_09_X)
sentiment_X = pd.concat([pd.Series(train_pred), pd.Series(pred)])
sentiment_X = gen_sentiment_score(sentiment_X, [-10, -1, -0.5, 0, 0.5, 1, 10])
# %%
sentiment_time = pd.read_csv(get_file_names(root, 'datasets', 'sentiment', 'csv')[0])['date']
sentiment_time = pd.to_datetime(sentiment_time)
sentiment_time = sentiment_time[11:-12]
sentiment_time = sentiment_time.dt.date
sentiment_X.reset_index(inplace=True, drop=True)
sentiment_time.reset_index(inplace=True, drop=True)
multi_sentiment_df = pd.DataFrame({'date': sentiment_time, '0': sentiment_X})
multi_sentiment_df.to_csv(os.path.join(os.path.join(root, 'datasets'), 'daily_svm_sentiment_6class.csv'))