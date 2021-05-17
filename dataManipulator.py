import pandas as pd
import numpy as np
import os
import glob
import tushare as ts
import datetime as dt
import time
from functools import reduce
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer


# %%
class innerDataset(Dataset):
    """
    author: Donglai jiang
    The class is generating the tensor for X and y of DataLoader
    """

    def __init__(self, tensor_data, tensor_target, window_len):
        self.data = tensor_data
        self.window_len = window_len
        self.target = tensor_target

    def __getitem__(self, index):
        return self.data[index:index + self.window_len], self.target[index + self.window_len]

    def __len__(self):
        return len(self.target) - self.window_len


class DataManipulator:
    """
    author: Donglai jiang
    The main targets of this class are:
    1.	Retrieve and generate all paths of data we need
    2.	Read in data from the generated path, select and name the used data
    3.	Connect data from the database needed (MongoDB)
    4.	Define a time range for all the data
    5.	Find and deal with Nan values and do normalization  EDA
    6.	Aggregate all factor data together and find the common trading day
    7.	Generates a dynamic data set based on the length of the moving window set by user
    """

    # ts_pro = ts.pro_api()

    def __init__(self, root: str, dataset_dir: str, tokenizer_dir: str, ts_token: str):
        self.root = root
        self.dataset_dir = dataset_dir
        self.tokenizer_dir = tokenizer_dir
        self.feature_dict = {}
        self.main_df = None  # pd.DataFrame()
        self.news_time_df = None
        self.news_df = None  # pd.DataFrame()
        self.LSTM_train_dict = {}
        self.LSTM_test_dict = {}
        self.Bert_train_dict = {}
        self.Bert_test_dict = {}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=os.path.join(self.root,
                                                                                                   self.tokenizer_dir))
        self.used_measure_list = []
        self.start_date = '20100101'
        self.end_data = '20201231'
        self.split_date = '2020-06-01'
        self.minute_split_date = '20100101'
        self.minute_start_date = '20201231'
        self.minute_end_data = '2020-06-01'
        self.sentiment_list_dict = {}
        self.scaled_list = []
        self.minute_scaled_list = []
        self.used_columns = None
        self.minute_used_columns = None
        self.label_num = 0
        self.rnn_x_len = 0
        # ts.set_token(ts_token)

    @classmethod
    def get_trading_dates(cls, start: str, end: str, path: str = None) -> pd.DataFrame:
        if path:
            df = pd.read_csv(path, index_col=0)
            df['main_trade_date'] = pd.to_datetime(df['main_trade_date'])
        else:
            try_time = 0
            while True:
                try:
                    df = cls.ts_pro.trade_cal(exchange='', start_date=start, end_date=end)
                    print("Tushare connect success: " + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    break
                except Exception as e:
                    print(
                        "Tushare connect failed: " + dt.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S") + ", caused by " + str(
                            e))
                    time.sleep(3)
                    if try_time >= 5:
                        raise e
                try_time += 1
            df.drop(df[df['is_open'] == 0].index, inplace=True)
            df.rename(columns={'cal_date': 'main_trade_date'}, inplace=True)
            df.sort_values('main_trade_date', inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['main_trade_date'] = pd.to_datetime(df['main_trade_date'], format='%Y%m%d')
            df = df.loc[:, ['main_trade_date']]
        return df

    @staticmethod
    def add_time_columns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Get the all time data from the given date.
        It is assumed the week starts on Monday,
        which is denoted by 0 and ends on Sunday which is denoted by 6.
        :param df:          target dataframe
        :param column_name: date column name
        :return:            dataframe
        """
        df['year'] = df[column_name].dt.year
        df['month'] = df[column_name].dt.month
        df['day'] = df[column_name].dt.day
        df['week'] = df[column_name].dt.dayofweek
        # df['hour'] = df[column_name].dt.hour
        # df['minute'] = df[column_name].dt.minute
        # df['second'] = df[column_name].dt.second
        return df

    @staticmethod
    def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
        file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
        return file_list

    @staticmethod
    def df_col_name_tag(df: pd.DataFrame, tag: str, not_change_list: list = None, change_list: list = None):
        if not_change_list is not None:
            df.rename(columns=lambda x: tag + '_' + x if x not in not_change_list else x, inplace=True)
        if change_list is not None:
            df.rename(columns=lambda x: tag + '_' + x if x in change_list else x, inplace=True)
        return df

    @staticmethod
    def columns_shift(df: pd.DataFrame, shift_clo_list: list, shift_len_list: tuple = (-1,),
                      add: bool = False) -> pd.DataFrame:
        if not add:
            for name in shift_clo_list:
                df[name] = df[name].shift(shift_len_list[0])
            df = DataManipulator.df_col_name_tag(df, tag='shift' + str(shift_len_list[0]), change_list=shift_clo_list)
        else:
            for name in shift_clo_list:
                for shift_len in shift_len_list:
                    df['shift' + str(shift_len) + '_' + name] = df[name].shift(shift_len)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def complement_df(df: pd.DataFrame, date_col_name: str, method: str = 'ffill') -> pd.DataFrame:
        """
        complete the dataframe by checking the date column, default method is 'ffill'
        :param df:
        :param date_col_name:  should be clear that this columns should in the format of datetime
        :param method:
        :return:
        """
        df[date_col_name] = df[date_col_name].dt.date
        df.set_index(df[date_col_name], inplace=True)
        idx = pd.date_range(df[date_col_name].min(), df[date_col_name].max())
        df = df.reindex(idx)
        df.drop(date_col_name, axis=1, inplace=True)
        df.index.name = date_col_name
        df.fillna(inplace=True, method=method)
        df.reset_index(inplace=True)
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        return df

    @staticmethod
    def clear_datetime_to_date(df: pd.DataFrame, date_col_name: str):
        df[date_col_name] = df[date_col_name].dt.date
        df[date_col_name] = pd.to_datetime(df[date_col_name])

    @staticmethod
    def cut_time_string(x, start_pos: int, end_pos: int):
        return x[start_pos:end_pos]

    @staticmethod
    def select_col_group_by(df: pd.DataFrame, group_by_col_name: str, class_list: list, date_col_name: str):
        """
        select some classes in the column to transform into different columns
        :param df:
        :param group_by_col_name:
        :param class_list:
        :return:
        """
        DataManipulator.clear_datetime_to_date(df, date_col_name)
        grouped = df.groupby(group_by_col_name)
        temp_df_list = []
        for name, group in grouped:
            if name in class_list:
                temp_df = group.copy(deep=True)
                temp_df.drop(group_by_col_name, inplace=True, axis=1)
                temp_df.sort_values(by=date_col_name, inplace=True)
                temp_df.rename(columns=lambda x: name + '_' + x if x != date_col_name else x, inplace=True)
                temp_df_list.append(temp_df)
        df = reduce(lambda left, right: pd.merge(left, right, on=date_col_name), temp_df_list)
        return df

    @staticmethod
    def rank_column(col: pd.Series, rank_list):
        rank_dict = {}
        for index in range(len(rank_list[:-1])):
            rank_dict[index] = (rank_list[index], rank_list[index + 1])
            print(index, (rank_list[index], rank_list[index + 1]))

        def make_rank(x, r_dict: dict):
            for r, interval in r_dict.items():
                if interval[0] <= x < interval[1]:
                    return int(r)

        col = col.apply(make_rank, args=(rank_dict,))
        return col

    @staticmethod
    def bert_token_process(df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        input_ids = []
        attention_masks = []
        for sent in df.values:
            encoded_dict = tokenizer.encode_plus(
                sent[0][:max_len - 2],  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                padding='max_length',
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # print('input_ids', encoded_dict['input_ids'].shape)
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            # print('attention_mask', encoded_dict['attention_mask'].shape)
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return TensorDataset(input_ids, attention_masks)

    @staticmethod
    def news_df_selection(df: pd.DataFrame, time_interval: tuple, date_col_name: str,
                          news_col_name: str) -> pd.DataFrame:
        df = df.loc[(df[date_col_name].dt.hour >= time_interval[0]) & (df[date_col_name].dt.hour <= time_interval[1]),
             :].copy(deep=True)
        DataManipulator.clear_datetime_to_date(df, date_col_name)
        df = df.groupby(date_col_name)[news_col_name].apply(lambda x: ''.join(x)).reset_index()
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        return df

    @staticmethod
    def minute_news_selection(news_df: pd.DataFrame, time_interval: tuple, date_col_name: str,
                              news_col_name: str, minute_threshold: int = 30) -> pd.DataFrame:
        news_df[date_col_name] = news_df[date_col_name].dt.floor('H') + \
                                 pd.to_timedelta(
                                     news_df[date_col_name].dt.minute.apply(
                                         lambda x: 0 if x <= minute_threshold else minute_threshold), unit='m')
        df = news_df.groupby(date_col_name)[news_col_name].apply(lambda x: ''.join(x)).reset_index()
        df = df.loc[(df[date_col_name].dt.hour >= time_interval[0]) & (df[date_col_name].dt.hour <= time_interval[1]),
             :].copy(deep=True)
        return df

    @staticmethod
    def df_date_cut(df: pd.DataFrame, start: str, end: str, date_col_name: str) -> pd.DataFrame:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        return df.loc[((df[date_col_name] >= start_time) & (df[date_col_name] <= end_time)), :].copy(deep=True)

    @staticmethod
    def resample_time(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
        rng = pd.date_range(df[date_col_name].min(), df[date_col_name].max() + pd.Timedelta(23, 'H'),
                            freq='30Min')
        df = pd.DataFrame({'main_trade_date': rng})
        return df

    def df_to_tensor(self, df: pd.DataFrame, label_list: list, split_time, is_train: bool, is_X: bool,
                     is_Bert: bool = False, bert_max_len: int = 512, normal: bool = False) -> torch.Tensor:
        if is_train and not is_X:
            result = torch.from_numpy(df.loc[df.index <= split_time, label_list].values.astype(np.float32))
        elif is_train and is_X:
            if not is_Bert:
                if not normal:
                    result = torch.from_numpy(df.loc[df.index <= split_time, :].drop(
                        label_list, axis=1).values.astype(np.float32))
                else:
                    result = normalize(torch.from_numpy(df.loc[df.index <= split_time, :].drop(
                        label_list, axis=1).values.astype(np.float32)), dim=0)
                self.rnn_x_len = result.shape[1]
            else:
                result = df.loc[df.index <= split_time, :].drop(label_list, axis=1).copy(deep=True)
                result = DataManipulator.bert_token_process(result, self.tokenizer, bert_max_len)

        elif not is_train and not is_X:
            result = torch.from_numpy(df.loc[df.index > split_time, label_list].values.astype(np.float32))
        else:
            if not is_Bert:
                if not normal:
                    result = torch.from_numpy(df.loc[df.index > split_time, :].drop(
                        label_list, axis=1).values.astype(np.float32))
                else:
                    result = normalize(torch.from_numpy(df.loc[df.index > split_time, :].drop(
                        label_list, axis=1).values.astype(np.float32)), dim=0)
                self.rnn_x_len = result.shape[1]
            else:
                result = df.loc[df.index > split_time, :].drop(label_list, axis=1).copy(deep=True)
                result = DataManipulator.bert_token_process(result, self.tokenizer, bert_max_len)
        return result

    def update_col_name(self, target_col_name_list: list, tag: str):
        for name_list in self.feature_dict.values():
            for index, name in enumerate(name_list['tag_name_list']):
                if name in target_col_name_list:
                    name_list['tag_name_list'][index] = tag + '_' + name

    def read_in_file(self, path: str, time_column_name: str, col_name_list: list, tag: str, time_fuc=None,
                     args=None, dtypes=None) -> pd.DataFrame:
        if dtypes is None:
            df = pd.read_csv(path, index_col=0)
        else:
            df = pd.read_csv(path, dtype=dtypes, index_col=0)
        self.feature_dict[path] = {'date_name': time_column_name, 'name_list': col_name_list}
        col_name_list.append(time_column_name)
        df = df.loc[:, col_name_list]
        if time_fuc is not None:
            df[time_column_name] = df[time_column_name].apply(time_fuc, args=args)
        df[time_column_name] = pd.to_datetime(df[time_column_name])
        df.rename(columns={time_column_name: 'date'}, inplace=True)
        df = DataManipulator.df_col_name_tag(df, tag, ['date'])
        self.feature_dict[path]['tag_date_name'] = 'date'
        self.feature_dict[path]['tag_name_list'] = list(df.columns)
        return df

    def trade_day_init(self, start: str, end: str, path: str = None):
        self.start_date = start
        self.end_data = end
        self.main_df = DataManipulator.get_trading_dates(start, end, path)
        self.main_df = DataManipulator.add_time_columns(self.main_df, 'main_trade_date')
        self.news_time_df = DataManipulator.get_trading_dates(start, end, path)
        self.news_time_df = DataManipulator.resample_time(self.news_time_df, 'main_trade_date')

    def add_column(self, in_df: pd.DataFrame, merge_column_name_list: list = None, time_column_name: str = 'date'):
        """
        add processed column(in dataframe) into self.main_df
        :param in_df:
        :param merge_column_name_list:
        :param time_column_name:
        :return:
        """
        if merge_column_name_list is not None:
            merge_column_name_list.append(time_column_name)
            in_df = in_df.loc[:, merge_column_name_list]
        self.main_df = self.main_df.merge(in_df, left_on='main_trade_date', right_on=time_column_name)
        self.main_df.drop(time_column_name, axis=1, inplace=True)
        self.main_df.dropna(axis=0, inplace=True)
        print(f'add columns:{in_df.columns} success!')

    def news_df_add_column(self, in_df: pd.DataFrame, merge_column_name_list: list = None,
                           time_column_name: str = 'date'):
        """
        add processed column(in dataframe) into self.news_df
        :param in_df:
        :param merge_column_name_list:
        :param time_column_name:
        :return:
        """
        if merge_column_name_list is not None:
            merge_column_name_list.append(time_column_name)
            in_df = in_df.loc[:, merge_column_name_list]
        self.news_df = self.news_time_df.merge(in_df, left_on='main_trade_date', right_on=time_column_name)
        self.news_df.drop(time_column_name, axis=1, inplace=True)
        self.news_df.dropna(axis=0, inplace=True)
        print(f'add columns:{in_df.columns} to news_df success!')

    def delete_column(self, delete_col_name_list: list):
        self.main_df.drop(delete_col_name_list, inplace=True)
        for name_list in self.feature_dict.values():
            temp_list = []
            for name in name_list['tag_name_list']:
                if name in delete_col_name_list:
                    temp_list.append(name)
            for name in temp_list:
                name_list.remove(name)

    def shift_columns(self, shift_column_name_list: list, shift_len_list: tuple = (-1,), add: bool = False):
        if not add:
            self.main_df = DataManipulator.columns_shift(self.main_df, shift_column_name_list, shift_len_list)
            self.update_col_name(shift_column_name_list, 'shift' + str(shift_len_list[0]))
        else:
            self.main_df = DataManipulator.columns_shift(self.main_df, shift_column_name_list, shift_len_list, add)
            for name_list in self.feature_dict.values():
                temp_list = []
                for name in name_list['tag_name_list']:
                    if name in shift_column_name_list:
                        for shift_len in shift_len_list:
                            temp_list.append('shift' + str(shift_len) + '_' + name)
                name_list['tag_name_list'].extend(temp_list)

    def shift_minute_columns(self, shift_column_name_list: list, shift_len_list: tuple = (-1,), add: bool = False):
        self.news_df = DataManipulator.columns_shift(self.news_df, shift_column_name_list, shift_len_list, add)

    def rank_minute_df_columns(self, target_col_name_list: list, rank_list=[-10, -1, -0.5, 0, 0.5, 1, 10]):
        self.label_num = len(rank_list) - 1
        for name in target_col_name_list:
            self.news_df[name] = DataManipulator.rank_column(self.news_df[name], rank_list)
        self.news_df.rename(columns={name: 'rank_' + name for name in target_col_name_list}, inplace=True)

    def rank_df_column(self, target_col_name_list: list, rank_list=[-10, -1, -0.5, 0, 0.5, 1, 10]):
        self.label_num = len(rank_list) - 1
        for name in target_col_name_list:
            self.main_df[name] = DataManipulator.rank_column(self.main_df[name], rank_list)
        self.main_df.rename(columns={name: 'rank_' + name for name in target_col_name_list}, inplace=True)
        self.update_col_name(target_col_name_list, 'rank')

    def add_change_news(self, file_name: str, hour_tuple: tuple, columns_type: dict, news_df: pd.DataFrame = None,
                        time_col_name: str = 'create_time'):
        if news_df is None:
            news_df = self.read_in_file(test_manipulator.get_file_names(root, 'datasets', file_name, 'csv')[0],
                                        time_col_name, ['text', ], file_name, dtypes=columns_type)
        temp_news = self.news_df_selection(news_df, hour_tuple, 'date', file_name + '_text')
        original_name = None
        for name in self.main_df.columns:
            if 'text' in name:
                original_name = name
        if original_name is None:
            self.add_column(temp_news)
        else:
            self.main_df.drop(original_name, inplace=True, axis=1)
            self.add_column(temp_news)

        print(
            '-' * 5 + f"news column name has change to {file_name + '_text'} with selected hour: {hour_tuple}" + '-' * 5)

    def add_minute_change_news(self, file_name: str, columns_type: dict, news_df: pd.DataFrame = None,
                           time_col_name: str = 'create_time'):
        if news_df is None:
            news_df = self.read_in_file(test_manipulator.get_file_names(root, 'datasets', file_name, 'csv')[0],
                                        time_col_name, ['text', ], file_name, dtypes=columns_type)
        minute_temp_news = self.minute_news_selection(news_df, (8, 17), 'date', file_name + '_text')
        self.news_df = pd.merge(self.news_df, minute_temp_news, left_on='main_trade_date', right_on='date')
        print('-' * 5 + f"minute news df is generated" + '-' * 5)

    def scaling_col(self, scaler=StandardScaler, not_scal_list: list = ['text', 'rank', 'main_trade', 'date']):
        temp_scaler = scaler()
        columns = self.main_df.columns.to_list()
        minute_columns = self.news_df.columns.to_list()
        temp_list = []
        minute_temp_list = []
        for col in columns:
            is_in = False
            for not_scal in not_scal_list:
                if not_scal in col:
                    is_in = True
            if not is_in and col not in self.scaled_list:
                self.scaled_list.append(col)
                temp_list.append(col)
        if len(temp_list) > 0:
            print('-' * 20 + 'scaling finished.' + '-' * 20)
            print('Scaled main_df features:', temp_list)
            self.main_df[temp_list] = temp_scaler.fit_transform(self.main_df[temp_list])
        for col in minute_columns:
            is_in = False
            for not_scal in not_scal_list:
                if not_scal in col:
                    is_in = True
            if not is_in and col not in self.minute_scaled_list:
                self.minute_scaled_list.append(col)
                minute_temp_list.append(col)
        if len(minute_temp_list) > 0:
            print('-' * 20 + 'scaling finished.' + '-' * 20)
            print('Scaled news_df features:', minute_temp_list)
            self.news_df[minute_temp_list] = temp_scaler.fit_transform(self.news_df[minute_temp_list])

    def clear(self):
        drop_list = []
        for name in self.main_df.columns.to_list():
            if 'Unnamed' in name:
                drop_list.append(name)
            if 'year' in name:
                drop_list.append(name)
            if 'day' in name:
                drop_list.append(name)
        if len(drop_list) > 0:
            print('Drop: ', drop_list)
            self.main_df.drop(drop_list, axis=1, inplace=True)

        minute_news_drop_list = []
        for name in self.news_df.columns.to_list():
            if 'Unnamed' in name:
                minute_news_drop_list.append(name)
            if 'year' in name:
                minute_news_drop_list.append(name)
            if 'day' in name:
                minute_news_drop_list.append(name)
        if len(minute_news_drop_list) > 0:
            print('Drop: ', minute_news_drop_list)
            self.news_df.drop(minute_news_drop_list, axis=1, inplace=True)

    def LSTM_train_test_split(self, name: str, window_len: int, start: str = '2014-01-01', end: str = '2020-12-31',
                              split_date: str = '2020-06-01',
                              label_list: list = ['rank_shift-1_ShangZheng_pct_chg', ],
                              news_col_name: str = 'sina_text', normal: bool = True, sentiment_df: pd.DataFrame = None):
        """

        :param name:            Give a name to the dataset
        :param window_len:      set the window length you want for the time series Dataset
        :param start            want the dataset to start at
        :param end              want the dataset to end at
        :param split_date:      The cutting data for the train and test set
        :param label_list:      The y label column's name list
        :param news_col_name:   The news column name in the self.main_df, we do not need it in this train test dataset
        :param normal           generate the X data with normalization, default = True
        :param sentiment_df:    The dataframe that contains the sentiment columns, must have a column named 'date'
                                in the type of pandas datetime 64
        :return:
        """
        # 'rank_shift-2_ShangZheng_pct_chg',
        # 'rank_shift-3_ShangZheng_pct_chg'
        self.split_date = split_date
        self.start_date = start
        self.end_data = end
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
        split_threshold = pd.to_datetime(split_date)
        if sentiment_df is not None:
            self.add_column(sentiment_df)
        temp_df = DataManipulator.df_date_cut(self.main_df, start, end, 'main_trade_date')
        temp_df.index = temp_df['main_trade_date']
        temp_df.index.name = 'main_trade_date'
        temp_df.drop('main_trade_date', inplace=True, axis=1)
        original_columns = temp_df.columns
        if news_col_name in original_columns:
            temp_df.drop(news_col_name, inplace=True, axis=1)
        self.used_columns = temp_df.columns.to_list()
        self.LSTM_train_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, True, True, normal=normal),
            self.df_to_tensor(temp_df, label_list, split_threshold, True, False),
            window_len
        )
        self.LSTM_test_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, False, True, normal=normal),
            self.df_to_tensor(temp_df, label_list, split_threshold, False, False),
            window_len
        )
        if sentiment_df is not None:
            name_list = sentiment_df.columns.to_list()
            name_list.remove('date')
            self.delete_column(name_list)

    def minute_LSTM_train_test_split(self, name: str, window_len: int, start: str = '2014-01-01', end: str = '2020-12-31',
                              split_date: str = '2020-06-01',
                              label_list: list = ['rank_shift-1_ShangZheng_30min_pct_chg', ],
                              news_col_name: str = 'sina_text', normal: bool = True, sentiment_df: pd.DataFrame = None):
        self.minute_split_date = split_date
        self.minute_start_date = start
        self.minute_end_data = end
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
        split_threshold = pd.to_datetime(split_date)
        if sentiment_df is not None:
            self.news_df_add_column(sentiment_df)
        temp_df = DataManipulator.df_date_cut(self.news_df, start, end, 'main_trade_date')
        temp_df.index = temp_df['main_trade_date']
        temp_df.index.name = 'main_trade_date'
        temp_df.drop('main_trade_date', inplace=True, axis=1)
        original_columns = temp_df.columns
        if news_col_name in original_columns:
            temp_df.drop(news_col_name, inplace=True, axis=1)
        self.minute_used_columns = temp_df.columns.to_list()
        self.LSTM_train_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, True, True, normal=normal),
            self.df_to_tensor(temp_df, label_list, split_threshold, True, False),
            window_len
        )
        self.LSTM_test_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, False, True, normal=normal),
            self.df_to_tensor(temp_df, label_list, split_threshold, False, False),
            window_len
        )
        if sentiment_df is not None:
            name_list = sentiment_df.columns.to_list()
            name_list.remove('date')
            self.delete_column(name_list)

    def Bert_train_test_split(self, name: str, window_len: int, start: str = '2014-01-01', end: str = '2020-12-31',
                              split_date: str = '2020-06-01',
                              label_list: list = ['rank_shift-1_ShangZheng_pct_chg', ],
                              news_col_name: str = 'sina_text'):
        # 'rank_shift-2_ShangZheng_pct_chg',
        # 'rank_shift-3_ShangZheng_pct_chg'
        self.split_date = split_date
        self.start_date = start
        self.end_data = end
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
        split_threshold = pd.to_datetime(split_date)
        temp_df = DataManipulator.df_date_cut(self.main_df, start, end, 'main_trade_date')
        temp_df.index = temp_df['main_trade_date']
        temp_df.index.name = 'main_trade_date'
        columns = [news_col_name]
        columns.extend(label_list)
        temp_df = temp_df[columns]
        self.used_columns = temp_df.columns.to_list()
        self.Bert_train_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, True, True, True),
            self.df_to_tensor(temp_df, label_list, split_threshold, True, False, True),
            window_len
        )
        self.Bert_test_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, False, True, True),
            self.df_to_tensor(temp_df, label_list, split_threshold, False, False, True),
            window_len
        )

    def minute_Bert_train_test_split(self, name: str, window_len: int, start: str = '2014-01-01', end: str = '2020-12-31',
                              split_date: str = '2020-06-01',
                              label_list: list = ['rank_shift-1_ShangZheng_30min_pct_chg', ],
                              news_col_name: str = 'sina_text'):
        self.minute_split_date = split_date
        self.minute_start_date = start
        self.minute_end_data = end
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name())
        split_threshold = pd.to_datetime(split_date)
        temp_df = DataManipulator.df_date_cut(self.news_df, start, end, 'main_trade_date')
        temp_df.index = temp_df['main_trade_date']
        temp_df.index.name = 'main_trade_date'
        columns = [news_col_name]
        columns.extend(label_list)
        temp_df = temp_df[columns]
        self.used_columns = temp_df.columns.to_list()
        self.Bert_train_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, True, True, True),
            self.df_to_tensor(temp_df, label_list, split_threshold, True, False, True),
            window_len
        )
        self.Bert_test_dict[name + str(window_len)] = innerDataset(
            self.df_to_tensor(temp_df, label_list, split_threshold, False, True, True),
            self.df_to_tensor(temp_df, label_list, split_threshold, False, False, True),
            window_len
        )


# %%
if __name__ == '__main__':
    root = r'G:\Master\master_course\Deep Learning\project\dl_project_code'
    token = 'your tushare token'
    test_manipulator = DataManipulator(root, 'datasets', 'tokenizer', token)
    test_manipulator.trade_day_init('20100101', '20201231',
                                    test_manipulator.get_file_names(root, 'datasets', 'trade_date', 'csv')[0])
    # read in csv files to dataframe
    # 中国CPI指数
    cpi = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'CPI', 'csv')[0],
                                        '日期', ['最新值', '涨跌幅', '近3月涨跌幅'], 'CPI')
    # 上海银行间同业拆放利率
    shibor = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'shibor', 'csv')[0],
                                           'date', ['on', '1w', '2w', '1m', '3m'], 'Shibor')
    # 上证大盘日度指数和指标
    shangzheng = test_manipulator.read_in_file(
        test_manipulator.get_file_names(root, 'datasets', 'ShangZheng', 'csv')[0],
        'trade_date', ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
                       'total_mv', 'float_mv', 'total_share', 'float_share',
                       'free_share', 'turnover_rate', 'turnover_rate_f', 'pe',
                       'pe_ttm', 'pb'],
        'ShangZheng')
    # 上证大盘30分钟指数和指标
    shangzheng_30min = test_manipulator.read_in_file(
        test_manipulator.get_file_names(root, 'datasets', 'ShangZheng_index_30min', 'csv')[0],
        'trade_time', ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount'],
        'ShangZheng_30min')
    # M2 广义货币量
    m2 = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'M2', 'csv')[0],
                                       '月份', ['M2数量(亿元)', 'M2同比增长', 'M2环比增长'], 'M2')
    m2 = test_manipulator.complement_df(m2, 'date')

    """
    open	    float	开盘价 （元）
    high	    float	最高价 （元）
    low	        float	最低价 （元）
    close	    float	收盘价 （元）
    pre_close	float	昨收价 （元） ** 由于有了close 所以没有pre_close
    change      float   涨跌量 （元） ** 由于有了pct_chg 所以没有change
    pct_chg	    float	涨跌幅
    vol	        float	成交量 （手）
    amount	    float	成交额 （千元）
    total_mv	float	当日总市值（元）
    float_mv	float	当日流通市值（元）
    total_share	float	当日总股本（股）
    float_share	float	当日流通股本（股）
    free_share	float	当日自由流通股本（股）
    turnover_rate	float	换手率
    turnover_rate_f	float	换手率(基于自由流通股本)
    pe	        float	市盈率
    pe_ttm	    float	市盈率TTM (最近十二月市盈率)
    pb	        float	市净率
    """
    # 人民币美元汇率
    rmb_usd = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'RMB_USD', 'csv')[0],
                                            'trade_date', ['bid_open', 'bid_close', 'bid_high', 'bid_low', 'ask_open',
                                                           'ask_close', 'ask_high', 'ask_low', 'tick_qty'], 'exchange')

    """
    bid_open	float	买入开盘价
    bid_close	float	买入收盘价
    bid_high	float	买入最高价
    bid_low	    float	买入最低价
    ask_open	float	卖出开盘价
    ask_close	float	卖出收盘价
    ask_high	float	卖出最高价
    ask_low	    float	卖出最低价
    tick_qty	int 	报价笔数
    """
    # 沪港通 沪深通 到岸 离岸资金流
    fund_flow = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'fund_flow', 'csv')[0],
                                              'trade_date', ['north_money', 'south_money'], 'fund_flow')
    """
    we only need the * north_money and * south_money
    ggt_ss	    float	港股通（上海）
    ggt_sz	    float	港股通（深圳）
    hgt	        float	沪股通（百万元）
    sgt	        float	深股通（百万元）
    north_money	float	北向资金（百万元）
    south_money	float	南向资金（百万元）
    """
    # 债券回购日行情
    repo = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'repo', 'csv')[0],
                                         'trade_date', ['repo_maturity', 'open', 'high', 'low', 'close',
                                                        'amount'], 'repo', test_manipulator.cut_time_string, (0, 10,))
    repo = test_manipulator.select_col_group_by(repo, 'repo_repo_maturity', ['GC001', 'GC007', 'GC014', 'GC028'],
                                                'date')
    """
    The Term Spread for Overnight reverse repurchase rate
    we only use the GC001, GC007, GC014, and GC028 class from repo_maturity
    trade_date	    str 	交易日期
    repo_maturity	str 	期限品种
    pre_close	    float	前收盘(%)
    open	        float	开盘价(%)
    high	        float	最高价(%)
    low	            float	最低价(%)
    close	        float	收盘价(%)
    weight	        float	加权价(%)
    weight_r	    float	加权价(利率债)(%)
    amount	        float	成交金额(万元)
    num	            int 	成交笔数(笔)
    """
    columns_type = {'create_time': str, 'text': str}
    sina_news = test_manipulator.read_in_file(test_manipulator.get_file_names(root, 'datasets', 'sina', 'csv')[0],
                                              'create_time', ['text', ], 'sina', dtypes=columns_type)

    # load in and processed selected dataframe to the main_df
    test_manipulator.add_column(cpi)
    del cpi
    test_manipulator.add_column(shibor)
    del shibor
    test_manipulator.add_column(shangzheng)
    del shangzheng

    # test_manipulator.shift_columns(['ShangZheng_pct_chg'], (-1, -2, -3), add=True)  # name has changed to
    # shift-1_ShangZheng_pct_chg
    # test_manipulator.rank_df_column(['shift-1_ShangZheng_pct_chg',
    # 'shift-2_ShangZheng_pct_chg', 'shift-3_ShangZheng_pct_chg'])  # name has changed to
    # rank_shift-1_ShangZheng_pct_chg
    test_manipulator.shift_columns(['ShangZheng_pct_chg'], (-1,),
                                   add=True)  # name has changed to shift-1_ShangZheng_pct_chg
    test_manipulator.rank_df_column(
        ['shift-1_ShangZheng_pct_chg'])  # name has changed to rank_shift-1_ShangZheng_pct_chg
    test_manipulator.news_df_add_column(shangzheng_30min)  # test_manipulator.news_df.columns
    del shangzheng_30min
    test_manipulator.shift_minute_columns(['ShangZheng_30min_pct_chg'], (-1,),
                                   add=True)
    test_manipulator.rank_minute_df_columns(['shift-1_ShangZheng_30min_pct_chg'])

    test_manipulator.add_column(m2)
    del m2
    test_manipulator.add_column(rmb_usd)
    del rmb_usd
    test_manipulator.add_column(fund_flow)
    del fund_flow
    test_manipulator.add_column(repo)
    del repo
    test_manipulator.scaling_col()
    test_manipulator.clear()
    # Do the train test split
    # without news sentiment LSTM separate
    test_manipulator.LSTM_train_test_split('LSTM_without_sentiment', 20, start='2014-01-01', end='2019-12-31',
                                           split_date='2019-05-01', label_list=['rank_shift-1_ShangZheng_pct_chg'],
                                           news_col_name='sina_text')
    print(1, test_manipulator.start_date, test_manipulator.end_data, test_manipulator.split_date)

    test_manipulator.minute_LSTM_train_test_split('minute_LSTM_without_sentiment', 20, start='2014-01-01', end='2019-12-31',
                                           split_date='2019-05-01', label_list=['rank_shift-1_ShangZheng_30min_pct_chg'],
                                           news_col_name='sina_text')
    print(1, test_manipulator.minute_start_date, test_manipulator.minute_end_data, test_manipulator.minute_split_date)
    # Bert parameter selection separate
    test_manipulator.add_change_news('sina', (7, 9), columns_type, sina_news, time_col_name='create_time')
    test_manipulator.Bert_train_test_split('Bert(7,9)', 1, start='2015-01-01', end='2019-12-31',
                                           split_date='2018-05-01', label_list=['rank_shift-1_ShangZheng_pct_chg'],
                                           news_col_name='sina_text')
    print(2, test_manipulator.start_date, test_manipulator.end_data, test_manipulator.split_date)
    test_manipulator.add_change_news('sina', (11, 13), columns_type, sina_news, time_col_name='create_time')
    test_manipulator.Bert_train_test_split('Bert(11,13)', 1, label_list=['rank_shift-1_ShangZheng_pct_chg'],
                                           news_col_name='sina_text')
    print(3, test_manipulator.start_date, test_manipulator.end_data, test_manipulator.split_date)
    test_manipulator.add_change_news('sina', (15, 17), columns_type, sina_news, time_col_name='create_time')
    test_manipulator.Bert_train_test_split('Bert(15,17)', 1, start='2016-01-01', end='2018-12-31',
                                           split_date='2017-05-01', label_list=['rank_shift-1_ShangZheng_pct_chg'],
                                           news_col_name='sina_text')
    print(4, test_manipulator.start_date, test_manipulator.end_data, test_manipulator.split_date)

    test_manipulator.add_minute_change_news('sina', columns_type, sina_news, time_col_name='create_time')
    test_manipulator.minute_Bert_train_test_split('minute_Bert', 1, start='2016-01-01', end='2018-12-31',
                                           split_date='2017-05-01', label_list=['rank_shift-1_ShangZheng_30min_pct_chg'],
                                           news_col_name='sina_text')
    a = iter(test_manipulator.Bert_test_dict['Bert(15,17)1'])
