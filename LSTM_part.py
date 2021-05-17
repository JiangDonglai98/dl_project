"""
author: Donglai jiang
"""
# %%
import numpy as np
import pandas as pd

from dataManipulator import DataManipulator
import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from pprint import pformat
import os
import json

plt.style.use('ggplot')


# %%
class Args:
    def __init__(self,
                 batch_size=20,  # 'input batch size for training (default: 64)'
                 test_batch_size=20,  # 'input batch size for testing (default: 1000)'
                 epochs=200,  # 'number of epochs to train (default: 20)'
                 lr=0.01,  # 'learning rate (default: 0.001)'
                 momentum=0.9,  # 'SGD momentum (default: 0.9)'
                 weight_decay=0.0005,  # 'weight decay (default: 0.0005)'
                 no_cuda=False,  # 'enables CUDA training'
                 seed=1,  # 'random seed (default: 1)'
                 log_interval=500,  # 'how many batches to wait before logging training status'
                 num_workers=0
                 ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = 2
        self.weight_decay = weight_decay
        self.no_cuda = no_cuda
        self.seed = seed
        self.log_interval = log_interval
        self.num_workers = num_workers
        self.input_dim = 64
        self.hidden_dim = 128
        self.num_layers = 2
        self.output_dim = 6


args = Args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.cuda)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
torch.cuda.get_device_name()


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


# %%
# Load data
root = r'G:\Master\master_course\Deep Learning\project\dl_project_code'
token = 'your tushare token'
data_manipulator = DataManipulator(root, 'datasets', 'tokenizer', token)
data_manipulator.trade_day_init('20100101', '20201231',
                                data_manipulator.get_file_names(root, 'datasets', 'trade_date', 'csv')[0])


# read in csv files to dataframe
def gen_main_df(add_list: list):
    """
    chose the finance list to get the final main_df in the data_manipulator
    :param add_list:  the list you want to add in
    ['cpi', 'shibor', 'shangzheng', 'm2', 'rmb_usd', 'fund_flow', 'repo', 'sina_news', 'scale', 'clear']
    :return: DataFrame
    """
    # 由Bert 计算得来的 sentiment信息
    if 'sentiment' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('sentiment')
        sentiment = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'daily_svm_sentiment_6class' , 'csv')[0],
                                              'date', ['0'], 'sentiment')   # 'daily_svm_sentiment_2class'  '0', '1', '2', '3', '4', '5'
        data_manipulator.add_column(sentiment)
    # 中国CPI指数
    if 'cpi' in add_list and 'cpi' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('cpi')
        cpi = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'CPI', 'csv')[0],
                                            '日期', ['最新值', '涨跌幅', '近3月涨跌幅'], 'CPI')
        data_manipulator.add_column(cpi)
    # 上海银行间同业拆放利率
    if 'shibor' in add_list and 'shibor' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('shibor')
        shibor = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'shibor', 'csv')[0],
                                               'date', ['on', '1w', '2w', '1m', '3m'], 'Shibor')
        data_manipulator.add_column(shibor)
    # 上证综指
    if 'shangzheng' in add_list and 'shangzheng' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('shangzheng')
        shangzheng = data_manipulator.read_in_file(
            data_manipulator.get_file_names(root, 'datasets', 'ShangZheng', 'csv')[0],
            'trade_date', ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount',
                           'total_mv', 'float_mv', 'total_share', 'float_share',
                           'free_share', 'turnover_rate', 'turnover_rate_f', 'pe',
                           'pe_ttm', 'pb'],
            'ShangZheng')
        data_manipulator.add_column(shangzheng)
        data_manipulator.shift_columns(['ShangZheng_pct_chg'], (-1,),
                                       add=True)  # name has changed to shift-1_ShangZheng_pct_chg
        data_manipulator.rank_df_column(['shift-1_ShangZheng_pct_chg'],
                                        rank_list=[-10, -1, -0.5, 0, 0.5, 1, 10])  # rank_list=[-10, 0, 10]  [-10, -1, -0.5, 0, 0.5, 1, 10]
        shangzheng_30min = data_manipulator.read_in_file(
            data_manipulator.get_file_names(root, 'datasets', 'ShangZheng_index_30min', 'csv')[0],
            'trade_time', ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount'],
            'ShangZheng_30min')
        data_manipulator.news_df_add_column(shangzheng_30min)
        data_manipulator.shift_minute_columns(['ShangZheng_30min_pct_chg'], (-1,),
                                              add=True)
        data_manipulator.rank_minute_df_columns(['shift-1_ShangZheng_30min_pct_chg'],
                                                rank_list=[-10, -1, -0.5, 0, 0.5, 1, 10])  # rank_list=[-10, 0, 10]  [-10, -1, -0.5, 0, 0.5, 1, 10]

    # M2 广义货币量
    if 'm2' in add_list and 'm2' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('m2')
        m2 = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'M2', 'csv')[0],
                                           '月份', ['M2数量(亿元)', 'M2同比增长', 'M2环比增长'], 'M2')
        m2 = data_manipulator.complement_df(m2, 'date')
        data_manipulator.add_column(m2)

    # 人民币美元汇率
    if 'rmb_usd' in add_list and 'rmb_usd' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('rmb_usd')
        rmb_usd = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'RMB_USD', 'csv')[0],
                                                'trade_date',
                                                ['bid_open', 'bid_close', 'bid_high', 'bid_low', 'ask_open',
                                                 'ask_close', 'ask_high', 'ask_low', 'tick_qty'], 'exchange')
        data_manipulator.add_column(rmb_usd)

    # 沪港通 沪深通 到岸 离岸资金流
    if 'fund_flow' in add_list and 'fund_flow' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('fund_flow')
        fund_flow = data_manipulator.read_in_file(
            data_manipulator.get_file_names(root, 'datasets', 'fund_flow', 'csv')[0],
            'trade_date', ['north_money', 'south_money'], 'fund_flow')
        data_manipulator.add_column(fund_flow)

    # 债券回购日行情
    if 'repo' in add_list and 'repo' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('repo')
        repo = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'repo', 'csv')[0],
                                             'trade_date', ['repo_maturity', 'open', 'high', 'low', 'close',
                                                            'amount'], 'repo', data_manipulator.cut_time_string,
                                             (0, 10,))
        repo = data_manipulator.select_col_group_by(repo, 'repo_repo_maturity', ['GC001', 'GC007', 'GC014', 'GC028'],
                                                    'date')
        data_manipulator.add_column(repo)

    # 新浪新闻
    if 'sina_news' in add_list and 'sina_news' not in data_manipulator.used_measure_list:
        data_manipulator.used_measure_list.append('sina_news')
        columns_type = {'create_time': str, 'text': str}
        sina_news = data_manipulator.read_in_file(data_manipulator.get_file_names(root, 'datasets', 'sina', 'csv')[0],
                                                  'create_time', ['text', ], 'sina', dtypes=columns_type)
        data_manipulator.add_change_news('sina', (7, 9), columns_type, sina_news, time_col_name='create_time')
        data_manipulator.add_minute_change_news('sina', columns_type, sina_news, time_col_name='create_time')
    if 'scale' in add_list:
        data_manipulator.scaling_col()
    if 'clear' in add_list:
        data_manipulator.clear()


# load in and processed selected dataframe to the main_df
# # Do the train test split
# # without news sentiment LSTM separate
# data_manipulator.LSTM_train_test_split('LSTM_without_sentiment', 20, news_col_name='sina_text')
# data_manipulator.LSTM_train_test_split('LSTM_without_sentiment', 10, news_col_name='sina_text')
# data_manipulator.LSTM_train_test_split('LSTM_without_sentiment', 30, news_col_name='sina_text')
# # Bert parameter selection separate
# data_manipulator.add_change_news('sina', (7, 9), columns_type, sina_news, time_col_name='create_time')
# data_manipulator.Bert_train_test_split('Bert(7,9)', 1, news_col_name='sina_text')
# data_manipulator.add_change_news('sina', (11, 13), columns_type, sina_news, time_col_name='create_time')
# data_manipulator.Bert_train_test_split('Bert(11,13)', 1, news_col_name='sina_text')
# data_manipulator.add_change_news('sina', (15, 17), columns_type, sina_news, time_col_name='create_time')
# data_manipulator.Bert_train_test_split('Bert(15,17)', 1, news_col_name='sina_text')
# del sina_news


# %%
def loader_generator(train_set, test_set, args: Args):
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)  # , shuffle=True
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, num_workers=args.num_workers)

    return train_loader, test_loader


def calculate_recall(f_x, y_true) -> float:
    if type(f_x) == torch.tensor and type(y_true) == torch.tensor:
        if f_x.device != 'cpu':
            f_x = f_x.cpu()
        if y_true.device != 'cpu':
            y_true = y_true.cpu()
    return sum((f_x == 1) & (y_true == 1)) / sum(y_true == 1)


def calculate_precision(f_x, y_true) -> float:
    if type(f_x) == torch.tensor and type(y_true) == torch.tensor:
        if f_x.device != 'cpu':
            f_x = f_x.cpu()
        if y_true.device != 'cpu':
            y_true = y_true.cpu()
    return sum((f_x == 1) & (y_true == 1)) / sum(f_x == 1)


def calculate_F1(f_x, y_true) -> float:
    recall = calculate_recall(f_x, y_true)
    precision = calculate_precision(f_x, y_true)
    return 2 * recall * precision / (recall + precision)


# train_loader20, test_loader20 = loader_generator(data_manipulator.LSTM_train_dict['LSTM_without_sentiment20'],
#                                                  data_manipulator.LSTM_test_dict['LSTM_without_sentiment20'], args)
# train_loader10, test_loader10 = loader_generator(data_manipulator.LSTM_train_dict['LSTM_without_sentiment10'],
#                                                  data_manipulator.LSTM_test_dict['LSTM_without_sentiment10'], args)
# train_loader30, test_loader30 = loader_generator(data_manipulator.LSTM_train_dict['LSTM_without_sentiment30'],
#                                                  data_manipulator.LSTM_test_dict['LSTM_without_sentiment30'], args)


# %%
def train(epoch, model, loss_func, train_loader, args):
    model.train()
    correct = 0
    train_loss = 0
    y_predict = []
    y_true = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        # print(data.shape)
        target = target.flatten().long()
        output = model(data)
        pred = output.data.max(1)[1]
        y_predict.append(pred)
        loss = loss_func(output, target)
        # print('output:',output, '\n', 'target:', target)
        train_loss += loss
        y_true.append(target.data)
        correct += pred.eq(target.data).cpu().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))  # loss.data[0]  loss.item()
    # if args.cuda:
    #     train_loss = train_loss.detach().cpu().numpy()
    # print('output:', output, '\n', 'target:', target)
    train_loss /= len(train_loader)
    y_predict = torch.cat(y_predict)
    y_true = torch.cat(y_true)
    # recall_rate = calculate_recall(y_predict, y_true)
    # precision_rate = calculate_precision(y_predict, y_true)
    # F1_rate = calculate_F1(y_predict, y_true)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    # print('Train set ---- recall_rate: {:.2f}% ---- precision_rate: {:.2f}% ---- F1_rate {:.2f}% '.format(
    #     recall_rate * 100, precision_rate * 100, F1_rate * 100))
    return train_loss, correct / len(train_loader.dataset)  # , F1_rate


def test(epoch, model, loss_func, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    y_predict = []
    y_true = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data, volatile=True), Variable(target)
        target = target.flatten().long()
        output = model(data)
        test_loss += loss_func(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        y_predict.append(pred)
        y_true.append(target.data)
        correct += pred.eq(target.data).cpu().sum()
    # if args.cuda:
    #     test_loss = test_loss.detach().cpu().numpy()
    test_loss /= len(test_loader)  # loss function already averages over batch size
    # y_predict = torch.cat(y_predict)
    # y_true = torch.cat(y_true)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # recall_rate = calculate_recall(y_predict, y_true)
    # precision_rate = calculate_precision(y_predict, y_true)
    # F1_rate = calculate_F1(y_predict, y_true)
    # print('Validation set ---- recall_rate: {:.2f}% ---- precision_rate: {:.2f}% ---- F1_rate {:.2f}% '.format(
    #     recall_rate * 100,
    #     precision_rate * 100,
    #     F1_rate * 100))
    return correct / len(test_loader.dataset)  # , F1_rate


loss_accuracy_dict = {}


def main(train_func, test_func, train_loader, test_loader, exp_name, net, loss_func, args):
    train_loss_list = []
    train_accu_list = []
    # train_F1_list = []
    test_accu_list = []
    # test_F1_list = []
    for epoch in range(1, args.epochs + 1):
        temp_train = train_func(epoch, net, loss_func, train_loader, args)
        temp_test = test_func(epoch, net, loss_func, test_loader, args)
        train_loss_list.append(temp_train[0])
        train_accu_list.append(temp_train[1])
        # train_F1_list.append(temp_train[2])
        test_accu_list.append(temp_test)  # .append(temp_test[0])
        # test_F1_list.append(temp_test[1])
    loss_accuracy_dict[exp_name] = {}
    loss_accuracy_dict[exp_name]['train_loss_list'] = train_loss_list
    loss_accuracy_dict[exp_name]['train_accu_list'] = train_accu_list
    loss_accuracy_dict[exp_name]['test_accu_list'] = test_accu_list
    # loss_accuracy_dict[exp_name]['train_F1_list'] = train_F1_list
    # loss_accuracy_dict[exp_name]['test_F1_list'] = test_F1_list
    torch.save(net.state_dict(), exp_name + '.pkl')
    print(f"finish training and testing for {exp_name}.")
    return {'train_loss_list': train_loss_list, 'train_accu_list': train_accu_list, 'test_accu_list': test_accu_list}


# %%
def run_model(path: str, exp_name, window_len, net, loss_func, args: Args, manipulator: DataManipulator,
              start='2014-01-01', end='2020-12-31', split_date='2020-06-01', is_Bert: bool = False,
              is_minute: bool = False):
    if not is_Bert:
        result_name = exp_name + f' start_{start} end_{end} split_{split_date} '
        if not is_minute:
            manipulator.LSTM_train_test_split(result_name, window_len, start=start, end=end,
                                              split_date=split_date, news_col_name='sina_text',
                                              normal=False)  # , normal=False
        else:
            manipulator.minute_LSTM_train_test_split(result_name, window_len, start=start, end=end,
                                                     split_date=split_date, news_col_name='sina_text', normal=False)
        train_loader, test_loader = loader_generator(manipulator.LSTM_train_dict[result_name + str(window_len)],
                                                     manipulator.LSTM_test_dict[result_name + str(window_len)],
                                                     args)
        args.input_dim = len(manipulator.used_columns) - 1
        args.output_dim = manipulator.label_num
        print('The output dim is:', args.output_dim)
        model = net(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim)
        if args.cuda:
            model.cuda()
        global optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        training_data = main(train, test, train_loader, test_loader, exp_name, model, loss_func, args)
        out_put_path = os.path.join(path, 'results')
        print(out_put_path)
        model_params = {'input_dim': args.input_dim, 'hidden_dim': args.hidden_dim,
                        'num_layers': args.num_layers, 'output_dim': args.output_dim,
                        'lr': args.lr, 'weight_decay': args.weight_decay}
        plot_path = os.path.join(path, 'plots')
        file_name = result_name + f'windowlen_{window_len} inputlen_{args.input_dim} class_{args.output_dim} hiddenlen_{args.hidden_dim} layers_{args.num_layers} lr_{args.lr} wd_{args.weight_decay}'
        draw_train_loss(training_data['train_loss_list'], file_name, plot_path)
        draw_train_accuracy(training_data['train_accu_list'], file_name, plot_path)
        draw_test_accuracy(training_data['test_accu_list'], file_name, plot_path)
        print('mean_train_loss', torch.mean(torch.stack(training_data['train_loss_list'])),
              'mean_train_accu', torch.mean(torch.stack(training_data['train_accu_list'])).numpy(),
              'mean_test_accu', torch.mean(torch.stack(training_data['test_accu_list'])).numpy(),
              'best_train_accu', torch.max(torch.stack(training_data['train_accu_list'])).numpy(),
              'best_test_accu', torch.max(torch.stack(training_data['test_accu_list'])).numpy())
        training_stats = {'train_loss': torch.mean(torch.stack(training_data['train_loss_list'])).detach().item(),
                          'train_accu': float(torch.mean(torch.stack(training_data['train_accu_list'])).numpy()),
                          'test_accu': float(torch.mean(torch.stack(training_data['test_accu_list'])).numpy()),
                          'best_train_accu': float(torch.max(torch.stack(training_data['train_accu_list'])).numpy()),
                          'best_test_accu': float(torch.max(torch.stack(training_data['test_accu_list'])).numpy())
                          }

        gen_log(out_put_path, file_name, start, end, split_date, args.batch_size, args.epochs, manipulator.label_num,
                manipulator.used_columns,
                model_params, training_stats)

    # Bert parameter selection separate
    # data_manipulator.add_change_news('sina', (7, 9), columns_type, sina_news, time_col_name='create_time')
    # data_manipulator.Bert_train_test_split('Bert(7,9)', 1, news_col_name='sina_text')


# %%
def draw_train_loss(train_loss_list, exp_name, path):
    try:
        plt.figure(1, figsize=(8, 8))
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        x_list = list(range(1, len(train_loss_list) + 1))
        train_loss_list = list(map(lambda x: x.cpu().detach().numpy(), train_loss_list))
        plt.plot(x_list, train_loss_list, color='red', label='train loss')
        title = 'Train_loss_plot'
        plt.title(title)
        # plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path, title + exp_name + '.png'))
        plt.show()
    except Exception as e:
        print(e)


def draw_train_accuracy(train_accu_list, exp_name, path):
    try:
        plt.figure(1, figsize=(8, 8))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        train_accu_list = list(map(lambda x: x.cpu().numpy(), train_accu_list))
        plt.plot(list(range(1, len(train_accu_list) + 1)), train_accu_list, color='yellow', label='train accuracy')
        title = 'Train_accuracy_plot'
        plt.title(title)
        # plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path, title + exp_name + '.png'))
        plt.show()
    except Exception as e:
        print(e)


def draw_train_F1(train_F1_list, exp_name, path):
    try:
        plt.figure(1, figsize=(8, 8))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        train_F1_list = list(map(lambda x: x.cpu().numpy(), train_F1_list))
        plt.plot(list(range(1, len(train_F1_list) + 1)), train_F1_list, color='blue', label='train F1 score')
        title = 'Train_F1_score_plot'
        plt.title(title)
        # plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path, title + exp_name + '.png'))
        plt.show()
    except Exception as e:
        print(e)


def draw_test_F1(test_F1_list, exp_name, path):
    try:
        plt.figure(1, figsize=(8, 8))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        test_F1_list = list(map(lambda x: x.cpu().numpy(), test_F1_list))
        plt.plot(list(range(1, len(test_F1_list) + 1)), test_F1_list, color='orange', label='validation F1 score')
        title = 'Validation_F1_score_plot'
        plt.title(title)
        # plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path, title + exp_name + '.png'))
        plt.show()
    except Exception as e:
        print(e)


def draw_test_accuracy(test_accu_list, exp_name, path):
    try:
        plt.figure(1, figsize=(8, 8))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        test_accu_list = list(map(lambda x: x.cpu().numpy(), test_accu_list))
        plt.plot(list(range(1, len(test_accu_list) + 1)), test_accu_list, color='green', label='validation accuracy')
        title = 'Validation_accuracy_plot'
        plt.title(title)
        # plt.grid()
        plt.legend()
        plt.savefig(os.path.join(path, title + exp_name + '.png'))
        plt.show()
    except Exception as e:
        print(e)


# %%
LSTM_total_dict = {"start": [], "end": [], "split_date": [], "batch_size": [], "epochs": [], "num_labels": [],
                   "input_dim": [], "hidden_dim": [], "num_layers": [], "output_dim": [], 'lr': [], 'weight_decay': [],
                   "train_loss_mean": [], "train_accu_mean": [], "test_accu_mean": [], 'best_train_accu': [],
                   'best_test_accu': []}


def write_result():
    df = pd.DataFrame(LSTM_total_dict)
    df.to_csv(os.path.join(os.path.join(root, 'results'), 'LSTM_total_results.csv'))


def gen_log(path: str, model_name, startDate, endDate, splitDate, batch_size, epochs, num_labels, feature_list,
            model_params, training_stats):
    """
    :param path:
    :param model_name:
    :param startDate:
    :param endDate:
    :param splitDate:
    :param batch_size:
    :param epochs:
    :param num_labels:
    :param feature_list:
    :param model_params: contain the params of the LSTM or Bert Model
    :param training_stats:
    :return:
    """
    LSTM_total_dict["start"].append(startDate)
    LSTM_total_dict["end"].append(endDate)
    LSTM_total_dict["split_date"].append(splitDate)
    LSTM_total_dict["batch_size"].append(batch_size)
    LSTM_total_dict["epochs"].append(epochs)
    LSTM_total_dict["num_labels"].append(num_labels)
    LSTM_total_dict["input_dim"].append(model_params['input_dim'])
    LSTM_total_dict["hidden_dim"].append(model_params['hidden_dim'])
    LSTM_total_dict["num_layers"].append(model_params['num_layers'])
    LSTM_total_dict["output_dim"].append(model_params['output_dim'])
    LSTM_total_dict["lr"].append(model_params['lr'])
    LSTM_total_dict["weight_decay"].append(model_params['weight_decay'])
    LSTM_total_dict["train_loss_mean"].append(training_stats['train_loss'])
    LSTM_total_dict["train_accu_mean"].append(training_stats['train_accu'])
    LSTM_total_dict["test_accu_mean"].append(training_stats['test_accu'])
    LSTM_total_dict["best_train_accu"].append(training_stats['best_train_accu'])
    LSTM_total_dict["best_test_accu"].append(training_stats['best_test_accu'])

    with open(os.path.join(path, model_name + '.txt'),
              'w') as f:
        f.write('model_name = {}\n'.format(model_name))
        f.write('start = {}'.format(startDate))
        f.write('split_date = {}'.format(splitDate))
        f.write('end = {}'.format(endDate))
        f.write('batch_size = {}\n'.format(batch_size))
        f.write('epochs = {}\n'.format(epochs))
        f.write('num_labels = {}\n'.format(num_labels))
        f.write('feature_list = {}\n'.format(feature_list))
        f.write('model_params = {}\n'.format(model_params))
        f.write('training_stats = {}\n'.format(pformat(training_stats)))

    with open(os.path.join(path, model_name + '.json'),
              'w') as j:
        # LSTM params have following
        # input_dim = 64
        # hidden_dim = 128
        # num_layers = 2
        total_dict = {}
        total_dict['model_name'] = model_name
        total_dict['start'] = startDate
        total_dict['end'] = endDate
        total_dict['split_date'] = splitDate
        total_dict['batch_size'] = batch_size
        total_dict['epochs'] = epochs
        total_dict['num_labels'] = num_labels
        total_dict['feature_list'] = feature_list
        total_dict['model_params'] = model_params
        total_dict['training_stats'] = training_stats
        json.dump(total_dict, j)
    write_result()


# %%
# count = 0
# for data, target in train_loader:
#     print(data.shape, target.shape)
#     print(data, target)
#     count += 1
#     if count == 1:
#         break


# %%
class MarketLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MarketLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # bidirectional=True then the bidirectional cyclic neural network is obtained
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)  # bidirectional=True

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # Readout layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim, bias=True)  # 2 *

    def forward(self, x):
        # # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        # # Initialize cell state
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, None)
        out = self.fc(out[:, -1, :])  # [:, -1, :]

        return out


# # %%
# input_dim = 64
# hidden_dim = 128
# num_layers = 2
# output_dim = 6
# root = r'G:\Master\master_course\Deep Learning\project\dl_project_code'


# ['cpi', 'shibor', 'shangzheng', 'm2', 'rmb_usd', 'repo', 'fund_flow', 'sina_news', 'scale', 'clear']
# %%
def check_data():
    a = list(data_manipulator.LSTM_train_dict.values())[0]
    a = iter(a)
    b = list(data_manipulator.LSTM_test_dict.values())[0]
    b = iter(b)
    return a, b


def check(a, b):
    print(a.__next__()[0][0])
    print(b.__next__()[0][0])


# %%
@timer
def tuning():
    for epochs in [100]:
        for hidden_dim in [128]:
            for lr in [0.001, 0.01]:
                for weight in [0, 0.01]:
                    # for end, split in zip(['2020-12-31', '2019-12-31'], ['2020-06-01', '2019-06-01']):
                    args.epochs = epochs
                    args.hidden_dim = hidden_dim
                    args.lr = lr
                    args.weight_decay = weight
                    loss_func = torch.nn.CrossEntropyLoss()
                    run_model(root, 'LSTM_with_daily_svm_sentiment_6class', 20, MarketLSTM, loss_func, args,
                              data_manipulator,
                              start='2017-01-01', end='2019-12-31', split_date='2019-06-01', is_Bert=False)


# %%
# Model1 'LSTM_without_sentiment'
gen_main_df(['shangzheng', 'scale', 'clear'])
# # args.epochs = 100
# # args.lr = 0.01
# # args.hidden_dim = 128
# # loss_func = torch.nn.CrossEntropyLoss()
# # run_model(root, 'LSTM_without_sentiment', 20, MarketLSTM, loss_func, args, data_manipulator,
# #           start='2014-01-01', end='2020-12-31', split_date='2020-06-01', is_Bert=False)
tuning()

# %%
gen_main_df(['cpi', 'shibor', 'shangzheng', 'm2', 'repo', 'scale', 'clear'])  # 'scale',
# args.epochs = 50
# loss_func = torch.nn.CrossEntropyLoss()
# run_model(root, 'LSTM_without_sentiment', 20, MarketLSTM, loss_func, args, data_manipulator,
#           start='2014-01-01', end='2020-12-31', split_date='2020-06-01', is_Bert=False)
tuning()

# %%
gen_main_df(['cpi', 'shibor', 'shangzheng', 'm2', 'repo', 'rmb_usd', 'fund_flow', 'scale',  'clear'])  # 'scale',
# args.epochs = 50
# loss_func = torch.nn.CrossEntropyLoss()
# run_model(root, 'LSTM_without_sentiment', 20, MarketLSTM, loss_func, args, data_manipulator,
#           start='2014-01-01', end='2020-12-31', split_date='2020-06-01', is_Bert=False)
tuning()
