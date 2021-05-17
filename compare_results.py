# %%
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd

# %%
df_without_sentiment = pd.read_csv(
    r'G:\Master\master_course\Deep Learning\project\dl_project_code\results\LSTM_total_results_2017-2019_nosentiment_6class.csv')
df_with_bert_sentiment = pd.read_csv(
    r'G:\Master\master_course\Deep Learning\project\dl_project_code\results\LSTM_total_results_2017-2019_bert_sentiment_6class.csv')
df_with_svm_sentiment = pd.read_csv(
    r'G:\Master\master_course\Deep Learning\project\dl_project_code\results\LSTM_total_results_2017-2019_svm_sentiment_6class.csv')

df_without_sentiment_2class = pd.read_csv(
    r'G:\Master\master_course\Deep Learning\project\dl_project_code\results\LSTM_total_results_2017-2019_nosentiment_2class.csv')
df_with_svm_sentiment_2class = pd.read_csv(
    r'G:\Master\master_course\Deep Learning\project\dl_project_code\results\LSTM_total_results_2017-2019_svm_sentiment_2class.csv')


# %%
def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
    file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
    return file_list


root = r'G:\Master\master_course\Deep Learning\project\dl_project_code'
# %%
print('df_with_bert_sentiment mean accu:', df_with_bert_sentiment.test_accu_mean.mean())
print('df_with_svm_sentiment mean accu:', df_with_svm_sentiment.test_accu_mean.mean())
print('df_without_sentiment mean accu:', df_without_sentiment.test_accu_mean.mean())

print('df_with_svm_sentiment_2class mean accu:', df_with_svm_sentiment_2class.test_accu_mean.mean())
print('df_without_sentiment_2class mean accu:', df_without_sentiment_2class.test_accu_mean.mean())

# df_with_bert_sentiment mean accu: 0.16237429156899452
# df_with_svm_sentiment mean accu: 0.1699220376710097
# df_without_sentiment mean accu: 0.17897038037578264
# df_with_svm_sentiment_2class mean accu: 0.4946614404519399
# df_without_sentiment_2class mean accu: 0.4975068271160126
# %%
print('df_with_bert_sentiment max accu:', df_with_bert_sentiment.best_test_accu.max())
print('df_with_svm_sentiment max accu:', df_with_svm_sentiment.test_accu_mean.max())
print('df_without_sentiment max accu:', df_without_sentiment.best_test_accu.max())

print('df_with_svm_sentiment_2class max accu:', df_with_svm_sentiment_2class.test_accu_mean.max())
print('df_without_sentiment_2class max accu:', df_without_sentiment_2class.best_test_accu.max())


# df_with_bert_sentiment mean accu: 0.16237429156899452
# df_with_svm_sentiment mean accu: 0.1699220376710097
# df_without_sentiment mean accu: 0.17897038037578264
# df_with_svm_sentiment_2class mean accu: 0.4946614404519399
# df_without_sentiment_2class mean accu: 0.4975068271160126

# %%
def df_to_latex_table(df, caption='', label=''):
    '''
    this function output the latex code for table of dataframe
    Notice that
        1. this function will NOT include the index of row !
        2. this function can handdle non-ASCII symbols but latex may NOT
             if your package setting do not support non-ASCII symbols !
    df: dataframe
    caption : caption in the latex code
    lable : label in latex code

    '''
    n_col = len(df.columns)
    print('\\begin{table}[t]  \n\
            \\caption{' + caption + '} \n\
            \\label{' + label + '} \n\
            \\centering \n\
            \\begin{tabular}{' + 'l' * (n_col) + '} \n\
            \\toprule')

    for colname in list(df.columns)[:n_col - 1]:
        print(colname, end='&')
    print(list(df.columns)[-1], '\\\\')
    print('\\midrule')

    for row in df.iterrows():
        for colvalve in row[1][:len(row[1]) - 1]:
            print(colvalve, end='&')
        print(row[1][-1], '\\\\')

    print('\\bottomrule\n\\end{tabular}\n\\end{table}')


# %%
LSTM_total_results_2class = pd.read_csv(get_file_names(root, 'results', 'LSTM_total_results_2class', 'csv')[0], index_col=0)
LSTM_total_results_6class = pd.read_csv(get_file_names(root, 'results', 'LSTM_total_results_6class', 'csv')[0], index_col=0)
LSTM_total_results_2class.drop_duplicates(inplace=True)
LSTM_total_results_6class.drop_duplicates(inplace=True)
LSTM_total_results_2class = LSTM_total_results_2class[['num_labels', 'input_dim', 'train_loss_mean', 'test_accu_mean', 'best_test_accu']]
LSTM_total_results_6class = LSTM_total_results_6class[['num_labels', 'input_dim', 'train_loss_mean', 'test_accu_mean', 'best_test_accu']]

LSTM_total_results_2class.sort_values(by=['best_test_accu', 'test_accu_mean'], inplace=True, ascending=False)
LSTM_total_results_6class.sort_values(by=['best_test_accu', 'test_accu_mean'], inplace=True, ascending=False)
LSTM_total_results_2class = LSTM_total_results_2class.groupby('input_dim').head(1).reset_index()
LSTM_total_results_6class = LSTM_total_results_6class.groupby('input_dim').head(1).reset_index()
# [['train_loss_mean', 'test_accu_mean', 'best_test_accu']]
# [['train_loss_mean', 'test_accu_mean', 'best_test_accu']]
LSTM_total_results_2class = LSTM_total_results_2class.round(4)
LSTM_total_results_6class = LSTM_total_results_6class.round(4)

df_to_latex_table(LSTM_total_results_2class)

df_to_latex_table(LSTM_total_results_6class)