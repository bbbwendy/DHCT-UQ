import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 增加操作条件列
def add_operating_condition(df):
    df_op_cond = df.copy()
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)
    return df_op_cond

# 数据归一化处理
def condition_scaler(df_train, df_test, sensor_names):
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_test

# 指数平滑
def exponential_smoothing(df, sensors, alpha=0.4):
    df = df.copy()
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
    return df

# 生成测试数据
def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values
    else:
        data_matrix = df[columns].values
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    yield data_matrix[start:stop, :]

def get_test_loader(dataset, sensors, sequence_length, alpha, threshold, batch_size):
    dir_path = './data/'
    test_file = 'test_' + dataset + '.txt'
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i + 1) for i in range(21)]
    col_names = index_names + setting_names + sensor_names
    
    # 读取测试文件
    test = pd.read_csv((dir_path + test_file), sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_' + dataset + '.txt'), sep=r'\s+', header=None, names=['RemainingUsefulLife'])

    # 删除不必要的传感器
    drop_sensors = [element for element in sensor_names if element not in sensors]
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    
    # 数据归一化处理和指数平滑
    X_test_pre = condition_scaler(X_test_pre, X_test_pre, sensors)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, alpha)

    # 测试集生成
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)
    y_test['RemainingUsefulLife'] = y_test['RemainingUsefulLife'].clip(upper=threshold)

    x_test = torch.tensor(x_test).to(torch.float32)
    y_test_rul = torch.tensor(y_test['RemainingUsefulLife']).to(torch.float32)

    # 创建测试集的TensorDataset
    test_dataset = TensorDataset(x_test, y_test_rul)

    # 如果 batch_size == -1，则使用所有数据作为一个批次
    if batch_size == -1:
        batch_size = len(x_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
