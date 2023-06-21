import sys

sys.path.append(r".")
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from time_series_model.train import train, test
from time_series_model.args import Model_args
from data.get_db import get_data_period
from statsmodels.tsa.seasonal import seasonal_decompose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adjust_col(df, target):
    """
    Adjusts the column order of a DataFrame to have 'Date' and the specified 'target' column as the first two columns.

    Args:
        df: A pandas DataFrame representing the dataset.
        target: A string indicating the target column.

    Returns:
        A modified DataFrame with the column order adjusted.
    """
    df = df[['Date', target] + [col for col in df.columns if col != 'Date' and col != target]]

    return df


def filter_time(df):
    """
    Filters the DataFrame by converting the 'Date' column to datetime format,
    resampling the data at 1-minute intervals, and filling missing values with the previous non-NaN value.

    Args:
        df: A pandas DataFrame representing the dataset.

    Returns:
        A modified DataFrame with the filtered data.
    """
    # 將 'date' 欄位轉換為時間格式
    df['Date'] = pd.to_datetime(df['Date'])

    # 以每 1 分鐘為間隔重新獲得資料
    df = df.resample('1T', on='Date').first()

    # 使用前一個非NaN值
    df = df.fillna(method='ffill')

    return df


def make_features(df, col1, col2):
    """
    Creates new features in the DataFrame based on the specified columns.

    Args:
        df: A pandas DataFrame representing the dataset.
        col1: A string indicating the first column.
        col2: A string indicating the second column.

    Returns:
        A modified DataFrame with new features created based on the specified columns.
    """
    # 判斷是哪一種特徵,'tM'開頭則兩行相減,'PM'開頭則是下一筆資料減去目前這一筆資料
    if col1[: 2] == 'tM':
        df[col1[-1] + col2[-1]] = abs(df[col1] - df[col2])
        df = df.drop([col1, col2], axis=1)
    else:
        df[col1] = df[col1].shift(-1) - df[col1]
        df = df.fillna(method='ffill')
    return df


def seasonal_decomp(seq_len, df, target):
    """
    Performs seasonal decomposition of a time series using STL decomposition.

    Args:
        seq_len: An integer representing the length of the seasonal cycle.
        df: A pandas DataFrame containing the time series data.
        target: A string indicating the target column.

    Returns:
        The modified DataFrame with additional columns for trend, seasonal, and residual components.
    """
    # 對時間序列進行STL分解
    decomposition = seasonal_decompose(df[target], period=seq_len, extrapolate_trend='freq')
    # 獲取分解結果的趨勢、季節性和殘差部分
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # 將分解結果合併到原始資料集
    df['trend'] = trend
    df['seasonal'] = seasonal
    df['residual'] = residual
    return df


def preprocess_dataset(seq_len, model_name, start_date, end_date):
    """
    Obtain data from the database and perform data preprocessing.

    Args:
        seq_len: seq len.
        model_name: model name.
        start_date:  data start date.
        end_date: data end date.

    Returns:
        The dataframe required by the corresponding model.
    """

    # 溫度及相關資料

    #室內資料
    indoor_data = get_data_period('indoor', start_date, end_date)
    indoor_data = pd.DataFrame(indoor_data, columns=["Date", "temp", "humi", "CO2"])
    # 將要 predict 的 column 移動到第一行
    indoor_data = adjust_col(indoor_data, "temp")
    # 篩選時間區段
    indoor_data = filter_time(indoor_data)
    print(indoor_data)

    #控制資料
    controller_data = get_data_period('controller', start_date, end_date)
    controller_data = pd.DataFrame(controller_data, columns=["Date", "I_O", "ac_temp", "fan", "mode"])
    # 篩選時間區段
    controller_data = filter_time(controller_data)
    # 沒有開啟冷氣機時的數值更改
    controller_data.loc[controller_data['I_O'] == 0, 'ac_temp'] = 0
    controller_data.loc[controller_data['I_O'] == 0, 'fan'] = 3
    controller_data.loc[controller_data['I_O'] == 0, 'mode'] = 3
    print(controller_data)

    #室外資料
    outdoor_data = get_data_period('outdoor', start_date, end_date)
    outdoor_data = pd.DataFrame(
        outdoor_data,
        columns=["Date",
                 "WDIR",
                 "WDSD",
                 "TEMP",
                 "HUMD",
                 "PRES",
                 "RAIN",
                 "HFX",
                 "HXD",
                 "HF",
                 "HD",
                 "HUVI",
                 "WEATHER"]
    )
    # 篩選時間區段
    outdoor_data = filter_time(outdoor_data)
    print(outdoor_data)

    #功耗資料
    #冰水機資料
    chiller_data = get_data_period('CS_nor', start_date, end_date)
    chiller_data = pd.DataFrame(
        chiller_data,
        columns=[
            "date",
            "time",
            "tM-TH8_AI.sensor0",
            "tM-TH8_AI.sensor1",
            "tM-TH8_AI.sensor2",
            "tM-TH8_AI.sensor3",
            "tM-TH8_AI.sensor4",
            "tM-TH8_AI.sensor5",
            "tM-TH8_AI.sensor6",
            "tM-TH8_AI.sensor7",
            "PM-3133_AI.V",
            "PM-3133_AI.I",
            "PM-3133_AI.PF",
            "PM-3133_AI.Kwh",
            "PM-3133_AI.Kvarh",
            "PM-3133_AI.Kvah"
        ]
    )
    chiller_data['time'] = pd.to_timedelta(chiller_data['time'])
    # 提取時間部分並以字串格式保存
    chiller_data['time'] = chiller_data['time'].dt.total_seconds(
    ).apply(lambda x: pd.to_datetime(x,
                                     unit='s').strftime('%H:%M:%S'))
    # 合併 date 和 time 列的數值
    chiller_data['Date'] = (chiller_data['date'].astype(str) + ' ' + chiller_data['time'].astype(str))
    # 刪除原來的 date 和 time 列
    chiller_data = chiller_data.drop(['date', 'time'], axis=1)
    # 將要 predict 的 column 移動到第一行
    chiller_data = adjust_col(chiller_data, "PM-3133_AI.Kwh")
    # 篩選時間區段
    chiller_data = filter_time(chiller_data)
    # 蒸發器差值
    chiller_data = make_features(chiller_data, 'tM-TH8_AI.sensor0', 'tM-TH8_AI.sensor5')
    # 壓縮機差值
    chiller_data = make_features(chiller_data, 'tM-TH8_AI.sensor1', 'tM-TH8_AI.sensor2')
    # 膨脹閥差值
    chiller_data = make_features(chiller_data, 'tM-TH8_AI.sensor6', 'tM-TH8_AI.sensor7')
    # 調整有功功率數值
    chiller_data = make_features(chiller_data, 'PM-3133_AI.Kwh', None)
    # 調整無功功率數值
    chiller_data = make_features(chiller_data, 'PM-3133_AI.Kvarh', None)
    # 調整視在功率數值
    chiller_data = make_features(chiller_data, 'PM-3133_AI.Kvah', None)
    print(chiller_data)
    if (model_name == 'Temp_Model'):
        # merge
        dataset = indoor_data.merge(outdoor_data, on='Date', how='outer')
        dataset = dataset.merge(controller_data, on='Date', how='outer')
        dataset = dataset.merge(chiller_data, on='Date', how='outer')
        dataset = seasonal_decomp(seq_len, dataset, 'temp')
    else:
        dataset = chiller_data.merge(indoor_data, on='Date', how='outer')
        dataset = dataset.merge(controller_data, on='Date', how='outer')
        dataset = dataset.merge(outdoor_data, on='Date', how='outer')
        dataset = seasonal_decomp(seq_len, dataset, 'PM-3133_AI.Kwh')
    dataset = dataset.fillna(method='ffill')
    # 第一天沒有開啟冷氣機時的數值更改
    dataset.loc[np.isnan(dataset['I_O']), 'ac_temp'] = 0
    dataset.loc[np.isnan(dataset['I_O']), 'fan'] = 3
    dataset.loc[np.isnan(dataset['I_O']), 'mode'] = 3
    dataset.loc[np.isnan(dataset['I_O']), 'I_O'] = 0
    print(dataset.isnull().any().any())
    print(dataset)
    return dataset


def split_data(dataset):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        dataset: A pandas DataFrame representing the dataset to be split.

    Returns:
        A pandas DataFrame containing the training set, validation set, and test set.

    """
    train = dataset[: int(len(dataset) * 0.8)]
    val = dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)]
    test = dataset[int(len(dataset) * 0.9): len(dataset)]

    return train, val, test


def find_max_min(data):
    """
    Finds the maximum and minimum values for each column of the given dataset.

    Args:
        data: A pandas DataFrame containing the dataset.

    Returns:
        An array containing the maximum and minimum values for each column.
        Each row of the array represents a column, with the first element
        being the maximum value and the second element being the minimum value.

    """
    max_vals = np.max(data, axis=0)
    min_vals = np.min(data, axis=0)
    max_min_vals = np.column_stack((max_vals, min_vals))

    return max_min_vals


def normalize_data(data, max_min_vals):
    """
    Normalizes the given data based on the provided maximum and minimum values.

    Args:
        data: A pandas DataFrame or numpy array containing the data to be normalized.
        max_min_vals: An array containing the maximum and minimum values for each column.

    Returns:
        A pandas DataFrame containing the normalized data, the maximum value of the target variable,
        and the minimum value of the target variable.

    """
    max_vals = max_min_vals[:, 0]
    min_vals = max_min_vals[:, 1]
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    normalized_data = normalized_data.fillna(0)
    return normalized_data.values, max_min_vals[0][0], max_min_vals[0][1]


def create_seq(input_size, seq_len, data):
    """
    Creates sequential data from the given input data.

    Args:
        input_size: The number of features to include from the input data.
        seq_len: The length of each sequential data sample.
        data: A pandas DataFrame representing the input data.

    Returns:
        A list of tuples, where each tuple contains a sequential data sample
        and its corresponding label.

    """
    target = data[:, 0]
    other = data[:, 1 :]
    final_seq = []
    for i in range(len(other) - seq_len):
        data_seq = []
        for j in range(i, i + seq_len):
            x = [target[j]] + other[j, : input_size - 1].tolist()
            data_seq.append(x)

        data_label = [target[i + seq_len]]
        final_seq.append((torch.FloatTensor(data_seq), torch.FloatTensor(data_label)))

    return final_seq


def batch_data(final_seq, batch_size):
    """
    Batches the sequence data using PyTorch DataLoader.

    Args:
        final_seq: A list of tuples containing sequence input data and target data.
        batch_size: The desired batch size.

    Returns:
        A DataLoader object containing the batched sequence data.

    """
    final_seq = DataLoader(dataset=final_seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    return final_seq


def preprocess_seq(args, model_name, start_date, end_date):
    """
    Calls all preprocessing functions to prepare the data.

    Args:
        args: Arguments.
        start_date:  data start date.
        end_date: data end date.

    Returns:
        processed sequence.
    """
    # Load the raw dataset
    dataset = preprocess_dataset(args.seq_len, model_name, start_date, end_date)

    # Split the dataset into training, validation, and test sets
    train, val, test = split_data(dataset)

    # Find the maximum and minimum values for each column
    max_min_array = find_max_min(train)

    # Normalize the training data
    train, train_max, train_min = normalize_data(train, max_min_array)

    # Create sequences based on the specified input size and sequence length
    final_seq = create_seq(args.input_size, args.seq_len, train)

    # Batch the training data
    Train_seq = batch_data(final_seq, args.batch_size)

    # Normalize and sequence the validation data
    val, _, _ = normalize_data(val, max_min_array)
    final_seq = create_seq(args.input_size, args.seq_len, val)
    Val_seq = batch_data(final_seq, args.batch_size)

    # Normalize and sequence the test data
    test, _, _ = normalize_data(test, max_min_array)
    final_seq = create_seq(args.input_size, args.seq_len, test)
    Test_seq = batch_data(final_seq, args.batch_size)

    return Train_seq, Val_seq, Test_seq, train_max, train_min


def Temp_Model(model_name, model_path, start_date, end_date):
    args = Model_args()
    Train_seq, Val_seq, Test_seq, train_max, train_min = preprocess_seq(args, model_name, start_date, end_date)
    # train(args, model_path, Train_seq, Val_seq)
    test(args, model_name, model_path, Test_seq, train_max, train_min)


def PD_Model(model_name, model_path, start_date, end_date):
    args = Model_args()
    Train_seq, Val_seq, Test_seq, train_max, train_min = preprocess_seq(args, model_name, start_date, end_date)
    # train(args, model_path, Train_seq, Val_seq)
    test(args, model_name, model_path, Test_seq, train_max, train_min)
