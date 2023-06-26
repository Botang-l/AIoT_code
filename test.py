from time_series_model.utils import Temp_Model, PD_Model, TimeSeriesData
from datetime import date
import torch

if __name__ == '__main__':
    # 指定時間區段
    start_date = date(2023, 6, 20)
    end_date = date(2023, 6, 20)

    
    #模型名稱
    model_name = 'Temp_Model'
    #儲存模型位置
    model_path = 'time_series_model/model/Temp_Model.pkl'
    #序列資料
    Temp_data = TimeSeriesData(model_name, start_date, end_date)
    # 訓練預測溫度模型
    Temp_Model(model_name, model_path, Temp_data, start_date, end_date)

    #模型名稱
    model_name = 'PD_Model'
    #儲存模型位置
    model_path = 'time_series_model/model/PD_Model.pkl'
    #序列資料
    PD_data = TimeSeriesData(model_name, start_date, end_date)
    # 訓練預測功耗模型
    PD_Model(model_name, model_path, PD_data, start_date, end_date)
    firstData = PD_data.displayFirstData()
    print('-'*20)
    print(firstData)
    print('-'*20)
    print(firstData[-1])
    print('-'*20)
    # path = 'time_series_model/model/PD_Model.pkl'
    # loaded_model = torch.load(path)
    # model = loaded_model['model']
    # model.eval()