from time_series_model.utils import Temp_Model, PD_Model
from datetime import date
import torch

if __name__ == '__main__':
    # 指定時間區段
    start_date = date(2023, 6, 15)
    end_date = date(2023, 6, 30)

    #模型名稱
    model_name = 'Temp_Model'
    #儲存模型位置
    model_path = 'time_series_model/model/Temp_Model.pkl'
    # 訓練預測溫度模型
    final_seq = Temp_Model(model_name, model_path, start_date, end_date)

    #模型名稱
    model_name = 'PD_Model'
    #儲存模型位置
    model_path = 'time_series_model/model/PD_Model.pkl'
    # 訓練預測功耗模型
    final_seq = PD_Model(model_name, model_path, start_date, end_date)
