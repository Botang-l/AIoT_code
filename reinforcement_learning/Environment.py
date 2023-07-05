# 載入相關套件
import random
import torch
from .Memory import ACData
import json


class ActionSpace():
    def __init__(self, num):    # 初始化
        self.n = num    # 行動空間數量(Action Space)

    def sample(self):
        return random.randint(0, self.n - 1)    #


s = 1
f = 15
b = (s + f) // 2

ACTION = {0: 32, 1: 20, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26, 8: 27}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 環境類別
class Environment:
    def __init__(self):    # 初始化

        self.__ori_data = torch.load('/home/remote_user/sharefile/AIoT_code/time_series_model/data/dataset.pt')
        self.__data = ACData(self.__ori_data)
        self.__state = self.__data.load_RLdata()
        self.__totalPD = 0
        # ACTION => [20, 21, 22, 23, 24, 25, 26, 27, 'Pending']
        self.action_space = ActionSpace(9)
        self.actionlist = []
        self.__action_times = [0] * 9

        path = '/home/remote_user/sharefile/AIoT_code/time_series_model/model/PD_Model.pkl'
        loaded_model = torch.load(path)
        self.__PD_model = loaded_model['model'].to(device)
        self.__PD_model.eval()

        path = '/home/remote_user/sharefile/AIoT_code/time_series_model/model/Temp_Model.pkl'
        loaded_model = torch.load(path)
        self.__Temp_model = loaded_model['model'].to(device)
        self.__Temp_model.eval()

        with open('/home/remote_user/sharefile/AIoT_code/time_series_model/data/MaxMinData.json', 'r') as file:
            data = json.load(file)
        self.__Tmax = data['Tmax']
        self.__Tmin = data['Tmin']
        self.__PDmax = data['PDmax']
        self.__PDmin = data['PDmin']
        self.__ACCmax = data['ACCmax']
        self.__ACCmin = data['ACCmin']
        # test
        self.Tt = 0
        self.PDt = 0
        self.tt = 0

    def get_observation(self):
        # 狀態空間(State Space)，共有5個位置
        return [i for i in range(s, f + 1)]

    def is_done(self):    # 判斷比賽回合是否結束
        return self.__data.isEmpty()

    ##############################################
    # 存在 memory 裡面的內容都必須先 normalization #
    ##############################################
    def step(self, action):
        action = 0
        self.__action_times[action] += 1
        #print('ACTION :', ACTION[action], " /  NUMBER :", action, self.ACC_normalize(ACTION[action]))
        # 將 RL model 的 action 存到 memory
        self.__data.store_RLdata(self.ACC_normalize(ACTION[action]))
        # 將 RL model 的 action 加到該回合決策集
        self.actionlist.append(ACTION[action])

        # 新的 RL model 資料進來後，要將它放到 lstm 去預測該 action 的功耗與溫度變化
        # Tout 和 Pout 分別是選擇特定 action 後，後續的功耗和溫度變化的結果
        Tdata, PDdata = self.__data.load_TSdata()
        Tout = self.__Temp_model(Tdata.to(device)).item()
        #PDout = self.PDadapter(action, PDdata)
        PDout = self.__PD_model(PDdata.to(device)).item()
        print('32度: 溫度', self.temp_denormalize(Tout), '功耗', self.PD_denormalize(PDout))
        Tdata[0][-1][2] = 0
        PDdata[0][-1][2] = 0
        Tout1 = self.__Temp_model(Tdata.to(device)).item()
        #PDout1 = self.PDadapter(action, PDdata)
        PDout1 = self.__PD_model(PDdata.to(device)).item()
        print('20度: 溫度', self.temp_denormalize(Tout1), '功耗', self.PD_denormalize(PDout1))
        print(True if (Tout >= Tout1) else False, True if (PDout <= PDout1) else False)
        self.tt += 1
        if (Tout >= Tout1):
            self.Tt += 1
        if (PDout <= PDout1):
            self.PDt += 1

        # 將結果存回 memory
        self.__data.store_TSdata(Tout, PDout)

        # 將資料回復成原始資料狀態(denormalize)
        Tout = self.temp_denormalize(Tout)
        PDout = self.PD_denormalize(PDout)

        # 將 reward 加進 total_reward
        self.__totalPD += PDout

        # 定義 reward 為預測功耗的負數 (功耗愈高，reward愈低)
        reward = -((PDout * 100)**2)

        #print('新溫度 :', Tout, '/ 新功耗 :', PDout, '/ 總功耗 :', self.__totalReward)

        self.__state = self.__data.load_RLdata()
        del Tout, PDout
        torch.cuda.empty_cache()
        return self.__state, reward, self.is_done(), False, 0

    def reset(self):
        print(self.__action_times)
        #if self.__totalReward > 0:
        #print(self.actionlist)
        self.__data = ACData(self.__ori_data)
        self.__state = self.__data.load_RLdata()
        self.__totalPD = 0
        self.actionlist = []
        self.__action_times = [0] * 9
        return (self.__state, None)

    def totalReward(self):
        return (self.__totalPD)

    def displayPosition(self):
        print('現在位置', self.__state[0])
        #return(self.__state[0])

    def displayTotalReward(self):
        print('Total Reward :', self.__totalPD)

    def temp_normalize(self, T):
        return ((T - self.__Tmin) / (self.__Tmax - self.__Tmin))

    def PD_normalize(self, PD):
        return ((PD - self.__PDmin) / (self.__PDmax - self.__PDmin))

    def ACC_normalize(self, ACC):
        return ((ACC - self.__ACCmin) / (self.__ACCmax - self.__ACCmin))

    def temp_denormalize(self, T):
        return ((self.__Tmax - self.__Tmin) * T + self.__Tmin)

    def PD_denormalize(self, PD):
        return ((self.__PDmax - self.__PDmin) * PD + self.__PDmin)

    def ACC_denormalize(self, ACC):
        return ((self.__ACCmax - self.__ACCmin) * ACC + self.__ACCmin)

    def PDadapter(self, action, PDdata):
        action = action
        if action == 0:
            return (0)
        times = 7 - action
        PDdata[0][-1][0] = 27
        PDout = self.__PD_model(PDdata.to(device)).item()
        for _ in range(times):
            PDout = PDout * (0.92)
        return (PDout)
