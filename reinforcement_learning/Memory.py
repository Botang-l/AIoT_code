from collections import namedtuple, deque
import torch
import copy
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        rb = batch_size // 100
        data = random.sample(self.memory, batch_size - rb)
        # print(len(data))
        # print(type(data))
        if rb:
            data += list(self.memory)[-rb :]
        # print(len(data))
        # print(type(data))

        return data

    def __len__(self):
        return len(self.memory)

class LSTMdata(object):
    def __init__(self, data):
        _, w, _ = data.shape
        self.__data = deque([], maxlen= w)
        self.__data_in_stock = deque([])
        self.init(data)

    def push(self, data):
        self.__data.append(data)
    
    def push_in_stock(self, data):
        self.__data_in_stock.append(data)

    def display(self):
        print(self.__data)

    def display_in_stock(self):
        print(self.__data_in_stock)   
    
    def init(self, data):
        l, w, _ = data.shape
        for i in range(w):
            self.push(data[0][i])
        for i in range(1, l):
            self.push_in_stock(data[i][-1])
    
    # 取得 RL 需要的當筆 data
    def load_RLdata(self):
        temp = self.__data[-1]
        print(temp)
        RLdata = torch.cat((temp[:2], temp[3:]))
        return(RLdata)
    
    # 將 RL 所生成的 data 放進 __data 變數
    def store_RLdata(self, RLdata):
       self.__data[-1][2] = RLdata

    # 取得 Time Series Model 需要的單筆 data
    def load_TSdata(self):
        print(list(self.__data))
        Tdata = torch.stack(list(self.__data))
        PDdata = copy.deepcopy(Tdata)
        PDdata[:, [0, 1]] = PDdata[:, [1, 0]]
        return(Tdata, PDdata)

    # 將 Time Series Model 所生成的 data 是下一個時間點的變化量。
    # 我們從 __data_in_stock 取得第一筆資料(即下一個時間點的變化量後)，並刪除 __data_in_stock該筆資料。
    # 我們將資料的第一筆和第二筆用模型預測出的溫度和工耗代替，然後放進 __data 變數。
    def store_TSdata(self, Tdata, PDdata):
        temp = self.__data_in_stock[0]
        temp[0] = Tdata
        temp[1] = PDdata
        self.push(temp)
        self.__data_in_stock.popleft()
        
