# 載入相關套件
import random

class ActionSpace():
    
    def __init__(self, num): # 初始化       
        self.n = num # 行動空間數量(Action Space)
    
    def sample(self):       
        return  random.randint(0, self.n-1) # 
    

s = 1
f = 15
b = (s + f) // 2



# 環境類別
class Environment:    
    def __init__(self, LSTM, num=2): # 初始化       
        self.__state = [b, 0] # 玩家一開始站中間位置
        self.__totalReward = 0 
        self.action_space = ActionSpace(num)
        self.actionlist = []
    def get_observation(self):
        # 狀態空間(State Space)，共有5個位置
        return [i for i in range(s, f+1)]

    def is_done(self): # 判斷比賽回合是否結束
        # 是否走到左右端點
        return self.__state[0] == s or self.__state[0] == f

    # def step(self, action):
    #     self.actionlist.append(action)
        
    # 步驟
    def step(self, action):
        # 是否回合已結束    
        self.actionlist.append(action)
        
        self.__state[0] = self.__state[0] + 1 if (action) else self.__state[0] - 1
        if self.__state[0] == s:
            reward = -2
        elif self.__state[0] == f:
            reward = 2
        elif self.__state[0] == f-2:
            reward = 0.1
        else:    
            if(action):
                reward = -0.01
            else:
                reward = -0.01
        self.__totalReward += reward
        # print(f"累計報酬: {self.totalReward:.4f}")
        # print('現在位置', self.state[0])
        # print('-'*20)
        self.__state[1] += 1
        return self.__state, reward, self.is_done(), False, 0

    def reset(self):
        self.__state = [b, 0] # 玩家一開始站中間位置
        self.__totalReward = 0
        self.actionlist = []
        
        return(self.__state, None)
    
    def TotalReward(self):
        return(self.__totalReward)

    def displayPosition(self):
        print('現在位置', self.__state[0])
        #return(self.__state[0])
    
    def displayTotalReward(self):
        print('Total Reward :', self.__totalReward)
      

