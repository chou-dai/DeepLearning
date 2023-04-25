# 深層強化学習の実装
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation, rc
from keras.models import Sequential
from keras.layers import Dense, ReLU
from keras.optimizers import RMSprop

optimizer = RMSprop()


# Brainクラス：エージェントの頭脳となるクラス
class Brain:
    def __init__(self, n_state, n_mid, n_action, gamma=0.9, r=0.99):
        self.eps = 1.0  # ε
        self.gamma = gamma  # 割引率
        self.r = r  # εの減衰率

        model = Sequential()
        model.add(Dense(n_mid, input_shape=(n_state,)))
        model.add(ReLU()) 
        model.add(Dense(n_mid))
        model.add(ReLU()) 
        model.add(Dense(n_action))
        model.compile(loss="mse", optimizer=optimizer)
        self.model = model

    def train(self, states, next_states, action, reward, terminal):
        q = self.model.predict(states)  
        next_q = self.model.predict(next_states)
        t = np.copy(q)
        if terminal:
            t[:, action] = reward  #  エピソード終了時の正解は、報酬のみ
        else:
            t[:, action] = reward + self.gamma*np.max(next_q, axis=1)
        self.model.train_on_batch(states, t)

    def get_action(self, states):
        q = self.model.predict(states)
        if np.random.rand() < self.eps:
            action = np.random.randint(q.shape[1], size=q.shape[0])
        else:
            action = np.argmax(q, axis=1)
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
        return action
    

# エージェントクラス
class Agent:
    def __init__(self, v_x, v_y_sigma, v_jump, brain):
        self.v_x = v_x  # x速度
        self.v_y_sigma = v_y_sigma  # y速度、初期値の標準偏差
        self.v_jump = v_jump  # ジャンプ速度
        self.brain = brain
        self.reset()

    def reset(self):
        self.x = -1  # 初期x座標
        self.y = 0  # 初期y座標
        self.v_y = self.v_y_sigma * np.random.randn()  # 初期y速度

    def step(self, g):  # 時間を1つ進める g:重力加速度
        states = np.array([[self.y, self.v_y]])
        self.x += self.v_x
        self.y += self.v_y

        reward = 0  # 報酬
        terminal = False  # 終了判定
        if self.x>1.0:
            reward = 1
            terminal = True
        elif self.y<-1.0 or self.y>1.0:
            reward = -1
            terminal = True
        reward = np.array([reward])

        action = self.brain.get_action(states)
        if action[0] == 0:
            self.v_y -= g   # 自由落下
        else:
            self.v_y = self.v_jump  # ジャンプ
        next_states = np.array([[self.y, self.v_y]])
        self.brain.train(states, next_states, action, reward, terminal)

        if terminal:
            self.reset()


# 環境クラス
class Environment:
    def __init__(self, agent, g):
        self.agent = agent
        self.g = g

    def step(self):
        self.agent.step(self.g)
        return (self.agent.x, self.agent.y)
    

# アニメーション
def animate(environment, interval, frames):
    fig, ax = plt.subplots()
    plt.close()
    ax.set_xlim(( -1, 1))
    ax.set_ylim((-1, 1))
    sc = ax.scatter([], [])

    def plot(data):
        x, y = environment.step()
        sc.set_offsets(np.array([[x, y]]))
        return (sc,)

    return animation.FuncAnimation(fig, plot, interval=interval, frames=frames, blit=True)


# Q学習の導入
n_state = 2
n_mid = 32
n_action = 2
brain = Brain(n_state, n_mid, n_action, r=0.99)  # εが減衰する

v_x = 0.05
v_y_sigma = 0.1
v_jump = 0.2
agent = Agent(v_x, v_y_sigma, v_jump, brain)

g = 0.2
environment = Environment(agent, g)

anim = animate(environment, 50, 1024)
rc('animation', html='jshtml')
anim