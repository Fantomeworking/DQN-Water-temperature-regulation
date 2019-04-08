import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import water

BATCH_SIZE=40
LR=0.01
EPSILON=0.9
GAMMA=0.9
TARGET_RELACE_ITER=100
MEMORY_CAPACITY=1000
N_ACTIONS=3#加、减冷水 and 加、减热水
N_STATES=4#冷水流量、温度 or 热水流量、温度） and 容器内水温、水量
class net(nn.Module):
    """docstring for net"""
    def __init__(self, ):
        super(net, self).__init__()
        self.fc1=nn.Linear(N_STATES,10)
        self.fc1.weight.data.normal_(0,0.1)
        self.out=nn.Linear(10,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        actions_value=self.out(x)
        return actions_value
class DQN(object):
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_hot_net,self.eval_cold_net,self.target_hot_net,self.target_cold_net=\
            net().cuda(),net().cuda(),net().cuda(),net().cuda()

        self.learn_step_counter=0#
        self.memory_counter=0#学习进度
        self.memory=np.zeros((MEMORY_CAPACITY,15))#记忆库
        self.optimizer_hot=torch.optim.Adam(self.eval_hot_net.parameters(),lr=LR)
        self.optimizer_cold = torch.optim.Adam(self.eval_cold_net.parameters(), lr=LR)
        self.loss_func=nn.MSELoss()
    def choose_action(self,x_hot,x_cold):
        x_hot=Variable(torch.unsqueeze(torch.FloatTensor(x_hot),0))
        x_cold = Variable(torch.unsqueeze(torch.FloatTensor(x_cold), 0))
        if np.random.uniform()<EPSILON:#选取最大概率的动作
            actions1value = self.eval_hot_net(x_hot.cuda())
            actions2value=self.eval_cold_net(x_cold.cuda())
            action_hot = torch.max(actions1value, 1)[1].data.cpu().numpy()[0]
            action_cold = torch.max(actions2value, 1)[1].data.cpu().numpy()[0]
        else:#任意选一个
            action_hot=np.random.choice([0,1,2],1)
            action_cold=np.random.choice([0,1,2],1)
        return action_hot,action_cold
    def store_transition(self,s,r,s_):
        transition=np.hstack((s,r,s_))
        #溢出后覆盖掉老的进度
        index=self.memory_counter%MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1
    def learn(self):
        #需不需要更新target.net
        if self.learn_step_counter%TARGET_RELACE_ITER==0:
            self.target_hot_net.load_state_dict(self.eval_hot_net.state_dict())
            self.target_cold_net.load_state_dict(self.eval_cold_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]
        b_s_hot = Variable(torch.FloatTensor(b_memory[:, [0,1,4,5]]))
        b_s_cold = Variable(torch.FloatTensor(b_memory[:,[2,3,4,5]]))
        b_a_hot = Variable(torch.LongTensor(b_memory[:, [6]].astype(int)))
        b_a_cold = Variable(torch.LongTensor(b_memory[:, [7]].astype(int)))
        b_r=Variable(torch.FloatTensor(b_memory[:,[8]]))
        b_s__hot = Variable(torch.FloatTensor(b_memory[:, [9, 10, 13, 14]]))
        b_s__cold = Variable(torch.FloatTensor(b_memory[:, [11, 12, 13, 14]]))
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval_hot=self.eval_hot_net(b_s_hot.cuda()).gather(1,b_a_hot.cuda())
        q_eval_cold = self.eval_cold_net(b_s_cold.cuda()).gather(1, b_a_cold.cuda())
        q_next_hot = self.target_hot_net(b_s__hot.cuda()).detach()  # 禁止反向传播
        q_next_cold = self.target_cold_net(b_s__cold.cuda()).detach()  # 禁止反向传播
        q_target_hot = b_r.cuda() + GAMMA * q_next_hot.max(1)[0]
        q_target_cold =b_r.cuda() + GAMMA * q_next_cold.max(1)[0]
        # 计算, 更新 eval
        loss_hot=self.loss_func(q_eval_hot,q_target_hot)
        self.optimizer_hot.zero_grad()
        loss_hot.backward()
        self.optimizer_hot.step()

        loss_cold = self.loss_func(q_eval_cold, q_target_cold)
        self.optimizer_cold.zero_grad()
        loss_cold.backward()
        self.optimizer_cold.step()
Dqn=DQN()
L=0
L1=200#目标水容量or 训练次数
k1=0#上一个训练的持续时间
K1=np.zeros([1,5])#上一个训练的效果
R1=100
R_=np.array([])#存储L1次循环中R值的变化
for i in range(L1):#使用训练次数循环
#while L<=L1:
    s_hot = np.array([100, 1/3, 0, 0])
    s_cold = np.array([20, 1/3, 0, 0])
    k = 0#当前个训练的持续时间
    K = np.zeros([1,5])#当前个训练的效果
    while True:
        a_hot,a_cold= Dqn.choose_action(s_hot,s_cold)
        s=np.hstack([s_hot[0:2],s_cold[0:2],s_hot[2:4],a_hot,a_cold])
        # 选动作, 得到环境反馈
        water_= water.step(s)
        s_, r, done= water_.water_change()
        # 存记忆s
        Dqn.store_transition(s, r, s_)
        if Dqn.memory_counter > MEMORY_CAPACITY:
            Dqn.learn() # 记忆库满了就进行学习
        k += 1
        K=np.append(K,[[s[-4],s[-3],r,s_hot[1],s_cold[1]]],axis=0)
        if done:    # 如果回合结束, 进入下回合
            #R=np.mean(np.sqrt((K[-40:, 0]-42)**2))
            R=-len([i for i in K[0:, 0] if 43 >=i>= 41])#以在41-43度区间内点数量最多的为最优结果
            #R = np.mean(K[-20:, 2])
            R_=np.append(R_,R)
            if R1>R:# 当前训练时间大于删一个训练时间,将长的保留下来
                K1,k1,R1,I=K,k,R,i
            break
        s_hot,s_cold = s_[[0,1,4,5]],s_[[2,3,4,5]]
        L=s[-3]
T_max = np.max(K1[0:, 0]).astype(int)
plt.figure()#最优训练结果温度变化曲线
plot1_0=plt.plot(K1[0:,1],K1[0:,0])#温度水量变化曲线
plot1_1 = plt.plot([0,K1[-1,1]],[42,42])#42度水温标准线

low=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if 41>i[0]>=0])
plot1_2=plt.plot(low[:,1],low[:,0],'.b',[low[0,1],low[0,1]],[0,T_max],'b',[low[-1,1],low[-1,1]],[0,T_max],'b')#温度水量变化曲线,低温

ok=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if 43>=i[0]>=41])
plot1_3=plt.plot(ok[:,1],ok[:,0],'.g',[ok[0,1],ok[0,1]],[0,T_max],'g',[ok[-1,1],ok[-1,1]],[0,T_max],'g')#温度水量变化曲线，合适

xList=np.hstack([low[[0,-1],1],ok[[0,-1],1]])
yList=np.hstack([low[[0,-1],2],ok[[0,-1],2]])

high=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if i[0]>43])
if len(high)>0:
    plot1_4=plt.plot(high[:,1],high[:,0],'.r',[high[0,1],high[0,1]],[0,T_max],'r',[high[-1,1],high[-1,1]],[0,T_max],'r')#温度水量变化曲线，高温
    xList = np.hstack([xList, high[[0, -1], 1]])
    yList = np.hstack([yList, high[[0, -1], 2]])

for s,[x, y] in enumerate(zip(xList, yList)):
    y=y*10//60+y*10%60/100
    y2=20.3+s*2.5
    plot1_5=plt.text(x,y2, '%.2f'%y, ha='center', va='bottom', fontsize=10.5)
plt.title('T_change')
plt.figure()#最优训练结果温R值变化曲线
plot3 = plt.plot(K1[0:,1],K1[0:,2])
plt.title('R')

plt.figure()#最优训练结果的冷热水流量变化曲线，红色：热水；蓝色：冷水
plot5 = plt.plot(K1[0:,1],K1[0:,3],'r')
plot6 = plt.plot(K1[0:,1],K1[0:,4],'b')
plt.title('water_hot and water_cold')

plt.figure()#训练次数内训练结果曲线
plot4 = plt.plot(range(L1),R_)
plt.title('R_change1-'+str(L1))

print('各温度的持续时间(分钟)：\n[0-41)°区间:',len(low)*10//60,'分',(len(low)*10%60),'秒',
      '\n[41-43]:',len(ok)*10//60,'分',(len(ok)*10%60),'秒',
      '\n(43,)：',len(high)*10//60,'分',(len(high)*10%60),'秒',
      '\n最高温度：',T_max,
      '\n水量：',K1[-1,1],
      '\n总时长：',K1.shape[0]*10//60,'分',K1.shape[0]*10%60,'秒',
      '\n第',I,'次时达到')
plt.show()

