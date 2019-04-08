import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy
import matplotlib.pyplot as plt
import water


BATCH_SIZE=100
LR=0.01
EPSILON=0.9
GAMMA=0.9
TARGET_RELACE_ITER=100
MEMORY_CAPACITY=1000
N_ACTIONS=4#加、减冷水 or 加、减热水
N_STATES=6#冷、热水流量、温度 and 容器内水温、水量

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
        actions1value=self.out(x)
        actions2value = self.out(x)
        return actions1value,actions2value
class DQN(object):
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net,self.target_net=net().cuda(),net().cuda()

        self.learn_step_counter=0#
        self.memory_counter=0#学习进度
        self.memory=np.zeros((MEMORY_CAPACITY,N_STATES*2+3))#记忆库
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()

    def choose_action(self,x):
        x=Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        if np.random.uniform()<EPSILON:#选取最大概率的动作
            actions1value, actions2value=self.eval_net(x.cuda())
            action1 = torch.max(actions1value, 1)[1].data.cpu().numpy()[0]  # return the argmax
            action2 = torch.max(actions2value, 1)[1].data.cpu().numpy()[0]
            action=np.array([action1,action2])
        else:#任意选一个
            action=np.random.choice([0,1,2,3],2)
        return action
    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,a,[r],s_))
        #溢出后覆盖掉老的进度
        index=self.memory_counter%MEMORY_CAPACITY

        self.memory[index,:]=transition
        self.memory_counter+=1
    def learn(self):
        #需不需要更新target.net
        if self.learn_step_counter%TARGET_RELACE_ITER==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]
        b_s=Variable(torch.FloatTensor(b_memory[:,:N_STATES]))
        b_a1=Variable(torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int)))
        b_a2 = Variable(torch.LongTensor(b_memory[:, N_STATES+1:N_STATES + 2].astype(int)))
        b_r=Variable(torch.FloatTensor(b_memory[:,N_STATES+2:N_STATES+3]))
        b_s_=Variable(torch.FloatTensor(b_memory[:,N_STATES+3:]))

        q1eval, q2eval=self.eval_net(b_s.cuda())
        q1eval, q2eval=q1eval.gather(1,b_a1.cuda()),q2eval.gather(1,b_a2.cuda())
        q1next, q2next=self.target_net(b_s_.cuda())[0].detach(),self.target_net(b_s_.cuda())[1].detach()#禁止反向传播
        q1target = b_r.cuda() + GAMMA * q1next.max(1)[0]
        q2target = b_r.cuda() + GAMMA * q2next.max(1)[0]
        q_eval, q_target=(q1eval+q2eval)/2,(q1target+q1target)/2
        loss=self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

Dqn=DQN()
Net=net()
#print(Net)
L=0
L1=200#目标水容量or 训练次数
k1=0#上一个训练的持续时间
K1=np.zeros([1,5])#上一个训练的效果
R1=100
R_=np.array([])
for i in range(L1):#使用训练次数循环
# while L<=L1:
    s=np.array([100, 1/3, 20, 1/3, 0, 0])
    k = 0#当前个训练的持续时间
    K = np.zeros([1,5])#当前个训练的效果
    while True:
        a = Dqn.choose_action(s)

        a1=numpy.append(s,a)
        # 选动作, 得到环境反馈
        water_= water.step(a1)
        s_, r, done= water_.water_change()
        # 存记忆s
        Dqn.store_transition(s, a, r, s_)
        if Dqn.memory_counter > MEMORY_CAPACITY:
            Dqn.learn() # 记忆库满了就进行学习
        k += 1
        K=numpy.append(K,[[s[-2],s[-1],r,s[1],s[3]]],axis=0)
        s = s_
        L = s[-1]
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

T_max=np.max(K1[0:,0]).astype(int)
plt.figure()#最优训练结果温度变化曲线
plot1_0=plt.plot(K1[0:,1],K1[0:,0])#温度水量变化曲线
plot1_1 = plt.plot([0,K1[-1,1]],[42,42])#42度水温标准线

low=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if 41>i[0]>=0])
plot1_2=plt.plot(low[:,1],low[:,0],'.b',low[[0,-1],1],[[20,20],[20,20]],'.b')#温度水量变化曲线,低温

ok=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if 43>=i[0]>=41])
plot1_3=plt.plot(ok[:,1],ok[:,0],'.g',ok[[0,-1],1],[[20,20],[20,20]],'.g')#温度水量变化曲线，合适

xList=np.hstack([low[[0,-1],1],ok[[0,-1],1]])
yList=np.hstack([low[[0,-1],2],ok[[0,-1],2]])

high=np.array([np.append(i,s) for s,i in enumerate(K1[0:,[0,1]]) if i[0]>43])
if len(high)>0:
    plot1_4=plt.plot(high[:,1],high[:,0],'.r',high[[0,-1],1],[[20,20],[20,20]],'.r')#温度水量变化曲线，高温
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
      '\n水量：',K1[-1,1],
      '\n最高温度：',T_max,
      '\n总时长：',K1.shape[0]*10//60,'分',K1.shape[0]*10%60,'秒',
      '\n第',I,'次时达到')
plt.show()