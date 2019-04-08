import numpy as np
class step(object):
	def __init__(self,arg):
		self.T_hot = arg[0]#热水入口温度
		self.L_hot = arg[1]#热水口流量
		self.T_cold = arg[2]#冷水入口温度
		self.L_cold = arg[3]#冷水口流量
		self.T_now = arg[4]#容器水温
		self.L_now = arg[5]#容器水量
		self.A_1 = arg[6]#动作
		self.A_2 = arg[7]
	def water_change(self):
		self.L_hot,self.L_cold=Action(self.A_1,self.A_2,self.L_hot , self.L_cold)
		L=self.L_hot + self.L_cold
		self.L_now=150*(L>150)+(self.L_now+L)*(L<=150)#容量变化,最高为150
		Water_out=(self.L_now-150)*(self.L_now>150)#溢出水量
		T_out=T_change(self.T_hot,self.T_cold,self.L_hot,self.L_cold)
		self.T_now=T_change(T_out,self.T_now,self.L_hot+self.L_cold,self.L_now)-0.2

		r1=np.sqrt(((self.T_now-42))**2)#r越接近42度越低
		r2=np.sqrt(Water_out)#溢出水量越少r越小
		r3=np.sqrt(((self.L_now/10-15))**2)#水量未满前越少r越大
		r4=4/L#水流量越大越好
		#按每分钟9升 热水上限为5/6，冷水上限为2
		#(1 > self.L_hot and self.L_hot > 0) and (3 / 2 > self.L_cold and self.L_cold > 0)
		if  Water_out<10 and self.T_now<80 :
			done=False
		else:
			done=True
		s_=np.array([self.T_hot,self.L_hot,self.T_cold,self.L_cold,self.T_now,self.L_now])
		return s_,(r1+r2+r4)/2,done
def T_change(t1,t2,l1,l2):
	rou1 = -3.84e-06 * t1 ** 2 - 4.791e-05 * t1 + 1 # t1温度下水密度
	rou2 = -3.84e-06 * t2 ** 2 - 4.791e-05 * t2 + 1 # t2温度下水密度
	M_1,M_2=rou1*l1,rou2*l2							#水的质量
	if (M_1 + M_2)==0:
		print(M_1 + M_2,l1,l2)
	T=(M_1*t1  + M_2*t2) / (M_1 + M_2)				#水的混合温度
	return T
def Action(A_hot,A_cold,L_hot,L_cold):
	#0->加水，1->减水
	L_times=1/36#每次变化水量
	L_hot += L_times * (A_hot == 0 )
	L_hot -= L_times * (A_hot == 1 )
	L_cold += L_times * (A_cold == 0)
	L_cold -= L_times * (A_cold == 1)
	# 为水量做限制
	L_hot = 1 * (1 < L_hot) + (L_hot < 0)+L_hot*(1>=L_hot >= 0)
	L_cold = 3/2 * (3/2 < L_cold) +(L_cold < 0)+L_cold*(3/2>=L_cold >= 0)
	return L_hot,L_cold
if __name__ == '__main__':
	s1 = np.array([100, 0, 20, 0, 0, 0, 0, 2])
	s=step(s1)
	s_, r, done=s.water_change()
	print(s_,r,done)