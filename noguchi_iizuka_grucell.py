import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from matplotlib import pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import h5py
from torch.autograd import detect_anomaly
from tqdm import tqdm
import argparse
import pickle
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		vision_input_size=768
		vision_feature_size=128
		motion_input_size=2
		motion_feature_size=128
		hidden_size = 256
		self.hidden_size = hidden_size
		self.vision_feature_size=vision_feature_size
		self.motion_feature_size = motion_feature_size
		self.fc1_v = nn.Linear(vision_input_size, vision_feature_size)
		self.fc1_m = nn.Linear(motion_input_size, motion_feature_size)
		self.Low = nn.GRUCell(vision_feature_size+motion_feature_size + hidden_size, hidden_size)
		self.High = nn.GRUCell(hidden_size, hidden_size)
		self.fc2_v = nn.Linear(vision_feature_size, vision_input_size)
		self.fc2_m = nn.Linear(motion_feature_size, motion_input_size)

	def forward(self, v_0, m_0, hiddens):
		f_v = F.elu(self.fc1_v(v_0), alpha=1.0)
		f_m = F.elu(self.fc1_m(m_0), alpha=1.0)
		all_input = torch.cat((f_v, f_m), dim=1)
		L_input = torch.cat((all_input, hiddens[1]), dim=1)
		output_L = self.Low(L_input, hiddens[0])
		output_H = self.High(hiddens[0], hiddens[1])
		f_v_n = F.elu(output_L[:, :self.vision_feature_size])
		f_m_n = F.elu(output_L[:, self.vision_feature_size:self.vision_feature_size+self.motion_feature_size])
		v_n = torch.sigmoid(self.fc2_v(f_v_n))
		m_n = torch.tanh(self.fc2_m(f_m_n))

		return v_n, m_n, [output_L, output_H]

	def initHidden(self):
		return [torch.zeros(50, self.hidden_size).to(device), torch.zeros(50, self.hidden_size).to(device)]


hdfpath="data.h5"

def MSE(x, y):
	a = 0.5*torch.sum((x - y)**2)
	return a

def cross_entropy_error(y, t):
	delta = 1e-7 # マイナス無限大を発生させないように微小な値を追加
	return - (1/768)* torch.sum(t * torch.log(y + delta)+(1- t) * torch.log(1 - y + delta))

def shuffle_samples(X, y):
	order = np.arange(X.shape[0])
	np.random.shuffle(order)
	X_result = np.zeros(X.shape, dtype='float32')
	y_result = np.zeros(y.shape, dtype='float32')
	for i in range(X.shape[0]):
		X_result[i, ...] = X[order[i], ...]
		y_result[i, ...] = y[order[i], ...]
	print('shuffled')
	return X_result, y_result

def train():
	#GPU
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	#model
	model = Model().to(device)
	optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
	model.train()

	with h5py.File(hdfpath,'r') as f:
		epoch=0
		loss_list = []
		for n in range(100):
			motion_lists=f['motion/train']
			vision_lists=f['vision/train']
			motion_lists, vision_lists = shuffle_samples(motion_lists, vision_lists)

			#vision_output_list = []
			#motion_output_list = []
			#hiddens1_list = []

			for motion_list, vision_list in tqdm(zip(motion_lists, vision_lists)):
      			#100 data
				mini_loss_list=[]
				m_0 = torch.from_numpy(motion_list[:50, :])
				v_0 = torch.from_numpy(vision_list[:50, :])
				m_0, v_0 = Variable(m_0, requires_grad=True).to(device), Variable(v_0, requires_grad=True).to(device)
				m_t = torch.from_numpy(motion_list[1:51, :])
				v_t = torch.from_numpy(vision_list[1:51, :])
				m_t, v_t = Variable(m_t, requires_grad=True).to(device), Variable(v_t, requires_grad=True).to(device)

				hiddens=model.initHidden()

				vision_output, motion_output, hiddens = model(v_0.view(50, 768), m_0.view(50, 2), hiddens)

				#vision_output_list.append(vision_output)
				#motion_output_list.append(motion_output)

				for i in range(18): #50stepsずつ入力
					m_0 = torch.from_numpy(motion_list[(i+1)*50:(i+2)*50, :])
					v_0 = torch.from_numpy(vision_list[(i+1)*50:(i+2)*50, :])
					m_0, v_0 = Variable(m_0).to(device), Variable(v_0).to(device)
					m_t = torch.from_numpy(motion_list[(i+1)*50+1:(i+2)*50+1, :])
					v_t = torch.from_numpy(vision_list[(i+1)*50+1:(i+2)*50+1, :])
					m_t, v_t = Variable(m_t).to(device), Variable(v_t).to(device)

					vision_output, motion_output, hiddens=model(v_0.view(50, 768), m_0.view(50, 2), hiddens)
					#vision_output_list.append(vision_output)
					#motion_output_list.append(motion_output)

					loss_vision1 = cross_entropy_error(vision_output, v_t)
					loss_motion1 = MSE(motion_output, m_t)

					"""
					vision_output2, motion_output2, hiddens=model(vision_output_list[i+1].data[0].view(50, 768), m_0.view(50, 2), hiddens)
					loss_vision2 = cross_entropy_error(vision_output2, v_t)
					loss_motion2 = MSE(motion_output2, m_t)

					vision_output3, motion_output3, hiddens=model(v_0.view(50, 768), motion_output_list[i+1].data[0].view(50, 2), hiddens)
					loss_vision3 = cross_entropy_error(vision_output3, v_t)
					loss_motion3 = MSE(motion_output3, m_t)
					"""
					regularization_loss = 0
					#for param in model.parameters():
					#	regularization_loss += torch.sum(torch.abs(param))
					#loss = loss_vision1+loss_motion1+loss_vision2+loss_motion2+loss_vision3+loss_motion3+0.01*regularization_loss
					loss = loss_vision1+loss_motion1
					optimizer.zero_grad()
					loss.backward(retain_graph=True)
					optimizer.step()
					mini_loss_list.append(loss.to('cpu'))

			loss=sum(mini_loss_list)
			loss_list.append(loss.to('cpu'))
			epoch += 1
			print("epoch{0}：終了, loss:{1} \n".format(epoch, loss))

	torch.save(model.state_dict(), "grucell.pt")
	f = open('list_grucell.txt', 'wb')
	pickle.dump(loss_list, f)

# test
def test():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Model().to(device)
	model.eval()
	model.load_state_dict(torch.load('gru.pt'))

	with h5py.File(hdfpath,'r') as f:
		motion_lists=f['motion/test']
		vision_lists=f['vision/test']
		motion_output=[]
		vision_output=[]
		for motion_list, vision_list in tqdm(zip(motion_lists, vision_lists)):
			#10 data
			m_0=torch.from_numpy(motion_list[:1000, :])
			v_0=torch.from_numpy(vision_list[:1000, :])
			m_0, v_0 = Variable(m_0).to(device), Variable(v_0).to(device)
			vision_pred, motion_pred = model(v_0.view(1, 1000, 768), m_0.view(1, 1000, 2))
			vision_pred = vision_pred.data[0].to('cpu')
			vision_output.append(vision_pred)
			motion_pred = motion_pred.data[0].to('cpu')
			print(motion_pred)
			motion_output.append(motion_pred)

		f_v = open('vision_output.txt', 'wb')
		f_m = open('motion_output.txt', 'wb')
		pickle.dump(vision_output, f_v)
		pickle.dump(motion_output, f_m)

def arg_parse():
	parser = argparse.ArgumentParser(description='RNN implemented with Pytorch')
	parser.add_argument('--train', dest='train', action='store_true')
	parser.add_argument('--test', dest='test', action='store_true')
	args = parser.parse_args()
	return args

#main
if __name__ == '__main__':
	args = arg_parse()

	if args.train:
		train()
	elif args.test:
		test()
	else:
		print("please select train or test flag")
		print("train: python main.py --train")
		print("test:  python main.py --test")
		print("both:  python main.py --train --test")
