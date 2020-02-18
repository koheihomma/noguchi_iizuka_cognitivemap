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

hdfpath="data_homma.h5"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
        self.Low = nn.GRUCell(vision_feature_size+motion_feature_size+hidden_size, hidden_size)
        self.High = nn.GRUCell(hidden_size, hidden_size)
        self.fc2_v = nn.Linear(vision_feature_size, vision_input_size)
        self.fc2_m = nn.Linear(motion_feature_size, motion_input_size)


    def forward(self, v_0, m_0, hidden1=None, hidden2=None):
        if hidden1 is None:
            hidden1 = torch.zeros(1, v_0.shape[1], self.hidden_size).to(device)
        if hidden2 is None:
            hidden2 = torch.zeros(1, v_0.shape[1], self.hidden_size).to(device)

        #PVM
        if v_0.shape==(50, 1, 768) and m_0.shape==(50, 1, 2):
            f_v = F.elu(self.fc1_v(v_0), alpha=1.0)
            f_m = F.elu(self.fc1_m(m_0), alpha=1.0)
            L_inputs = torch.cat((f_v, f_m), dim=2)
            time_len = L_inputs.shape[0]
            L_output = []

            for t in range(time_len):
                #L_inputs (50, 1, 256), hidden2 (1, 1, 256)
                L_input = torch.cat((L_inputs[t], hidden2[0]), dim=1)
                hidden1 = self.Low(L_input, hidden1[0])
                hidden1 = hidden1.view(1, hidden1.shape[0], hidden1.shape[1])
                hidden2 = self.High(hidden1[0], hidden2[0])
                hidden2 = hidden2.view(1, hidden2.shape[0], hidden2.shape[1])

                L_output.append(hidden1)

            L_output = torch.cat(L_output, dim=0)
            assert L_output.shape[2] == hidden1.shape[2]

            f_v_n = F.elu(L_output[:, :, :self.vision_feature_size])
            f_m_n = F.elu(L_output[:, :, self.vision_feature_size:self.vision_feature_size+self.motion_feature_size+1])
            v_n = torch.sigmoid(self.fc2_v(f_v_n))
            m_n = torch.tanh(self.fc2_m(f_m_n))
            #print("PVM")

            return v_n, m_n

		    #POM
        if v_0.shape==(1, 1, 768) and m_0.shape==(50, 1, 2):
            v_list = []
            m_list = []
            v_list.append(v_0)
            f_m = F.elu(self.fc1_m(m_0), alpha=1.0)
            time_len = m_0.shape[0]
            for i in range(time_len):
                f_v = F.elu(self.fc1_v(v_list[i]), alpha=1.0)
                L_input = torch.cat((f_v[0], f_m[i], hidden2[0]), dim=1)
                hidden1 = self.Low(L_input, hidden1[0])
                hidden1 = hidden1.view(1, hidden1.shape[0], hidden1.shape[1])
                hidden2 = self.High(hidden1[0], hidden2[0])
                hidden2 = hidden2.view(1, hidden2.shape[0], hidden2.shape[1])
                f_v_n = F.elu(hidden1[:, :, :self.vision_feature_size])
                f_m_n = F.elu(hidden1[:, :, self.vision_feature_size:self.vision_feature_size+self.motion_feature_size+1])
                v_n = torch.sigmoid(self.fc2_v(f_v_n))
                m_n = torch.tanh(self.fc2_m(f_m_n))
                v_list.append(v_n)
                m_list.append(m_n)

            v_n = torch.cat(v_list[1:], dim=0)
            m_n = torch.cat(m_list, dim=0)
            #print("POM")

            return v_n, m_n

		    #POV
        if v_0.shape==(50, 1, 768) and m_0.shape==(1, 1, 2):
            v_list = []
            m_list = []
            m_list.append(m_0)
            f_v = F.elu(self.fc1_v(v_0), alpha=1.0)
            time_len = v_0.shape[0]
            for i in range(time_len):
                f_m = F.elu(self.fc1_m(m_list[i]), alpha=1.0)
                L_input = torch.cat((f_v[i], f_m[0], hidden2[0]), dim=1)
                hidden1 = self.Low(L_input, hidden1[0])
                hidden1 = hidden1.view(1, hidden1.shape[0], hidden1.shape[1])
                hidden2 = self.High(hidden1[0], hidden2[0])
                hidden2 = hidden2.view(1, hidden2.shape[0], hidden2.shape[1])
                f_v_n = F.elu(hidden1[:, :, :self.vision_feature_size])
                f_m_n = F.elu(hidden1[:, :, self.vision_feature_size:self.vision_feature_size+self.motion_feature_size+1])
                v_n = torch.sigmoid(self.fc2_v(f_v_n))
                m_n = torch.tanh(self.fc2_m(f_m_n))
                m_list.append(m_n)
                v_list.append(v_n)

            v_n = torch.cat(v_list, dim=0)
            m_n = torch.cat(m_list[1:], dim=0)
            #print("POV")

            return v_n, m_n

		#test
        if v_0.shape==(1000, 1, 768) and m_0.shape==(1000, 1, 2):
            f_v = F.elu(self.fc1_v(v_0), alpha=1.0)
            f_m = F.elu(self.fc1_m(m_0), alpha=1.0)
            L_input = torch.cat((f_v, f_m), dim=2)
            time_len = L_input.shape[0]
            L_output = []

            for t in range(time_len):
                hidden = self.Low(L_input[t], hidden[0])
                hidden = hidden.view(1, hidden.shape[0], hidden.shape[1])
                L_output.append(hidden)

            L_output = torch.cat(L_output, dim=0)

            assert L_output.shape[2] == hidden.shape[2]

            f_v_n = F.elu(L_output[:, :, :self.vision_feature_size])
            f_m_n = F.elu(L_output[:, :, self.vision_feature_size:])
            v_n = torch.sigmoid(self.fc2_v(f_v_n))
            m_n = torch.tanh(self.fc2_m(f_m_n))

            return v_n, m_n



def MSE(x, y):
	a = 0.5*torch.sum((x - y)**2)
	return a

def cross_entropy_error(y, t):
	delta = 1e-7 # マイナス無限大を発生させないように微小な値を追加
	return - (1/768)* torch.sum(t * torch.log(y + delta)+(1- t) * torch.log(1 - y + delta))

import random

def shuffle_samples(X, y):
	order = np.arange(X.shape[0])
	np.random.shuffle(order)
	X_result = np.zeros(X.shape, dtype='float32')
	y_result = np.zeros(y.shape, dtype='float32')
	for i in range(X.shape[0]):
		X_result[i, ...] = X[order[i], ...]
		y_result[i, ...] = y[order[i], ...]
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
        for n in range(100):#100 epoch
            motion_lists=f['motion/train']
            vision_lists=f['vision/train']
            motion_lists, vision_lists=shuffle_samples(motion_lists, vision_lists)

            for motion_list, vision_list in zip(motion_lists, vision_lists):#100 data

                mini_loss_list=[]
                m_0=torch.from_numpy(motion_list[:50, :])
                v_0=torch.from_numpy(vision_list[:50, :])
                m_0, v_0 = Variable(m_0).to(device), Variable(v_0).to(device)
                m_t=torch.from_numpy(motion_list[1:51, :])
                v_t=torch.from_numpy(vision_list[1:51, :])
                m_t, v_t = Variable(m_t).to(device), Variable(v_t).to(device)

                vision_output1, motion_output1=model(v_0.view(50, 1, 768), m_0.view(50, 1, 2))

                for i in range(19):#1000 steps

                    m_0=torch.from_numpy(motion_list[(1+i)*50:(2+i)*50, :])
                    v_0=torch.from_numpy(vision_list[(1+i)*50:(2+i)*50, :])
                    m_0, v_0 = Variable(m_0).to(device), Variable(v_0).to(device)

                    m_t=torch.from_numpy(motion_list[(1+i)*50+1:(2+i)*50+1, :])
                    v_t=torch.from_numpy(vision_list[(1+i)*50+1:(2+i)*50+1, :])
                    m_t, v_t = Variable(m_t).to(device), Variable(v_t).to(device)

                    #POM
                    vision_output2, motion_output2=model(vision_output1[-1].view(1, 1, 768), m_0.view(50, 1, 2))
                    loss_vision2 = cross_entropy_error(vision_output2.view(50, 768), v_t.view(50, 768))
                    loss_motion2 = MSE(motion_output2.view(50, 2), m_t.view(50, 2))

                    #POV
                    vision_output3, motion_output3=model(v_0.view(50, 1, 768), motion_output1[-1].view(1, 1, 2))
                    loss_vision3 = cross_entropy_error(vision_output3.view(50, 768), v_t.view(50, 768))
                    loss_motion3 = MSE(motion_output3.view(50, 2), m_t.view(50, 2))

                    #PVM
                    vision_output1, motion_output1=model(v_0.view(50, 1, 768), m_0.view(50, 1, 2))
                    loss_vision1 = cross_entropy_error(vision_output1.view(50, 768), v_t.view(50, 768))
                    loss_motion1 = MSE(motion_output1.view(50, 2), m_t.view(50, 2))

                    #Loss
                    regularization_loss=0
                    for param in model.parameters():
                      regularization_loss += torch.sum(torch.abs(param))

                    loss = loss_vision1+loss_motion1+loss_vision2+loss_motion2+loss_vision3+loss_motion3 + 0.01 * regularization_loss
                    #loss = loss_vision1+loss_motion1+loss_vision2+loss_motion2+loss_vision3+loss_motion3

                    #BPTT
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    mini_loss_list.append(loss.to('cpu'))

            loss=sum(mini_loss_list)
            loss_list.append(loss.to('cpu'))
            epoch += 1
            print("epoch{0}：終了, loss:{1} \n".format(epoch, loss))

    torch.save(model.state_dict(), "gru_2L_crossmodal.pt")
    f = open('loss_2L_crossmodal.txt', 'wb')
    pickle.dump(loss_list, f)

# test
def test():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Model().to(device)
	model.eval()
	model.load_state_dict(torch.load('gru_2L_crossmodal.pt'))

	with h5py.File(hdfpath,'r') as f:
		motion_lists=f['motion/test']
		vision_lists=f['vision/test']
		motion_output=[]
		vision_output=[]
		for motion_list, vision_list in zip(motion_lists, vision_lists):
			#10 data
			m_0=torch.from_numpy(motion_list[:1000, :])
			v_0=torch.from_numpy(vision_list[:1000, :])
			m_0, v_0 = Variable(m_0).to(device), Variable(v_0).to(device)
			vision_pred, motion_pred = model(v_0.view(1000, 1, 768), m_0.view(1000, 1, 2))
			vision_pred = vision_pred.data[0].to('cpu')
			vision_output.append(vision_pred)
			motion_pred = motion_pred.data[0].to('cpu')
			motion_output.append(motion_pred)

		f_v = open('vision_output_2L_crossmodal.txt', 'wb')
		f_m = open('motion_output_2L_crossmodal.txt', 'wb')
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
