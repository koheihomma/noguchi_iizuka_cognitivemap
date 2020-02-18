import pickle
import sys
from matplotlib import pyplot as plt

args = sys.argv

f = open(args[1],"rb")
loss_list = pickle.load(f)

domain= [d for d in range(len(loss_list))]
plt.plot(domain, loss_list)
plt.show()
