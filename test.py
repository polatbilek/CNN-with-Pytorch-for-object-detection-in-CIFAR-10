import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
from net import Net

def load_net(fname, net): 
	import h5py
	h5f = h5py.File(fname, mode='r')
	for k, v in net.state_dict().items():
		param = torch.from_numpy(np.asarray(h5f[k]))
		v.copy_(param)


model_name = os.path.join(os.getcwd(), "ozan_acc_81.21_epoch_27.h5") # the location is "./model_name"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
net = Net(100, 150, 250, 500)
net.cuda()
load_net(model_name, net)

print("Testing Started...")
correct = 0
total = 0
for data in testloader:
	images, labels = data
	images = Variable(images.cuda())
	outputs = net(images)
	_, predicted = torch.max(outputs.data, 1)
	total += labels.size(0)
	correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
