# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from net import Net

def save_net(fname, net):
	import h5py
	h5f = h5py.File(fname, mode='w')
	for k, v in net.state_dict().items():
		h5f.create_dataset(k, data=v.cpu().numpy())


def weights_normal_init(model, dev=0.01):
	if isinstance(model, list):
		for m in model:
			weights_normal_init(m, dev)
	else:
		for m in model.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0.0, dev)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, dev)


num_of_epoch = 30
learning_rate = 0.001
print_every = 100
batch_size = 100
training_test_size = 10000

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

saving_mode = False

net = Net(conv1_dim=100, conv2_dim=150, conv3_dim=250, conv4_dim=500).cuda()
weights_normal_init(net, dev=0.01)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

print("Training Started...")
for epoch in range(num_of_epoch):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        correct = 0
        total = 0

        optimizer.zero_grad()
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data
        if i % print_every == (print_every-1):    
            print('[' + str(epoch + 1) + ', ' + str(i + 1) + '], ' + str((running_loss.cpu().numpy() / print_every)[0]))
            correct = 0
            total = 0
            index = 0
            for data in testloader:
                images, labels = data
                images = Variable(images.cuda())
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum()
                if index % training_test_size == training_test_size-1:
                    break
                index += 1
            accuracy = float(100*correct)/total
            print('Training Accuracy of ' + str(training_test_size) + ' images: %d %%' % (100 * correct / total))
            if accuracy >= 80 and saving_mode:
                network.save_net('ozan_acc_'+str(accuracy)+'_epoch_'+str(epoch)+'.h5', net)
            running_loss = 0.0

print("Finished Training")



