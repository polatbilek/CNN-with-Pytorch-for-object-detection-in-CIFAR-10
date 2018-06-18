import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

	def __init__(self, conv1_dim=100, conv2_dim=150, conv3_dim=250, conv4_dim=500):
		super(Net, self).__init__()
		self.conv4_dim = conv4_dim

		self.conv1 = nn.Conv2d(3, conv1_dim, 5, stride=1, padding=2)
		self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, 3, stride=1, padding=2)
		self.conv3 = nn.Conv2d(conv2_dim, conv3_dim, 3, stride=1, padding=2)
		self.conv4 = nn.Conv2d(conv3_dim, conv4_dim, 3, stride=1, padding=2)

		self.pool = nn.MaxPool2d(2, 2)

		self.fc1 = nn.Linear(conv4_dim * 3 * 3, 270)
		self.fc2 = nn.Linear(270, 150)
		self.fc3 = nn.Linear(150, 10)

		self.normalize1 = nn.BatchNorm2d(conv1_dim)
		self.normalize2 = nn.BatchNorm2d(conv2_dim)
		self.normalize3 = nn.BatchNorm2d(conv3_dim)
		self.normalize4 = nn.BatchNorm2d(conv4_dim)

	def forward(self, x):
		x = self.pool(F.relu(self.normalize1((self.conv1(x)))))
		x = self.pool(F.relu(self.normalize2((self.conv2(x)))))
		x = self.pool(F.relu(self.normalize3((self.conv3(x)))))
		x = self.pool(F.relu(self.normalize4((self.conv4(x)))))

		x = x.view(-1, self.conv4_dim * 3 * 3)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x