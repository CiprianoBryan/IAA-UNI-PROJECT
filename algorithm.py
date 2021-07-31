import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

class CNNModel(nn.Module):
	def __init__(self, config_model, config_training):
		super().__init__()
		self.device=config_training['device']
		
		self.conv1 = nn.Conv2d(config_model['conv1']['in_channel'], config_model['conv1']['out_channel'], config_model['conv1']['kernel_size'], config_model['conv1']['stride'])
		self.conv2 = nn.Conv2d(config_model['conv2']['in_channel'], config_model['conv2']['out_channel'], config_model['conv2']['kernel_size'], config_model['conv2']['stride'])
		self.pool = nn.MaxPool2d(config_model['pool']['stride'], config_model['pool']['stride'])
		self.linear1 = nn.Linear(config_model['linear1']['in_features'], config_model['linear1']['out_features'])

		self.ReLU = nn.ReLU()
		# self.dropout = nn.Dropout(config_model['dropout'])
		self.coste = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=config_training['learning_rate'], momentum=config_training['momentum'])

		self.to(config_training['device'])

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = self.linear1(x)
		# x = F.log_softmax(x, dim=1)
		return x