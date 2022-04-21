import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from cnn_hu import HuEtAl

"""ハイパーパラメーター"""
SEED = 0
BATCH_SIZE = 32
PATHIENCE = 1000
# indian pinesは200，botswanaは145
CHANNELS = 200
# indian pinesは12，botswanaは14
N_CLASS = 12


def fix_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def main():
	fix_seed(SEED)
	
	"""test_setは自分で実装する"""
	test_set = 
	test_loader = torch.utils.data.DataLoader(
		dataset=test_set,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=2)

	device = 'cpu'
	net = net = HuEtAl(
		# 入力次元数
		input_channels=CHANNELS,
		# クラス数
		n_classes=N_CLASS
		)
	net = net.to(device)
	save_path = input('Enter your weights path!  ')
	net.load_state_dict(torch.load(save_path, map_location=device))
	
	total_label = []
	total_pred_label = []
	bar = tqdm(test_loader, leave=False)

	"""test"""
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(bar):
			bar.set_description('Testing batch ' + str(i+1))
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = net(inputs)
			_, predicted = torch.max(outputs.data, 1)

			total_label.extend(labels.cpu().numpy())
			total_pred_label.extend(predicted.cpu().numpy())

	total_label = np.stack(total_label).squeeze()
	total_pred_label = np.stack(total_pred_label).squeeze()

	cm = confusion_matrix(total_label, total_pred_label)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	classwise_acc = cm.diagonal()
	for i in range(classwise_acc.shape[0]):
		print(f'Class{i} Accuracy {classwise_acc[i]:.4f}')
	overall_acc = classwise_acc.sum()/classwise_acc.shape[0]
	print(f'{overall_acc:.4f}')

if __name__ == '__main__':
	main()
