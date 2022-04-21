import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from tqdm import tqdm

from cnn_hu import HuEtAl


"""ハイパーパラメーター"""
SEED = 0
GPU = 'cuda:0'
EPOCH = 5000
BATCH_SIZE = 32
EA_FLAG = True
PATHIENCE = 1000
# indian pinesは200，botswanaは145
CHANNELS = 200
# indian pinesは12，botswanaは14
N_CLASS = 12
LR = 0.01


def worker_init_fn(worker_id):
	# copied from https://stackoverflow.com/questions/67196075/
	torch_seed = torch.initial_seed()
	random.seed(torch_seed + worker_id)
	if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
		torch_seed = torch_seed % 2**30
	np.random.seed(torch_seed + worker_id)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
	fix_seed(SEED)

	device = torch.device(GPU if torch.cuda.is_available() else 'cpu')
	
	"""train_set, valid_setは自分で実装する"""
	train_set = 
	valid_set = 
	train_loader = torch.utils.data.DataLoader(
		dataset=train_set,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		worker_init_fn = worker_init_fn
		)
	valid_loader = torch.utils.data.DataLoader(
		dataset=valid_set,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=2
		)
	
	net = HuEtAl(
        input_channels=CHANNELS,
        n_classes=N_CLASS
		)
	net = net.to(device)
	optimizer = optim.SGD(net.parameters(), lr=LR)
	criterion = nn.CrossEntropyLoss()

	with mlflow.start_run():
		# mlflowで保存したいパラメータを指定
		mlflow.log_param('epoch', EPOCH)

		best_loss, best_acc = float('inf'), -float('inf')
		counter = 0
		for epoch in range(EPOCH):
			"""train"""
			train_loss, train_total, train_correct  = 0.0, 0.0, 0.0
			bar = tqdm(train_loader, leave=False)
			net.train()
			for i, (inputs, labels) in enumerate(bar):
				bar.set_description('Training batch ' + str(i+1))
				inputs, labels = inputs.to(device), labels.to(device)
				optimizer.zero_grad()
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				_, predicted = torch.max(outputs.data, 1)
				train_total += labels.size(0)
				train_correct += (predicted == labels).sum().item()
			avg_train_loss = train_loss / len(train_loader)
			avg_train_acc = train_correct / train_total
			# mlflowでlogを残す
			mlflow.log_metric('epoch train loss', avg_train_loss, epoch+1)
			mlflow.log_metric('epoch train accuracy', avg_train_acc, epoch+1)

			"""valdation"""
			valid_loss, valid_total, valid_correct = 0.0, 0.0, 0.0
			bar = tqdm(valid_loader, leave=False)
			net.eval()
			with torch.no_grad():
				for i, (inputs, labels) in enumerate(bar):
					bar.set_description('Validation batch ' + str(i+1))
					inputs, labels = inputs.to(device), labels.to(device)
					outputs = net(inputs)
					loss = criterion(outputs, labels)
					valid_loss += loss.item()
					_, predicted = torch.max(outputs.data, 1)
					valid_total += labels.size(0)
					valid_correct += (predicted == labels).sum().item()
			avg_valid_loss = valid_loss / len(valid_loader)
			avg_valid_acc = valid_correct / valid_total
			# mlflowでlogを残す
			mlflow.log_metric('epoch validation loss', avg_valid_loss, epoch+1)
			mlflow.log_metric('epoch validation accuracy', avg_valid_acc, epoch+1)

			# validation lossがbest lossより小さかったら更新し、モデルを保存する
			if best_loss > avg_valid_loss:
				torch.save(net.state_dict(), './best_loss_cnn_hu.pth')
				best_loss = avg_valid_loss
			
			# validation accuracyがbest accuracyより大きかったら更新し、モデルを保存する
			if best_acc < avg_valid_acc:
				torch.save(net.state_dict(), './best_acc_cnn_hu.pth')
				best_acc = avg_valid_acc
			
			# early stoppingはvalidation lossがbest lossを一定のエポック数から更新しなくなったら学習を打ち切る
			if EA_FLAG:
				if avg_valid_loss > best_loss:
					counter += 1
					if counter >= PATHIENCE:
						print(f'Early Stopping! ({epoch+1})')
						break
				else:
					counter = 0


if __name__ == '__main__':
	main()