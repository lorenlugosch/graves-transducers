import numpy as np
import torch
from tqdm import tqdm # for displaying progress bar
import os
import pandas as pd
from jiwer import wer as compute_WER

class Trainer:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		self.lr = config.lr
		self.checkpoint_path = os.path.join(self.config.folder, "training")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
		self.epoch = 0
		self.df = None
		if torch.cuda.is_available(): self.model.cuda()

	def load_checkpoint(self):
		if os.path.isfile(os.path.join(self.checkpoint_path, "model_state.pth")):
			try:
				device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
				self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth"), map_location=device))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def save_checkpoint(self):
		try:
			torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
		except:
			print("Could not save model")

	def log(self, results):
		if self.df is None:
			self.df = pd.DataFrame(columns=[field for field in results])
		self.df.loc[len(self.df)] = results
		self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

	def train(self, dataset, print_interval=100):
		train_WER = 0
		train_loss = 0
		num_examples = 0
		self.model.train()
		for g in self.optimizer.param_groups:
			print("Current learning rate:", g['lr'])
		#self.model.print_frozen()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,y,T,U = batch
			batch_size = len(x)
			num_examples += batch_size
			log_probs = self.model(x,y,T,U)
			loss = -log_probs.mean()
			self.optimizer.zero_grad()
			loss.backward()
			clip_value = 5
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
			self.optimizer.step()
			train_loss += loss.item() * batch_size #train_loss += loss.cpu().data.numpy().item() * batch_size
			#train_WER += WER.cpu().data.numpy().item() * batch_size
			if idx % print_interval == 0:
				print("loss: " + str(loss.cpu().data.numpy().item()))
				guess = self.model.infer(x)[0][:U[0]]
				print("guess:", dataset.tokenizer.DecodeIds(guess))
				truth = [yy for yy in y[0].cpu().data.numpy().tolist() if yy != -1]
				print("truth:", dataset.tokenizer.DecodeIds(truth))
				print("WER: ", compute_WER(dataset.tokenizer.DecodeIds(truth), dataset.tokenizer.DecodeIds(guess)))
				print("")

		train_loss /= num_examples
		train_WER /= num_examples
		#self.model.unfreeze_one_layer()
		results = {"loss" : train_loss, "WER" : train_WER, "set": "train"}
		self.log(results)
		self.epoch += 1
		return train_WER, train_loss

	def test(self, dataset, set):
		test_WER = 0
		test_loss = 0
		num_examples = 0
		self.model.eval()
		#self.model.cpu(); self.model.is_cuda = False # beam search is memory-intensive; do on CPU for now
		for idx, batch in enumerate(dataset.loader):
			x,y,T,U = batch
			batch_size = len(x)
			num_examples += batch_size
			log_probs = self.model(x,y,T,U)
			loss = -log_probs.mean()
			test_loss += loss.item() * batch_size #loss.cpu().data.numpy().item() * batch_size
			WERs = []
			guesses = self.model.infer(x)
			for i in range(batch_size):
				guess = guesses[i][:U[i]]
				truth = y[i].cpu().data.numpy().tolist()[:U[i]]
				WERs.append(compute_WER(dataset.tokenizer.DecodeIds(truth), dataset.tokenizer.DecodeIds(guess)))
			WER = np.array(WERs).mean()
			test_WER += WER * batch_size
			print("guess:", dataset.tokenizer.DecodeIds(guess))
			print("truth:", dataset.tokenizer.DecodeIds(truth))
			print("WER: ", compute_WER(dataset.tokenizer.DecodeIds(truth), dataset.tokenizer.DecodeIds(guess)))
			print("")

		test_loss /= num_examples
		self.scheduler.step(test_loss)
		test_WER /= num_examples
		results = {"loss" : test_loss, "WER" : test_WER, "set": set}
		self.log(results)
		return test_WER, test_loss
