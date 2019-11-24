import torch
import numpy as np
from models import Model
from data import get_ASR_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
train = args.train
restart = args.restart
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Initialize model
model = Model(config=config)

if train:
	# Generate datasets
	train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

	# Train the final model
	trainer = Trainer(model=model, config=config)
	if restart: trainer.load_checkpoint()

	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_WER, train_loss = trainer.train(train_dataset)
		valid_WER, valid_loss = trainer.test(valid_dataset, set="valid")

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train WER: %.2f| train loss: %.2f| valid WER: %.2f| valid loss: %.2f\n" % (train_WER, train_loss, valid_WER, valid_loss) )

		trainer.save_checkpoint()

	test_WER, test_loss = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f \n" % (test_WER, test_loss) )
