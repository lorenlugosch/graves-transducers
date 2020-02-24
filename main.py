import torch
import numpy as np
from models import CTCModel, TransducerModel
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
model = TransducerModel(config=config) #CTCModel(config=config)
print(model)

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_ASR_datasets(config)

trainer = Trainer(model=model, config=config)
if restart: trainer.load_checkpoint()

#######################
#from data import CollateWavsASR
#c = CollateWavsASR()
#indices = [247139, 46541, 53391, 188748, 276074, 196684, 211271, 159381, 137485, 204068, 273793, 4477, 68569, 240939, 233307, 52096, 72788, 168682, 131440, 50584, 157714, 190209, 79150, 159430, 252179, 21565, 44222, 277035, 6492, 165880, 192524, 262075, 254863, 173450, 80020, 232210, 4045, 89528, 126462, 67697, 267582, 119138, 252103, 275955, 179933, 238829, 228952, 253370, 87137, 251881, 213141, 139106, 20439, 174443, 193189, 50926, 209070, 25481, 221399, 36916, 54646, 265435, 175335, 59256]
#b = [ train_dataset.__getitem__(idx) for idx in indices]
#batch = c.__call__(b)
#x,y,T,U,idxs = batch
#log_probs = model(x,y,T,U)
#######################

# Train the final model
if train:
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_WER, train_loss = trainer.train(train_dataset)
		valid_WER, valid_loss = trainer.test(valid_dataset, set="valid")

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train WER: %.2f| train loss: %.2f| valid WER: %.2f| valid loss: %.2f\n" % (train_WER, train_loss, valid_WER, valid_loss) )

		trainer.save_checkpoint(WER=valid_WER)

	trainer.load_best_model()
	test_WER, test_loss = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f \n" % (test_WER, test_loss) )
