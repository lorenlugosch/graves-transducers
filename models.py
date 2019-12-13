import torch
import torchaudio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import ctcdecode
#from losses import TransducerLoss

class CTCModel(torch.nn.Module):
	def __init__(self, config):
		super(CTCModel, self).__init__()
		self.encoder = Encoder(config)

		# beam search
		self.blank_index = self.encoder.blank_index; self.num_outputs = self.encoder.num_outputs
		labels = ["a" for _ in range(self.num_outputs)] # doesn't matter, just need 1-char labels
		self.decoder = ctcdecode.CTCBeamDecoder(labels, blank_id=self.blank_index, beam_width=config.beam_width)

	def load_pretrained(self, model_path=None):
		if model_path == None:
			model_path = os.path.join(self.checkpoint_path, "model_state.pth")
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.load_state_dict(torch.load(model_path, map_location=device))

	def forward(self, x, y, T, U):
		"""
		returns log probs for each example
		"""

		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()
			y = y.cuda()

		# run the neural network
		encoder_out = self.encoder.forward(x, T)

		# run the forward algorithm to compute the log probs
		downsampling_factor = max(T) / encoder_out.shape[1]
		T = [round(t / downsampling_factor) for t in T]
		encoder_out = encoder_out.transpose(0,1) # (N, T, #labels) --> (T, N, #labels)
		log_probs = -torch.nn.functional.ctc_loss(	log_probs=encoder_out,
								targets=y,
								input_lengths=T,
								target_lengths=U,
								reduction="none",
								blank=self.blank_index)
		return log_probs

	def infer(self, x, T=None):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		# run the neural network
		out = self.encoder.forward(x, T)

		# run a beam search
		out = torch.nn.functional.softmax(out, dim=2)
		beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(out)
		decoded = [beam_result[i][0][:out_seq_len[i][0]].tolist() for i in range(len(out))]
		return decoded


class TransducerModel(torch.nn.Module):
	def __init__(self, config):
		super(TransducerModel, self).__init__()
		self.encoder = Encoder(config)
		self.decoder = AutoregressiveDecoder(config)
		self.transducer_loss = TransducerLoss()

	def forward(self, x, y, T, U):
		"""
		returns log probs for each example
		"""

		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()
			y = y.cuda()

		# run the neural network
		encoder_out = self.encoder.forward(x, T) # (N, T, #labels)
		decoder_out = self.decoder.forward(y, U) # (N, U, #labels)

		"""
		log_probs = -self.transducer_loss(encoder_out=encoder_out,
						decoder_out=decoder_out,
						#joint_network=self.joint_network,
						targets=y,
						input_lengths=T,
						target_lengths=U,
						reduction="none",
						blank=self.blank_index)
		return log_probs
		"""
		return 1

class Conv(torch.nn.Module):
	def __init__(self, in_dim, out_dim, filter_length, stride):
		super(Conv, self).__init__()
		self.conv = torch.nn.Conv1d(	in_channels=in_dim,
						out_channels=out_dim,
						kernel_size=filter_length,
						stride=stride
		)
		self.filter_length = filter_length

	def forward(self, x):
		out = x.transpose(1,2)
		left_padding = int(self.filter_length/2)
		right_padding = int(self.filter_length/2)
		out = torch.nn.functional.pad(out, (left_padding, right_padding))
		out = self.conv(out) / self.filter_length # HACK seeing if this helps
		out = out.transpose(1,2)
		return out

class Encoder(torch.nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()
		self.layers = []

		# compute FBANK features from waveform
		layer = ComputeFBANK(config)
		self.layers.append(layer)
		out_dim = config.num_mel_bins

		# convolution to give future context
		context_len = 11 # 11 fbank frames
		layer = Conv(in_dim=out_dim, out_dim=config.num_hidden, filter_length=context_len, stride=2)
		self.layers.append(layer)
		out_dim = config.num_hidden
		layer = torch.nn.LeakyReLU(0.125)
		self.layers.append(layer)

		for idx in range(config.num_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.num_hidden, batch_first=True, bidirectional=config.bidirectional)
			self.layers.append(layer)

			out_dim = config.num_hidden
			if config.bidirectional: out_dim *= 2

			# grab hidden states for each timestep
			layer = RNNOutputSelect()
			self.layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=0.2)
			self.layers.append(layer)

			# fully-connected
			layer = torch.nn.Linear(out_dim, config.num_hidden)
			out_dim = config.num_hidden
			self.layers.append(layer)
			layer = torch.nn.LeakyReLU(0.125)
			self.layers.append(layer)

			# downsample
			if idx == 0:
				layer = Downsample(method="avg", factor=2, axis=1)
				self.layers.append(layer)

		#layer = torch.nn.LeakyReLU(0.125)
		#self.layers.append(layer)

		# final classifier
		self.num_outputs = config.num_tokens + 1 # for blank symbol
		self.blank_index = config.num_tokens
		layer = torch.nn.Linear(out_dim, self.num_outputs)
		self.layers.append(layer)

		layer = torch.nn.LogSoftmax(dim=2)
		self.layers.append(layer)

		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, x, T):
		out = (x,T)
		for layer in self.layers:
			out = layer(out)

		return out

class AutoregressiveDecoder(torch.nn.Module):
	def __init__(self, config):
		super(AutoregressiveDecoder, self).__init__()
		#self.layers = []
		self.num_outputs = config.num_tokens + 1 # for blank symbol

	def forward(self, y, U):
		batch_size = y.shape[0]
		return torch.randn(batch_size, max(U), self.num_outputs)

class RNNOutputSelect(torch.nn.Module):
	def __init__(self):
		super(RNNOutputSelect, self).__init__()

	def forward(self, input):
		return input[0]

class NCL2NLC(torch.nn.Module):
	def __init__(self):
		super(NCL2NLC, self).__init__()

	def forward(self, input):
		"""
		input : Tensor of shape (batch size, T, Cin)
		Outputs a Tensor of shape (batch size, Cin, T).
		"""

		return input.transpose(1,2)


class Downsample(torch.nn.Module):
	"""
	Downsamples the input in the time/sequence domain
	"""
	def __init__(self, method="none", factor=1, axis=1):
		super(Downsample,self).__init__()
		self.factor = factor
		self.method = method
		self.axis = axis
		methods = ["none", "avg", "max"]
		if self.method not in methods:
			print("Error: downsampling method must be one of the following: \"none\", \"avg\", \"max\"")
			sys.exit()

	def forward(self, x):
		if self.method == "none":
			return x.transpose(self.axis, 0)[::self.factor].transpose(self.axis, 0)
		if self.method == "avg":
			return torch.nn.functional.avg_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)
		if self.method == "max":
			return torch.nn.functional.max_pool1d(x.transpose(self.axis, 2), kernel_size=self.factor, ceil_mode=True).transpose(self.axis, 2)

class ComputeFBANK(torch.nn.Module):
	def __init__(self, config):
		super(ComputeFBANK,self).__init__()
		self.num_mel_bins = config.num_mel_bins
		self.subtract_mean = config.normalize_fbank
		self.fbank_params = {
                    "channel": 0,
                    "dither": 0.0,
                    "window_type": "hanning",
                    "num_mel_bins":self.num_mel_bins,
                    "remove_dc_offset": False,
                    "round_to_power_of_two": False,
                    "sample_frequency":16000.0,
                }

	def forward(self, input):
		"""
		input : (x,T)
		x : waveforms
		T : durations
		returns : (normalized) FBANK feature vectors
		"""
		#fbank = torch.stack([torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **self.fbank_params) for xx in x])
		fbanks = []
		x,T = input
		batch_size = len(x)
		for idx in range(batch_size):
			fbank_ = torchaudio.compliance.kaldi.fbank(x[idx].unsqueeze(0), **self.fbank_params)
			fbanks.append(fbank_)

		if self.subtract_mean:
			downsample_factor = max(T) / fbanks[0].shape[1]
			for idx in range(batch_size):
				T_ = int(T[idx] / downsample_factor)
				fbanks[idx][:T_, :] -= fbanks[idx][:T_, :].mean(0)

		fbank = torch.stack(fbanks)
		return fbank

class SpecAugment(torch.nn.Module):
	def __init__(self):
		super(SpecAugment,self).__init__()
		self.T_param = 100
		self.F_param = 27 # from Table 1: https://arxiv.org/pdf/1904.08779.pdf

	def forward(self, x):
		"""
		x : spectrogram
		returns : zero'd spectrogram
		"""
		if self.training:
			N = x.shape[0]
			T = x.shape[1]
			F = x.shape[2]

			mask = torch.ones(x.shape)
			for idx in range(N):
				f_len = int(np.random.rand()*self.F_param)
				f_min = int((F - f_len) * np.random.rand())
				f_max = f_min + f_len

				t_len = int(np.random.rand()*self.T_param)
				t_min = int((T - t_len) * np.random.rand())
				t_max = t_min + t_len

				mask[idx,t_min:t_max,:] = 0
				mask[idx,:,f_min:f_max] = 0

			mask.to(x.device)
			out = x * mask
			return out
		else:
			return x
