import torch
import torchaudio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.layers = []

		# compute FBANK features from waveform
		layer = ComputeFBANK()
		self.layers.append(layer)
		out_dim = 40

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
			layer = torch.nn.Dropout()
			self.layers.append(layer)

			# downsample
			layer = Downsample(method="avg", factor=2, axis=1)
			self.layers.append(layer)

		# final classifier
		self.num_outputs = config.num_tokens + 1 # for blank symbol
		self.blank_index = config.num_tokens
		layer = torch.nn.Linear(out_dim, self.num_outputs)
		self.layers.append(layer)

		layer = torch.nn.LogSoftmax(dim=2)
		self.layers.append(layer)

		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, x, y, T, U):
		"""
		returns log probs for each example
		"""

		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()
			y = y.cuda()

		# run the neural network
		out = x
		for layer in self.layers:
			#try:
			#	print(out.shape)
			#except:
			#	pass
			out = layer(out)

		# run the forward algorithm to compute the log probs
		downsampling_factor = max(T) / out.shape[1]
		T = [round(t / downsampling_factor) for t in T]
		out = out.transpose(0,1) # (N, T, #labels) --> (T, N, #labels)
		log_probs = -torch.nn.functional.ctc_loss(	log_probs=out,
								targets=y,
								input_lengths=T,
								target_lengths=U,
								reduction="none",
								blank=self.blank_index)
		"""
		encoder_out = ... # (N, T, C1)
		decoder_out = ... # (N, U, C2)
		log_probs = transducer_forward(	encoder_out=encoder_out,
						decoder_out=decoder_out,
						joint_network=self.joint_network,
						targets=y,
						input_lengths=T,
						target_lengths=U,
						reduction="none",
						blank=self.blank_index)
		"""
		return log_probs

	def infer(self, x):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		# run the neural network
		out = x
		for layer in self.layers:
			out = layer(out)

		# run a greedy search
		peaks = out.max(2)[1]
		decoded = []
		for idx in range(len(peaks)):
			decoded_ = []
			p_ = None
			for p in peaks[idx]:
				p = p.item()
				if p != self.num_outputs-1 and p != p_:
					decoded_.append(p)
				p_ = p
			decoded.append(decoded_)
		return decoded

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
	def __init__(self):
		super(ComputeFBANK,self).__init__()
		self.fbank_params = {
                    "channel": 0,
                    "dither": 0.0,
                    "window_type": "hanning",
                    "num_mel_bins":40,
                    "remove_dc_offset": False,
                    "round_to_power_of_two": False,
                    "sample_frequency":16000.0,
                }

	def forward(self, x):
		"""
		x : waveforms
		returns : FBANK feature vectors
		"""
		fbank = torch.stack([torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **self.fbank_params) for xx in x])
		return fbank
