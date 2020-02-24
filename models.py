import torch
import torchaudio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import ctcdecode
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

class TransducerLoss(torch.nn.Module):
	def __init__(self):
		super(TransducerLoss, self).__init__()

	def compute_log_alpha_and_log_prob(self, encoder_out, decoder_out, y, blank):
		"""
		encoder_out: FloatTensor (T, #labels)
		decoder_out: FloatTensor (U+1, #labels)
		y: LongTensor (U,)
		blank: int
		"""
		T = len(encoder_out)
		U = len(y)
		log_alpha = torch.zeros(T, U+1) #[]
		log_alpha = log_alpha.to(encoder_out.device)
		for t in range(T):
			for u in range(U + 1):
				if u == 0:
					if t == 0:
						log_alpha[t,u] = 0.

					else: #t > 0
						null_t_1_0 = encoder_out[t-1, blank] + decoder_out[0, blank]
						log_alpha[t,u] = log_alpha[t-1,u] + null_t_1_0

				else: #u > 0
					if t == 0:
						y_0_u_1 = encoder_out[t, y[u-1]] + decoder_out[u-1, y[u-1]]
						log_alpha[t,u] = log_alpha[t,u-1] + y_0_u_1

					else: #t > 0
						y_t_u_1 = encoder_out[t, y[u-1]] + decoder_out[u-1, y[u-1]]
						null_t_1_u = encoder_out[t-1, blank] + decoder_out[u, blank]
						log_alpha[t,u] = torch.logsumexp(torch.stack([
							log_alpha[t-1,u] + null_t_1_u,
							log_alpha[t,u-1] + y_t_u_1
						]), dim=0)

		null_T_1_U = encoder_out[T-1, blank] + decoder_out[U, blank]
		log_p_y_x = log_alpha[T-1,U] + null_T_1_U
		return log_p_y_x

	def forward(self,encoder_out,decoder_out,targets,input_lengths,target_lengths,reduction="none",blank=0):
		"""
		encoder_out: FloatTensor (N, max(input_lengths), #labels)
		decoder_out: FloatTensor (N, max(target_lengths)+1, #labels)
		targets: LongTensor (N, max(target_lengths))
		input_lengths: LongTensor (N)
		target_lengths: LongTensor (N)
		reduction: "none", "avg"
		blank: int
		"""

		batch_size = len(input_lengths)
		log_probs = []
		for i in range(0, batch_size):
			encoder_out_ = encoder_out[i, :input_lengths[i], :]
			decoder_out_ = decoder_out[i, :target_lengths[i]+1, :]
			y = targets[i, :target_lengths[i]]
			log_p_y_x = self.compute_log_alpha_and_log_prob(encoder_out_, decoder_out_, y, blank)
			log_probs.append(log_p_y_x)
		log_probs = torch.stack(log_probs)

		return log_probs

class TransducerModel(torch.nn.Module):
	def __init__(self, config):
		super(TransducerModel, self).__init__()
		self.encoder = Encoder(config)
		self.decoder = AutoregressiveDecoder(config)
		self.blank_index = self.encoder.blank_index
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

		log_probs = self.transducer_loss(encoder_out=encoder_out,
						decoder_out=decoder_out,
						#joint_network=self.joint_network,
						targets=y,
						input_lengths=T,
						target_lengths=U,
						reduction="none",
						blank=self.blank_index)
		return log_probs

class TimeRestrictedSelfAttention(torch.nn.Module):
	def __init__(self, in_dim, out_dim, key_dim, filter_length, stride):
		super(TimeRestrictedSelfAttention, self).__init__()
		self.key_linear = torch.nn.Linear(in_dim, key_dim)
		self.query_linear = torch.nn.Linear(in_dim, key_dim)
		self.value_linear = torch.nn.Linear(in_dim, out_dim)

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
		out = self.conv(out)
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

		# convolutional
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

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()
		# self.gru = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden)
		# self.dropout = torch.nn.Dropout(dropout)

		self.layers = []
		self.num_layers = num_decoder_layers
		for index in range(num_decoder_layers):
			if index == 0: 
				layer = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
			else:
				layer = torch.nn.GRUCell(input_size=num_decoder_hidden, hidden_size=num_decoder_hidden) 
			layer.name = "gru%d"%index
			self.layers.append(layer)

			layer = torch.nn.Dropout(p=dropout)
			layer.name = "dropout%d"%index
			self.layers.append(layer)
		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, input, previous_state):
		"""
		input: Tensor of shape (batch size, input_size)
		previous_state: Tensor of shape (batch size, num_decoder_layers, num_decoder_hidden)
		Given the input vector, update the hidden state of each decoder layer.
		"""
		# return self.gru(input, previous_state)

		state = []
		batch_size = input.shape[0]
		gru_index = 0
		for index, layer in enumerate(self.layers):
			if index == 0:
				layer_out = layer(input, previous_state[:, gru_index])
				state.append(layer_out)
				gru_index += 1
			else:
				if "gru" in layer.name:
					layer_out = layer(layer_out, previous_state[:, gru_index])
					state.append(layer_out)
					gru_index += 1
				else: 
					layer_out = layer(layer_out)
		state = torch.stack(state, dim=1)
		return state

class AutoregressiveDecoder(torch.nn.Module):
	def __init__(self, config):
		super(AutoregressiveDecoder, self).__init__()
		self.layers = []
		#self.rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, input_size, dropout=0.5)
		self.num_outputs = config.num_tokens + 1 # for blank symbol

	def forward(self, y, U):
		batch_size = y.shape[0]
		out = torch.zeros(batch_size, max(U)+1, self.num_outputs) #.log_softmax(2)
		out = out.to(y.device)
		return out

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
		self.normalizer = torch.nn.Parameter(torch.tensor([1/10] * self.num_mel_bins))

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
			fbank_ = fbank_ * self.normalizer.unsqueeze(0)
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
