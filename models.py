import torch
import torchaudio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import ctcdecode
import time

import torch.utils
import torch.utils.checkpoint

#awni_transducer_path = '/home/lugosch/code/transducer'
#sys.path.insert(0, awni_transducer_path)
#from transducer import Transducer

from warprnnt_pytorch import RNNTLoss
rnnt_loss = RNNTLoss()

class TransducerModel(torch.nn.Module):
	def __init__(self, config):
		super(TransducerModel, self).__init__()
		self.encoder = Encoder(config)
		self.decoder = AutoregressiveDecoder(config)
		self.joiner = Joiner(config)
		self.blank_index = self.joiner.blank_index; self.num_outputs = self.joiner.num_outputs
		#self.transducer_loss = Transducer(blank_label=self.blank_index)
		self.ctc_decoder = ctcdecode.CTCBeamDecoder(["a" for _ in range(self.num_outputs)], blank_id=self.blank_index, beam_width=config.beam_width)
		self.beam_width = config.beam_width

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
		encoder_out = self.encoder.forward(x, T) # (N, T, #labels)
		decoder_out = self.decoder.forward(y, U) # (N, U, #labels)
		joint_out = self.joiner.forward(encoder_out, decoder_out)
		#joint_out = torch.utils.checkpoint.checkpoint(self.joiner, encoder_out, decoder_out)

		downsampling_factor = max(T) / encoder_out.shape[1]
		T = [round(t / downsampling_factor) for t in T]
		T = torch.IntTensor(T)
		U = torch.IntTensor(U)
		if next(self.parameters()).is_cuda:
			T = T.cuda()
			U = U.cuda()
		y = y.int()
		#yy = [y[i, :U[i]].tolist() for i in range(len(y))]
		#y = torch.IntTensor([yyyy for yyy in yy for yyyy in yyy])

		#log_probs = -self.transducer_loss.apply(joint_out, y, T, U)
		log_probs = -rnnt_loss(acts=joint_out, labels=y, act_lens=T, label_lens=U)
		return log_probs

	def infer(self, x, T=None, tokenizer=None):
		# move inputs to GPU
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		with torch.no_grad():
			encoder_out = self.encoder.forward(x, T)
			batch_size = encoder_out.shape[0]
			downsampling_factor = max(T) / encoder_out.shape[1]
			T = [round(t / downsampling_factor) for t in T]
			decoded = []
			# Alex Graves's search (https://arxiv.org/pdf/1211.3711.pdf, section 2.6)
			"""
			for i in range(batch_size):
				# Beam structure: [y, state, log_prob]
				B = [
					[[self.decoder.start_symbol], torch.stack([self.decoder.initial_state] * 1), 0.]
				]
				for t in range(T[i]):
					A = B
					B = []
					A_y = [y[0] for y in A]
					for y in A:
						pref_y = [y[0,:i] for i in range(1, len(y[0]))]
						pref_y_intersect_A = [pref for pref in pref_y if pref in A_y]

					while :
						y_star = 

			"""
			U_max = max(T) # don't allow long outputs
			# my hacky search
			for i in range(batch_size):
				# Initialize the beam
				# t, u, y, state, log_prob
				beam = [
					[0, 0, [self.decoder.start_symbol], torch.stack([self.decoder.initial_state] * 1), 0.]
				]

				# Until all hypotheses have reached the final timestep, continue
				done = False
				while not done:

					# Consider each hypothesis in beam
					extended_hypotheses = []
					for hypothesis in beam:
						t = hypothesis[0]
						u = hypothesis[1]
						y_star = hypothesis[2]
						decoder_state = hypothesis[3]
						log_prob = hypothesis[4]

						if t == T[i]-1: # we've reached the final timestep
							extended_hypotheses.append(hypothesis)
						else:
							# Compute next step probs
							decoder_input = torch.tensor([ y_star[-1] ] * 1).to(x.device)
							decoder_out, decoder_state = self.decoder.forward_one_step(decoder_input, decoder_state)
							joint_out = self.joiner.forward_one_step(encoder_out[i, t], decoder_out).reshape(-1) #(encoder_out[i, t] + decoder_out).log_softmax(1).reshape(-1)

							# Append extended hypotheses to extended_hypotheses
							for l in range(joint_out.shape[0]): # not necessary to prune here?
								y_u = l
								t_extended = t + 1 if y_u == self.blank_index else t
								u_extended = u     if y_u == self.blank_index else u + 1
								y_star_extended = y_star if y_u == self.blank_index else y_star + [y_u]
								log_prob_extended = log_prob + joint_out[y_u].item()

								extended_hypothesis = [
									t_extended,
									u_extended,
									y_star_extended,
									decoder_state,
									log_prob_extended
								]
								extended_hypotheses.append(extended_hypothesis)

					# Merge identical hypotheses
					#for idx, i_hypothesis in enumerate(extended_hypotheses):
					#	for jdx, j_hypothesis in enumerate(extended_hypotheses):
					#		if i_hypothesis[2] == j_hypothesis[2] and idx != jdx:
					#			i_hypothesis[4] = torch.tensor([i_hypothesis[4], j_hypothesis[4]]).logsumexp(0).item()
					#			_ = extended_hypotheses.pop(jdx)

					# Set beam equal to top (beam_width) extended hypotheses
					beam_scores = torch.tensor([hypothesis[4] for hypothesis in extended_hypotheses])
					top_indices = beam_scores.topk(self.beam_width)[1]
					beam = [extended_hypotheses[index] for index in top_indices]
					t_beam = torch.tensor([hypothesis[0] for hypothesis in beam]) # timestep pointer for each hypothesis
					u_beam = torch.tensor([hypothesis[1] for hypothesis in beam])
					done = (u_beam > U_max).any() or not (t_beam < T[i]-1).all() # if hypothesis too long or all hypotheses have reached the final timestep, we're done

					if tokenizer is not None:
						for hypothesis in beam:
							print(tokenizer.DecodeIds(hypothesis[2][1:]) + " | " + str(hypothesis[4]) )
						print("")
						time.sleep(1)

				# get top (0) hypothesis, which is the second (2) element of the beam entry, and remove the start symbol (1:)
				decoded_i = beam[0][2][1:]
				decoded.append(decoded_i)
		return decoded


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

class Joiner(torch.nn.Module):
	def __init__(self, config):
		super(Joiner, self).__init__()
		self.tanh = torch.nn.Tanh()
		self.num_outputs = config.num_tokens + 1
		self.blank_index = 0 # hard coded
		self.linear = torch.nn.Linear(config.num_joiner_hidden, self.num_outputs)

	def forward(self, encoder_out, decoder_out):
		combined = (encoder_out.unsqueeze(2) + decoder_out.unsqueeze(1))
		out = self.tanh(combined)
		out = self.linear(out).log_softmax(3)
		return out

	def forward_one_step(self, encoder_out, decoder_out):
		combined = encoder_out + decoder_out
		out = self.tanh(combined)
		out = self.linear(out)
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
		layer = Conv(in_dim=out_dim, out_dim=config.num_encoder_hidden, filter_length=context_len, stride=2)
		self.layers.append(layer)
		out_dim = config.num_encoder_hidden
		layer = torch.nn.LeakyReLU(0.125)
		self.layers.append(layer)

		for idx in range(config.num_encoder_layers):
			# recurrent
			layer = torch.nn.GRU(input_size=out_dim, hidden_size=config.num_encoder_hidden, batch_first=True, bidirectional=config.bidirectional)
			self.layers.append(layer)

			out_dim = config.num_encoder_hidden
			if config.bidirectional: out_dim *= 2

			# grab hidden states for each timestep
			layer = RNNOutputSelect()
			self.layers.append(layer)

			# dropout
			layer = torch.nn.Dropout(p=0.2)
			self.layers.append(layer)

			# fully-connected
			layer = Conv(in_dim=out_dim, out_dim=config.num_encoder_hidden, filter_length=3, stride=1) #layer = torch.nn.Linear(out_dim, config.num_encoder_hidden)
			out_dim = config.num_encoder_hidden
			self.layers.append(layer)
			layer = torch.nn.LeakyReLU(0.125)
			self.layers.append(layer)

			# downsample
			#if idx == 0:
			#	layer = Downsample(method="avg", factor=2, axis=1)
			#	self.layers.append(layer)

		# final classifier
		self.num_outputs = config.num_joiner_hidden #config.num_tokens + 1 # for blank symbol
		layer = torch.nn.Linear(out_dim, self.num_outputs)
		self.layers.append(layer)

		#layer = Downsample(method="avg", factor=4, axis=1)
		#self.layers.append(layer)

		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, x, T):
		out = (x,T)
		for layer in self.layers:
			out = layer(out)

		return out

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()

		self.layers = []
		self.num_decoder_layers = num_decoder_layers
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
			if "gru" in layer.name:
				if index == 0:
					gru_input = input
				else:
					gru_input = layer_out
				layer_out = layer(gru_input, previous_state[:, gru_index])
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
		num_decoder_hidden = config.num_decoder_hidden
		num_decoder_layers = config.num_decoder_layers
		input_size = num_decoder_hidden

		self.initial_state = torch.nn.Parameter(torch.randn(num_decoder_layers,num_decoder_hidden))
		self.embed = torch.nn.Embedding(num_embeddings=config.num_tokens + 1, embedding_dim=num_decoder_hidden)
		self.rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, input_size, dropout=0.5)
		self.num_outputs = config.num_joiner_hidden
		self.linear = torch.nn.Linear(num_decoder_hidden, self.num_outputs)
		self.start_symbol = config.num_tokens

	def forward_one_step(self, input, previous_state):
		embedding = self.embed(input)
		state = self.rnn.forward(embedding, previous_state)
		out = self.linear(state[:,-1])
		return out, state

	def forward(self, y, U):
		batch_size = y.shape[0]
		"""
		out = torch.zeros(batch_size, max(U)+1, self.num_outputs) #.log_softmax(2)
		out = out.to(y.device)
		"""
		U_max = y.shape[1]
		outs = []
		state = torch.stack([self.initial_state] * batch_size).to(y.device)
		for u in range(U_max + 1): # need U+1 to get null output for final timestep 
			if u == 0:
				decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
			else:
				decoder_input = y[:,u-1]
			out, state = self.forward_one_step(decoder_input, state)
			outs.append(out)
		out = torch.stack(outs, dim=1)

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
