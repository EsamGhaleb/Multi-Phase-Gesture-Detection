#!/usr/bin/env python
import numpy as np
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as td
import torch.nn.functional as F
import math


from itertools import chain

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Labeler(nn.Module):
    def __init__(self, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        super().__init__()
        self._labelset_size = labelset_size
        self._pad = pad_id
        # _bos is the id of the begin of sentence token, it is the index of the bos token in the tagset
        self._bos = labels.index('bos')
        self._eos = len(labels)
        self._labels = labels
        self._classes = classes
        self._max_seq_len = max_seq_len
        self._ngram_size = ngram_size
        self._label_embed_dim = label_embed_dim
        self._include_embeddings = include_embeddings
        self._labeler_name = labeler_name
        self._recurrent_encoder = recurrent_encoder
        self._training_seq = training_seq
        self._num_of_classes = len(classes)

    # Python properties allow client code to access the property 
    # without the risk of modifying it.

    @property
    def labelset_size(self):
        return self._labelset_size

    @property
    def pad(self):
        return self._pad

    @property
    def bos(self):
        return self._bos
    @property
    def eos(self):
        return self._eos
    @property
    def labels(self):
        return self._labels
    @property
    def classes(self):
        return self._classes
    @property
    def max_seq_len(self):
        return self._max_seq_len
    @property
    def ngram_size(self):
        return self._ngram_size
    @property
    def label_embed_dim(self):
        return self._label_embed_dim
    @property
    def include_embeddings(self):
        return self._include_embeddings
    @property
    def labeler_name(self):
        return self._labeler_name
    @property
    def recurrent_encoder(self):
        return self._recurrent_encoder
    @property
    def training_seq(self):
        return self._training_seq
    @property
    def num_of_classes(self):
        return self._num_of_classes
        

    def num_parameters(self):
        return sum(np.prod(theta.shape) for theta in self.parameters())
        
    def forward(self, x, y):
        raise NotImplementedError("Each type of tagger will have a different implementation here")

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)   

    def greedy(self, x):
        """
        For each cpd Y[i]|X=x, predicts the mode of the cpd.
        x: [batch_size, max_length]

        Return: tag sequences [batch_size, max_length]
        """
        raise NotImplementedError("Each type of tagger differs here")

    def sample(self, x, sample_size=None):
        """
        Per snippet sequence in the batch, draws a number of samples from the model, each sample is a complete tag sequence.

        x: [batch_size, max_len]

        Return: tag sequences with shape [batch_size, max_len] if sample_size is None
            else with shape [sample_size, batch_size, max_len]
        """
        raise NotImplementedError("Each type of tagger differs here")

    def loss(self, x, y):   
        """
        Compute a scalar loss from a batch of sentences.
        The loss is the negative log likelihood of the model estimated on a single batch:
            - 1/batch_size * \sum_{s} log P(y[s]|x[s], theta)

        x: snippet sequences [batch_size, max_length] 
        y: tag sequences [batch_size, max_length] 
        """
        return -self.log_prob(x=x, y=y).mean(0)
class IndependentLabeler(Labeler):
    def __init__(self, gcns_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """        
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        hidden_size: dimensionality of hidden layers
        recurrent_encoder: enable recurrent encoder
        bidirectional_encoder: for a recurrent encoder, make it bidirectional
        """
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        self.num_of_labels = len(labels) - 1 # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        # we need to embed tags in the history 
        # we need to encode snippet sequences
        if include_embeddings:
            first_fc_dim =  2 * joints_embed_dim
        else:
            first_fc_dim = joints_embed_dim
        if recurrent_encoder:
            # use transformer here
            nhead = 8
            num_layers = 4
            dim_feedforward = hidden_size * 4
            self.positional_encoding = PositionalEncoding(d_model=joints_embed_dim, dropout=0.1, max_len=100)
            encoder_layers = nn.TransformerEncoderLayer(d_model=joints_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.encoder.apply(self.initialize_weights)
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
        else:
            self.encoder = None
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
        self.logits_predictor = nn.Sequential(
            nn.Linear(int(first_fc_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            )
        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
                      
    def forward(self, x, y, eval=False, get_embeddings=False, get_predictions=False, e=None, lengths=None, keep_prob=0.9):
        """
        Parameterise the conditional distributions over Y[i] given history y[:i] and all of x.

        This procedure takes care that the ith output distribution conditions only on the n-1 observations before y[i].
        It also takes care of padding to the left with BOS symbols.

        x: snippet sequences [batch_size, max_length]
        y: tag sequences  [batch_size, max_length]

        Return: a batch of V-dimensional Categorical distributions, one per step of the sequence.
        """
        # Let's start by encoding the snippet sequences
        # 1. we embed the snippets independently
        # [batch_size, max_length, embed_dim]
        
        # We begin by embedding the tokens        
        # [batch_size, max_length, embed_dim]
        if not get_predictions:
            N, S, C, F, V, M = x.shape
            x = x.view(N*S, C, F, V, M)
            # actual lengths of each sequence in the batch is needed for packing
            e = self.joints_embed(x, keep_prob=keep_prob)
            # [batch_size, max_length, embed_dim]
            e = e.view(N, S, -1)
            if self.encoder is not None:
                # 2. and then encode them in their left-to-right and right-to-left context
                # [batch_size, max_length, 2*hidden_size] 
                e = self.positional_encoding(e.permute(1, 0, 2))
                src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
                for i in range(N):
                    src_mask[i, lengths[i]:] = True
                h = self.encoder(e, src_key_padding_mask=src_mask)
                try:
                    assert h.shape == (S, N, self.joints_embed_dim)
                except AssertionError as error:
                    print('the input shape is ' + str(e.shape))
                    print('the output shape is ' + str(h.shape))
                    print('The mask shape is ' + str(src_mask.shape))
                    print('The lengths array is ' + str(lengths))
                    print(S, N, self.joints_embed_dim)
                    print(error)
                    # halt execution
                    raise error
                h = h.permute(1, 0, 2)
                 # concatenate h and e, in the embedding dimension
                # [batch_size, max_length, 2*hidden_size + embed_dim]
                if self.include_embeddings:
                    e = torch.cat([e, h], 2)
                else:
                    e = h
        if get_embeddings:
            return e
        # We are now ready to map the state of each step of the sequence to a C-dimensional vector of logits
        # we do so using our FFNN
        # [batch_size, max_length, num_of_labels]
        
        s = self.logits_predictor(e)
        if not eval:
            return s
        else:
            cat_log_probs = td.Categorical(logits=s)
            return cat_log_probs.probs

class AutoregressiveLabeler(Labeler):
    def __init__(self, gcns_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """
        ngram_size: longest ngram (for tag sequence)
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        label_embed_dim: dimensionality of tag embeddings (needed to encode the history of ngram_size-1 tags)
        hidden_size: dimensionality of hidden layers
        """
        # we need to embed tags in the history 
        
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        assert ngram_size > 1, "This class expects at least ngram_size 2. If you want ngram_size=1, use the NeuralUnigramLabeler"        
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        # we need to embed tags in the history 
        self.label_embed = nn.Embedding(labelset_size, embedding_dim=label_embed_dim) 
        nn.init.normal_(self.label_embed.weight, 0, math.sqrt(2. / label_embed_dim))
        # we need to encode snippet sequences
        if recurrent_encoder:
            # use transformer
            nhead = 8
            num_layers = 4
            dim_feedforward = hidden_size * 4
            self.positional_encoding = PositionalEncoding(d_model=joints_embed_dim, dropout=0.1, max_len=100)
            encoder_layers = nn.TransformerEncoderLayer(d_model=joints_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.encoder.apply(self.initialize_weights)
        else:
            self.encoder = None
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
        # use transformer for tag encoding: always
        nhead = 4
        num_layers = 2
        dim_feedforward = hidden_size * 2
        self.tag_positional_encoding = PositionalEncoding(d_model=label_embed_dim, dropout=0.1, max_len=100)
        decoder_layers = nn.TransformerEncoderLayer(d_model=label_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
        self.tag_encoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)
        self.tag_encoder.apply(self.initialize_weights) 
        
        if self.include_embeddings:
            first_fc_dim = label_embed_dim + 2 * joints_embed_dim
        else:
            first_fc_dim = label_embed_dim + joints_embed_dim
        self.logits_predictor = nn.Sequential(
            nn.Linear(int(first_fc_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
        )
        
         # initialise the logits_predictor

        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))
    
    def forward(self, x, y, eval=False, get_embeddings=False, get_predictions=False, u=None, lengths=None, keep_prob=0.9):
        """
        Parameterise the conditional distributions over Y[i] given history y[:i] and all of x.

        This procedure takes care that the ith output distribution conditions only on the n-1 observations before y[i].
        It also takes care of padding to the left with BOS symbols.

        x: snippet sequences [batch_size, max_length]
        y: tag sequences  [batch_size, max_length]

        Return: a batch of V-dimensional Categorical distributions, one per step of the sequence.
        """
        # Let's start by encoding the snippet sequences
        # 1. we embed the snippets independently
        # [batch_size, max_length, embed_dim]
        
        # We begin by embedding the tokens        
        # [batch_size, max_length, embed_dim]
        N = x.shape[0]
        S = x.shape[1]
        if not get_predictions:
            N, S, C, F, V, M = x.shape
            x = x.view(N*S, C, F, V, M)
            # actual lengths of each sequence in the batch is needed for packing
            e = self.joints_embed(x, keep_prob=keep_prob)
            # [batch_size, max_length, embed_dim]
            e = e.view(N, S, -1)
            if self.encoder is not None:
                # 2. and then encode them in their left-to-right and right-to-left context
                # [batch_size, max_length, 2*hidden_size] 
                e = self.positional_encoding(e.permute(1, 0, 2))
                src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
                for i in range(N):
                    src_mask[i, lengths[i]:] = True

                h = self.encoder(e, src_key_padding_mask=src_mask)
                try:
                    assert h.shape == (S, N, self.joints_embed_dim)
                except AssertionError as error:
                    print('the input shape is ' + str(e.shape))
                    print('the output shape is ' + str(h.shape))
                    print('The mask shape is ' + str(src_mask.shape))
                    print('The lengths array is ' + str(lengths))
                    print(S, N, self.joints_embed_dim)
                    print(error)
                    # halt execution
                    raise error
                h = h.permute(1, 0, 2)
                
                 # concatenate h and e, in the embedding dimension
                # [batch_size, max_length, 2*hidden_size + embed_dim]
                if self.include_embeddings:
                    u = torch.cat([e, h], 2)
                else:
                    u = h
                if get_embeddings:
                    return u
        # here we pad the tag sequence with BOS on the left
        # this is how we make sure that the current position of the tag sequence
        # is not available for conditioning (only the past is)
        batch_size, max_len = N, S
        bos = torch.full((batch_size, 1), self.bos, device=y.device)
        _y = torch.cat([bos, y], 1)
        # [batch_size, max_len, tag_emb_dim]
        t_in = self.label_embed(_y[:,:max_len])
       
        t_in = self.tag_positional_encoding(t_in.permute(1, 0, 2))
        src_mask = torch.zeros((N, S), dtype=torch.bool, device=u.device)
        for i in range(N):
            #TODO check if the lengths here are for the generated sequences, if we generate sequences
            src_mask[i, lengths[i]:] = True

        v = self.tag_encoder(t_in, src_key_padding_mask=src_mask)
        try:
            assert v.shape == (S, N, self.label_embed_dim)
        except AssertionError as error:
            print('the input shape is ' + str(e.shape))
            print('the output shape is ' + str(h.shape))
            print('The mask shape is ' + str(src_mask.shape))
            print('The lengths array is ' + str(lengths))
            print(S, N, self.joints_embed_dim)
            print(error)
            # halt execution
            raise error
        v = v.permute(1, 0, 2)
        # [batch_size, max_len, hidden_size]
        # # Now we can combine the encodings of x and the encodings of histories, we do so via concatenation
        # memory = torch.zeros(1, 1, self.hidden_size, device=y.device), torch.zeros(1, 1, self.hidden_size, device=y.device)        
        # for i in range(1, max_len + 1):
        #     v, memory = self.decoder(t_in[:,:i], memory)

        # Now we can combine the encodings of x and the encodings of histories, we do so via concatenation
        # since there's a fixed number of such encodings per step of the sequence
        # [batch_size, max_length, 3*hidden_size]
        u = torch.cat([u, v], -1)
        # We are now ready to map the state of each step of the sequence to a C-dimensional vector of logits
        # we do so using our FFNN
        # [batch_size, max_length, labelset_size]
        if get_predictions:
            # we are in the prediction mode, so we need to generate the sequence, 
            # and we do so by sampling from the distribution over the next tag
            # select the first l tokens of the sequence
            # l = lengths.max()
            # s = self.logits_predictor(u[:,:l])
            return self.logits_predictor(u)
        s = self.logits_predictor(u)
        if not eval:
            return s
            # return s.view(N*S, self.num_of_classes)
        else:
            cat_log_probs = td.Categorical(logits=s)
            return cat_log_probs.probs  

class MarkovLabeler(Labeler):
    def __init__(self, gcns_model, labeler_name: str, recurrent_encoder: bool, training_seq: str, labelset_size: int, pad_id=-1, labels=['outside', 'begin', 'inside', 'end', 'bos'], classes=['outside', 'begin', 'inside', 'end'], max_seq_len=40, ngram_size=3, joints_embed_dim=256, label_embed_dim=32, include_embeddings=False, hidden_size=128, **kwargs):
        """
        ngram_size: longest ngram (for tag sequence)
        labelset_size: number of known tags
        joints_embed_dim: dimensionality of snippet embeddings
        label_embed_dim: dimensionality of tag embeddings (needed to encode the history of ngram_size-1 tags)
        hidden_size: dimensionality of hidden layers
        """
        # we need to embed tags in the history 
        
        super().__init__(labeler_name, recurrent_encoder, training_seq, labelset_size, pad_id, labels, classes, max_seq_len, ngram_size, joints_embed_dim, label_embed_dim, include_embeddings, hidden_size, **kwargs)
        assert ngram_size > 1, "This class expects at least ngram_size 2. If you want ngram_size=1, use the NeuralUnigramLabeler"        
        self.joints_embed_dim = joints_embed_dim
        self.hidden_size = hidden_size
        
        # tagset is extended with a BOS tag, so it is the number of labels which are not bos
        # we need to embed snippets in x
        self.joints_embed = gcns_model
        # we need to embed tags in the history 
        self.label_embed = nn.Embedding(labelset_size, embedding_dim=label_embed_dim) 
        nn.init.normal_(self.label_embed.weight, 0, math.sqrt(2. / label_embed_dim))
        # we need to encode snippet sequences
        if recurrent_encoder:
            # use transformer
            nhead = 8
            num_layers = 4
            dim_feedforward = hidden_size * 4
            self.positional_encoding = PositionalEncoding(d_model=joints_embed_dim, dropout=0.1, max_len=100)
            encoder_layers = nn.TransformerEncoderLayer(d_model=joints_embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.encoder.apply(self.initialize_weights)

            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
            if include_embeddings:
                first_dim = (ngram_size - 1) * label_embed_dim + 2 * joints_embed_dim
            else:
                first_dim = (ngram_size - 1) * label_embed_dim + joints_embed_dim
            
            self.logits_predictor = nn.Sequential(
            nn.Linear(int(first_dim), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_of_classes),
            ) 
            # initialise the logits_predictor
        else:
            self.encoder = None
            # for each position i, we need to combine the encoding of x[i] in context 
            # as well as the history of ngram_size-1 tags
            # so we use a FFNN for that:
            self.logits_predictor = nn.Sequential(
                nn.Linear(joints_embed_dim + (ngram_size - 1) * label_embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_of_classes),
            )
        for layer in self.logits_predictor:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, math.sqrt(2. / hidden_size))                    
    def forward(self, x, y, eval=False, get_embeddings=False, get_predictions=False, u=None, lengths=None, keep_prob=0.9):
        """
        Parameterise the conditional distributions over Y[i] given history y[:i] and all of x.

        This procedure takes care that the ith output distribution conditions only on the n-1 observations before y[i].
        It also takes care of padding to the left with BOS symbols.

        x: snippet sequences [batch_size, max_length]
        y: tag sequences  [batch_size, max_length]

        Return: a batch of V-dimensional Categorical distributions, one per step of the sequence.
        """
        # Let's start by encoding the snippet sequences
        # 1. we embed the snippets independently
        # [batch_size, max_length, embed_dim]
        
        # We begin by embedding the tokens        
        # [batch_size, max_length, embed_dim]
        if not get_predictions:
            N, S, C, F, V, M = x.shape
            x = x.view(N*S, C, F, V, M)
            # actual lengths of each sequence in the batch is needed for packing
            e = self.joints_embed(x, keep_prob=keep_prob)
            # [batch_size, max_length, embed_dim]
            e = e.view(N, S, -1)
            if self.encoder is not None:
                # 2. and then encode them in their left-to-right and right-to-left context
                # [batch_size, max_length, 2*hidden_size] 
                e = self.positional_encoding(e.permute(1, 0, 2))
                src_mask = torch.zeros((N, S), dtype=torch.bool, device=e.device)
                for i in range(N):
                    src_mask[i, lengths[i]:] = True
                h = self.encoder(e, src_key_padding_mask=src_mask)
                try:
                    assert h.shape == (S, N, self.joints_embed_dim)
                except AssertionError as error:
                    print('the input shape is ' + str(e.shape))
                    print('the output shape is ' + str(h.shape))
                    print('The mask shape is ' + str(src_mask.shape))
                    print('The lengths array is ' + str(lengths))
                    print(S, N, self.joints_embed_dim)
                    print(error)
                    # halt execution
                    raise error
                h = h.permute(1, 0, 2)
               
                 # concatenate h and e, in the embedding dimension
                # [batch_size, max_length, 2*hidden_size + embed_dim]
                if self.include_embeddings:
                    u = torch.cat([e, h], 2)
                else:
                    u = h
            else:
                u = e
        if get_embeddings:
            return u

        # Let's now encode the history of ngram_size-1 tags
        # 1. create a sequence of BOS symbols to be prepended to y.
        # [batch_size, ngram_size - 1]
        bos = torch.full((y.shape[0], self.ngram_size - 1), self.bos, device=y.device)
        # 2. preprend it to y
        # # [batch_size, max_length + ngram_size - 1]
        _y = torch.cat([bos, y], 1)
        # 3. for each output step, we will have ngram_size - 1 inputs, so we collect those from y
        # [batch_size, max_length, ngram_size - 1]
        history = torch.cat([_y.unsqueeze(-1)[:,i:i+self.ngram_size-1].reshape(y.shape[0], 1, -1) for i in range(y.shape[1])], 1)
        # 4. embed the tags in the history
        # [batch_size, max_length, ngram_size - 1, tag_emb_dim]
        history = self.label_embed(history)
        # 5. concatenate the embeddings for the tags in the history
        # [batch_size, max_length, (ngram_size - 1) * tag_emb_dim]
        history = history.reshape(y.shape + (-1,)) 

        # Now we can combine the encodings of x and the encodings of histories, we do so via concatenation
        # since there's a fixed number of such encodings per step of the sequence
        # [batch_size, max_length, 2*hidden_size + (ngram_size - 1) * tag_emb_dim]

        u = torch.cat([u, history], -1)
        # We are now ready to map the state of each step of the sequence to a C-dimensional vector of logits
        # we do so using our FFNN
        # [batch_size, max_length, num_of_labels]
        # if get_predictions:
        #     l = lengths.max()
        #     u = u[:, :l]
        s = self.logits_predictor(u)
        return s
        # if not eval:
        #     return s.view(N*S, self.num_of_labels)
        # else:
        #     cat_log_probs = td.Categorical(logits=s)
        #     return cat_log_probs.probs    