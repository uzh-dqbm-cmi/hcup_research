'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
### Training model using batches of sequences that are packed before passing to RNN network 
### The whole sequences are passed
##################################

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utilities import get_tensordtypes



class RNN_Labeler(nn.Module):
    def __init__(self, input_dim, hidden_dim, y_codebook, 
                 embed_dim=0, interm_dim=0, num_hiddenlayers=1, 
                 bidirection= False, pdropout=0., rnn_class=nn.LSTM, 
                 nonlinear_func=F.relu, to_gpu=True):
        super(RNN_Labeler, self).__init__() 
        self.fdtype, self.ldtype = get_tensordtypes(to_gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.y_codebook = y_codebook
        self.tagset_size = len(y_codebook)
        self.num_hiddenlayers = num_hiddenlayers
        self.pdropout = pdropout
        if(embed_dim): # embed layer
            self.embed = nn.Linear(self.input_dim, embed_dim)
            self.rnninput_dim = embed_dim
        else:
            self.embed = None
            self.rnninput_dim = self.input_dim
        self.rnn = rnn_class(self.rnninput_dim, hidden_dim, num_layers=num_hiddenlayers, 
                             dropout=pdropout, bidirectional=bidirection, batch_first=True)
        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1
        if(interm_dim): 
            # intermediate layer embedding after rnn
            self.intermlayer = nn.Linear(self.num_directions*hidden_dim, interm_dim) 
            self.hiddenTotag = nn.Linear(interm_dim, self.tagset_size)
        else:
            self.intermlayer = None
            self.hiddenTotag = nn.Linear(self.num_directions*hidden_dim, self.tagset_size)
        self.nonlinear_func = nonlinear_func       

        
    def init_hidden(self, batch_size):
        """initialize hidden vectors at t=0
        
        Args:
            batch_size: int, the size of the current evaluated batch
        """
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)
        h0=autograd.Variable(torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype))
        if(isinstance(self.rnn, nn.LSTM)):
            c0=autograd.Variable(torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype))
            hiddenvec = (h0,c0)
        else:
            hiddenvec = h0
        return(hiddenvec)
    
    def forward(self, batch_seqs, seqs_len, tagprob=False):
        """ perform forward computation
        
            Args:
                batch_seqs: Variable of shape (batch, seqlen, input_dim)
                seqs_len: torch Tensor comprising length of the sequences in the batch
                tagprob: bool indicating if probability over class labels is required (default:False)
        """
        if(self.embed):
            batch_seqs = self.nonlinear_func(self.embed(batch_seqs))
        # init hidden
        hidden = self.init_hidden(batch_seqs.size(0))
        # pack the batch
        packed_embeds = pack_padded_sequence(batch_seqs, seqs_len.cpu().numpy(), batch_first=True)
        packed_rnn_out, hidden = self.rnn(packed_embeds, hidden)

        # we need to unpack sequences
        unpacked_output, out_seqlen = pad_packed_sequence(packed_rnn_out, batch_first=True)

        if(self.intermlayer):
            intermediate_res = self.nonlinear_func(self.intermlayer(unpacked_output))
        else:
            intermediate_res = unpacked_output
        tag_space = self.hiddenTotag(intermediate_res).view(-1, self.tagset_size)
        
        tag_prob = None
        if(tagprob):
            tag_prob = F.softmax(tag_space, dim=1)
        tag_scores = F.log_softmax(tag_space, dim=1)

#         print("packed_embeds \n", packed_embeds)
#         print()
#         print("packed_rnn_out \n", packed_rnn_out)
#         print()
#         print("unpacked output \n", unpacked_output)
#         print("unpacked output seqlen \n", out_seqlen)
#         print()
#         print("tag_space \n", tag_space)
#         print()
#         print("tag_prob \n", tag_prob)
#         print("tag_scores log probability \n", tag_scores)
#         print()
        return(tag_scores, tag_prob)
