'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
##################################
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from utilities import get_tensordtypes


class RNNSS_Labeler(nn.Module):
    def __init__(self, input_dim, hidden_dim, y_codebook, 
                 embed_dim=0, interm_dim=0, num_hiddenlayers=1, 
                 bidirection= False, pdropout=0., rnn_class=nn.LSTM, 
                 nonlinear_func=F.relu, startstate_symb="__START__", to_gpu=True):
        
        super(RNNSS_Labeler, self).__init__() 
        self.fdtype, self.ldtype = get_tensordtypes(to_gpu)
        self.hidden_dim = hidden_dim
        self.y_codebook = y_codebook
        self.tagset_size = len(y_codebook)
        self.startstate_symb = startstate_symb
        self.input_dim = input_dim + self.tagset_size # use the previous label as input 
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
            # intermediate layer between hidden and tag layer
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
        h0=autograd.Variable(torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim)).type(self.fdtype)
        if(isinstance(self.rnn, nn.LSTM)):
            c0=autograd.Variable(torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim)).type(self.fdtype)
            hiddenvec = (h0,c0)
        else:
            hiddenvec = h0
        return(hiddenvec)
    
    def concat_prevlabel(self, seq_t, chosen_ind):
        """ concat previous label y_{t-1} to the current input x_{t}
        
            TODO:
                update the assignment using advanced indexing --- it is not supported in pytorch version <=3
        """
        dim1, dim2, __ = seq_t.size() # 1*1*input_dim
        z = torch.zeros(dim1,dim2,self.tagset_size).type(self.fdtype)
        for indx in chosen_ind:
            z[0,0,indx[0]] = 1
        seq_t = torch.cat((seq_t, z), dim=-1)
        return(seq_t)
        
    def curriculum_learning(self, e_i, batch_size, pred, ref):
        pred = pred.unsqueeze(-1)
        ref = ref.unsqueeze(-1)
        ycat = torch.cat([pred, ref], dim=1)
#         print("ycat \n", ycat)
        berprob = torch.bernoulli(torch.Tensor(batch_size,1).fill_(e_i)).type(self.ldtype) # bernoulli prob.
#         print("bernoulli outcome \n", berprob)
        chosen_tagind =ycat.gather(1,berprob).type(self.ldtype)
#         print("chosen indices \n", chosen_tagind)
        return(chosen_tagind)
    
    def forward_t(self, seq_t, hidden, tagprob=True):
        """ apply forward computation at time t
            
            Args:
                seq_t: tensor of size (1,1,inputdim)
                hidden: Variable representing hidden vector at t-1
                tagprob: bool indicating if probability over class labels is required (default:False)
        """
        seq_t = autograd.Variable(seq_t)
        if(self.embed):
            seq_t = self.nonlinear_func(self.embed(seq_t))
        rnn_out, hidden = self.rnn(seq_t, hidden)
        if(self.intermlayer):
            intermediate_res = self.nonlinear_func(self.intermlayer(rnn_out))
        else:
            intermediate_res = rnn_out
            
        tag_space = self.hiddenTotag(intermediate_res).view(-1, self.tagset_size)
        
        tag_prob = None
        if(tagprob):
            tag_prob = F.softmax(tag_space, dim=1)
        tag_scores = F.log_softmax(tag_space, dim=1)

#         print("tag_space \n", tag_space)
#         print()
#         print("tag_prob \n", tag_prob)
#         print("tag_scores log probability \n", tag_scores)
#         print()
        return(tag_scores, tag_prob, hidden)
    
    def forward(self, seq, labels = None, e_i = 0, tagprob=True):
        """ perform forward computation
        
            Args:
                seq: torch Tensor of shape (1, seqlen, input_dim)
                
            Keyword args:
                labels: torch Tensor of shape (seqlen,) representing labels/tags
                e_i: float, probability of choosing ground-truth/reference label
                tagprob: boolean, indicator to return tags probability
                
        """
        batch_size = 1
        # initialize hidden vector
        hidden = self.init_hidden(batch_size)
        T = seq.size(1)
        tag_logprobs = autograd.Variable(torch.zeros(1, self.tagset_size).type(self.fdtype))
        
        tag_probs = None
        if(tagprob):
            tag_probs = autograd.Variable(torch.zeros(1, self.tagset_size).type(self.fdtype))
            
        predtag_indxs = torch.zeros(1).type(self.ldtype)
        for tstep in range(T):
#             print("tstep: ", tstep)
            current_seq_t = seq[0, tstep, :].view(1,1,-1)
#             print("current_seq_t \n", current_seq_t) 
            if(tstep>0):
                # update the input at time t with either predicted or reference output at t-1
                if(isinstance(labels, torch.Tensor)):
                    chosen_ind = self.curriculum_learning(e_i, batch_size, predtag_indx, labels[tstep-1])
                else:
                    chosen_ind =  predtag_indx.unsqueeze(-1)
            else:
                # concat the label at t-1 to the original input 
                chosen_ind = torch.Tensor([self.y_codebook[self.startstate_symb]]).type(self.ldtype).view(1,1)
            current_seq_t = self.concat_prevlabel(current_seq_t, chosen_ind)
#             print("updated input_seq_t \n", current_seq_t)
            tag_logprobs_t, tag_probs_t, hidden = self.forward_t(current_seq_t, hidden, tagprob=tagprob)
            # get the argmax of tag_logprobs_t (i.e. the index of highest scoring tags)
            predtag_scores, predtag_indx = torch.max(tag_logprobs_t, dim=1)
#             print("tag_logprobs_t \n", tag_logprobs_t)
#             print("max log prob scores \n", predtag_scores)

            # keep only tensor -- no need for variable
            predtag_indx = predtag_indx.data
#             print("indices of max log prob scores \n", predtag_indx)
            tag_logprobs = torch.cat((tag_logprobs, tag_logprobs_t),dim=0)
            if(tagprob):
                tag_probs = torch.cat((tag_probs, tag_probs_t), dim=0)
            predtag_indxs = torch.cat((predtag_indxs, predtag_indx), dim=0)
#         print("tag_logprobs \n", tag_logprobs)
#         print("predtag_indxs \n", predtag_indxs)
        if(tagprob):
            tag_probs = tag_probs[1:]
        return(tag_logprobs[1:], tag_probs, predtag_indxs[1:])
    
if __name__ == '__main__':
    pass