'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from utilities import logsumexp_var, max_argmax_var, get_tensordtypes


class RNNCRF_Unary_Labeler(nn.Module):
    def __init__(self, input_dim, hidden_dim,  y_codebook,
                 embed_dim=0, interm_dim=0, num_hiddenlayers=1,
                 bidirection= False, pdropout=0., rnn_class=nn.LSTM,
                 nonlinear_func=F.relu, startstate_symb="__START__",
                 stopstate_symb=None, to_gpu=True):
        
        super(RNNCRF_Unary_Labeler, self).__init__()
        self.fdtype, self.ldtype = get_tensordtypes(to_gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(y_codebook)
        self.y_codebook = y_codebook
        self.num_hiddenlayers = num_hiddenlayers
        self.pdropout = pdropout
        self.neginf = -100000
        
        if(embed_dim): # embed layer
            self.embed = nn.Linear(self.input_dim, embed_dim)
            self.rnninput_dim = embed_dim
        else:
            self.embed = None
            self.rnninput_dim = self.input_dim
        self.rnn = rnn_class(self.rnninput_dim, hidden_dim, num_layers= num_hiddenlayers,
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
        self._check_ycodebook(startstate_symb, stopstate_symb)
        
    def _check_ycodebook(self, startstate_symb, stopstate_symb):
        y_codebook = self.y_codebook
        if(startstate_symb not in y_codebook):
            raise(Exception('START state must be defined in y_codebook'))
        self.startstate_symb = startstate_symb
        
        if(stopstate_symb == None):
            self.stopstate_symb = stopstate_symb
        elif(stopstate_symb not in y_codebook):
            raise(Exception('STOP state must be defined in y_codebook'))
        else:
            self.stopstate_symb = stopstate_symb

        self.y_transparams = torch.randn(self.tagset_size, self.tagset_size).type(self.fdtype)
        self.y_transparams[:, y_codebook[startstate_symb]] = self.neginf
        if(self.stopstate_symb):
            self.y_transparams[y_codebook[stopstate_symb],:] = self.neginf
        self.y_transparams = nn.Parameter(self.y_transparams)
        self.y_codebookrev = {code:label for label, code in self.y_codebook.items()}
        
    def init_hidden(self, batch_size):
        """ initialize hidden vectors at t=0
        
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
    
    def compute_potential(self, seq):
        """ compute unary potential \phi(x_t, y_t)
            
            Args:
                seq: sequence, variable of shape (batch, seqlen, input_dim)
                
        """        
        if(self.embed): # in case an embed layer is defined
            seq = self.nonlinear_func(self.embed(seq))
        hidden = self.init_hidden(seq.size(0)) # initialize hidden vectors by sending the batch size
        rnnout, hidden = self.rnn(seq, hidden)

        if(self.intermlayer): # in case intermediate layer is defined
            intermres = self.nonlinear_func(self.intermlayer(rnnout))
            feat = self.hiddenTotag(intermres).view(-1,self.tagset_size)
                
        else:
            feat = self.hiddenTotag(rnnout).view(-1,self.tagset_size)
        # feat will have a shape (seqlen, tagset_size)
#         print("feat \n", feat)
        feat = torch.unsqueeze(feat, 1)
        return(feat)
    
    def compute_forward_vec(self, seq_potential):
        """ compute alpha matrix
        
            Args:
                seq_potential: computed potential at every time step using :func:`self.compute_potential`.
                               It has the shape (seqlen, 1, tagset_size)
                     
        """
#         print("we are in compute_forward_vec")
        y_codebook = self.y_codebook
        T = seq_potential.size(0)
        
        offset = 1
        feat_score = seq_potential + self.y_transparams # feature score
        if(self.stopstate_symb): 
            offset = 2
            
        num_rows = T+offset
        # create alpha matrix
        alpha = torch.Tensor(num_rows, self.tagset_size).fill_(self.neginf).type(self.fdtype)
        alpha[0, y_codebook[self.startstate_symb]] = 0.
        alpha = autograd.Variable(alpha)
#         print("alpha matrix: \n", alpha)
        for t in range(T):
            score = alpha[t,:].view(-1,1) + feat_score[t,:,:]
            log_score = logsumexp_var(score, dim=0)
            alpha[t+1,:] = log_score
        if(self.stopstate_symb):
            t = T
            score = alpha[t,:].view(-1,1) + self.y_transparams
            log_score = logsumexp_var(score, dim=0)
            alpha[t+1,:] = log_score
            Z = alpha[t+1, y_codebook[self.stopstate_symb]] # partition function
        else:
            t = T
            Z = logsumexp_var(alpha[t, :].view(-1,1), dim=0)[0] # compute partition function Z
#         print("alpha \n", alpha)
#         print("Z: ", Z)
        return(Z, alpha)
    
    def compute_backward_vec(self, seq_potential):
        """ compute beta matrix
        
            Args:
                seq_potential: computed potential at every time step using :func:`self.compute_potential`.
                               It has the shape (seqlen,1,tagset_size)
                     
        """
#         print("we are in compute_backward_vec")        
        y_codebook = self.y_codebook
        T = seq_potential.size(0)
            
        offset = 1
        feat_score = seq_potential + self.y_transparams
        if(self.stopstate_symb):
            offset = 2
            
        num_rows = T+offset            
        # create beta matrix
        beta = torch.Tensor(num_rows, self.tagset_size).fill_(self.neginf).type(self.fdtype)
        if(self.stopstate_symb):
            beta[-1, y_codebook[self.stopstate_symb]] = 0
            beta = autograd.Variable(beta)
            t = T
            score = beta[t+1,:].view(1,-1) + self.y_transparams
            log_score = logsumexp_var(score, dim=1)
            beta[t,:] = log_score
        else:
            beta[-1,:] = 0
            beta = autograd.Variable(beta)
            
#         print("beta matrix: \n", beta)
        for t in reversed(range(0, T)):
            score = beta[t+1,:].view(1,-1) + feat_score[t,:,:]
            log_score = logsumexp_var(score, dim=1)
            beta[t,:] = log_score
        Z = beta[0, y_codebook[self.startstate_symb]]
#         print("beta \n", beta)
#         print("Z: ", Z)
        return(Z, beta)
    
    def compare_alpha_beta(self, seq_potential):
        """ sanity check function for comparing the alpha/beta computation
        """
        Z_alpha, alpha = self.compute_forward_vec(seq_potential)
        print("alpha: \n", alpha)
        print("Z_alpha: ", Z_alpha)
        Z_beta, beta = self.compute_backward_vec(seq_potential)
        print("beta: \n", beta)
        print("Z_beta: ", Z_beta)  
        print("abs diff: ", torch.abs(Z_alpha-Z_beta))
        P_marginal = self.compute_marginal_prob(alpha, beta, Z_alpha)
        print("P_marginal: \n", P_marginal)
        print("Sum P_marginal: \n", torch.sum(P_marginal, dim=1)) # this should sum to 1 if everything is implemented correctly
        
    def compute_marginal_prob(self, alpha, beta, Z):
        """ compute the marginal probability of every state/label at each timestep
        """
        # create marginal probability matrix        
        P_marginal = alpha + beta - Z
        P_marginal = torch.exp(P_marginal)
#         print("P_marginal: \n", P_marginal)    
#         print("Sum P_marginal: \n", torch.sum(P_marginal, dim=1))
        return(P_marginal)
        
    def compute_refscore(self, seq_potential, y_labels):
        """ computes the probability of reference label sequence Y given observation sequence X (i.e. P(Y|X))
            
            Args:
                seq_potential: computed sequence potential of shape (seqlen,1,tagset_size)
                y_labels: encoded reference/gold-truth labels of the sequence (i.e. [1,4,3,2,2])
        """
#         print("we are in compute_refscore")
#         assert T == len(y_labels)

        refscore = autograd.Variable(torch.zeros(1).type(self.fdtype), requires_grad=True)
        T = seq_potential.size(0)
        feat_score = seq_potential + self.y_transparams
                    
        for t in range(T):
            if(t>0):
                prev_tag = y_labels[t-1]
            else:
                prev_tag = self.y_codebook[self.startstate_symb]
            curr_tag = y_labels[t]
#             print("curr_tag ", curr_tag)
#             print("prev_tag ", prev_tag)  
            refscore = refscore + feat_score[t, prev_tag, curr_tag]
        if(self.stopstate_symb):
            t = T
            prev_tag = y_labels[-1]
            curr_tag = self.y_codebook[self.stopstate_symb]
            transition_score = self.y_transparams
            refscore = refscore + transition_score[prev_tag, curr_tag]
#         print("refscore \n", refscore)
        return(refscore)
    
    def decode(self, seq_potential, labels=[], method='viterbi'):
        if(method == 'viterbi'):
            return(self.decode_viterbi(seq_potential))
        elif(method == 'posterior_prob'):
            return(self.decode_posteriorprob(seq_potential))
        elif(method == 'guided_viterbi'):
            return(self.decode_guided(seq_potential, labels))
        
    def decode_posteriorprob(self, seq_potential):
        """ decode a sequence using posterior probability(i.e. using marginal probability of each state at every timestep)
        
            Args:
                seq_potential: computed sequence potential of shape (seqlen,tagset_size,tagset_size)             
        """
        Z_alpha, alpha = self.compute_forward_vec(seq_potential)
        Z_beta, beta = self.compute_backward_vec(seq_potential)
        P_marginal = self.compute_marginal_prob(alpha, beta, Z_alpha)
        Y_score, Y_decoded = torch.max(P_marginal, dim=1)
        # taking out the _START_ state
        target_y = Y_decoded.data[1:].tolist()
        target_score = Y_score.data[1:].tolist()
        score_mat = P_marginal.data[1:,:]
        if(self.stopstate_symb): # taking out _STOP_ state
            target_y = Y_decoded.data[1:-1].tolist()
            target_score = Y_score.data[1:-1].tolist()
            score_mat = P_marginal.data[1:-1, :]
        
        if(not self.map_dict):
            Y_decoded = target_y
        else:
            Y_decoded = self._map_target(target_y)
            
        Y_score = target_score
        return(Y_decoded, Y_score, Z_alpha, score_mat)
    
    def _map_target(self, target_y):
        Y_decoded = [self.map_dict[self.y_codebookrev[y_code]] for y_code in target_y]
        return(Y_decoded)
    
    def decode_viterbi(self, seq_potential):
        """ decode a sequence of vector observations to produce the output sequence (X -> Y) using viterbi decoder
        
            Args:
                seq_potential: computed sequence potential of shape (seqlen,1,tagset_size)
                
            TODO:
                - further cleaning and optimization                
        """
#         print("we are in decode")

        y_codebook = self.y_codebook
        T = seq_potential.size(0)
            
        offset = 1
        feat_score = seq_potential + self.y_transparams
        if(self.stopstate_symb): 
            offset = 2
            
        num_rows = T+offset

        # create score matrix
        score_mat = autograd.Variable(torch.Tensor(num_rows, self.tagset_size).fill_(self.neginf)).type(self.fdtype)
        # back pointer to hold the index of the state that achieved highest score while decoding
        backpointer = torch.Tensor(num_rows, self.tagset_size).fill_(-1).type(self.ldtype)
        score_mat[0, y_codebook[self.startstate_symb]] = 0
        for t in range(T):
            score = score_mat[t,:].view(-1,1) + feat_score[t,:,:]
            bestscores, bestscore_tags = max_argmax_var(score, dim=0)
            backpointer[t+1, :] = bestscore_tags.data
            score_mat[t+1,:] = bestscores
#         print("score_matrix \n", score_mat)
#         print("backpointer \n", backpointer)

        if(self.stopstate_symb):
            t = T
            score = score_mat[t,:].view(-1,1) + self.y_transparams
            bestscores, bestscore_tags = max_argmax_var(score, dim=0)
            backpointer[t+1, :] = bestscore_tags.data
            score_mat[t+1,:] = bestscores
            bestscore_tag = backpointer[t+1, y_codebook[self.stopstate_symb]]
            optimal_score = score_mat[t+1, y_codebook[self.stopstate_symb]].data[0]
        else:
            t = T
            bestscore, bestscore_tag = max_argmax_var(score_mat[t,:].view(-1,1), dim=0)
            bestscore_tag = bestscore_tag.data[0]
            optimal_score = bestscore.data[0]
            
#         print("score_matrix \n", score_mat)
#         print("backpointer \n", backpointer)
#         print("bestscore_tag: \n", bestscore_tag)
#         print("optimal_score: \n", optimal_score)
        

        Y_decoded, Y_score = self._traverse_optimal_seq(score_mat, backpointer, bestscore_tag, T)
        if(self.map_dict):
            Y_decoded = self._map_target(Y_decoded)
        norm_scoremat = self._normalize_scoremat(score_mat)
        return(Y_decoded, Y_score, optimal_score, norm_scoremat)
    
    def compute_normalized_alpha(self, seq_potential):
        Z, alpha = self.compute_forward_vec(seq_potential)
        scoremat = self._normalize_scoremat(alpha)
        return(scoremat)
    
    def _normalize_scoremat(self,score_mat):
        if(self.stopstate_symb):
            scoremat = score_mat[1:-1,:]
        else:
            scoremat = score_mat[1:,:]
        # normalize score_mat
        exp_scoremat  = torch.exp(scoremat)
        norm_scoremat =  exp_scoremat/torch.sum(exp_scoremat, dim=1, keepdim=True)
        return(norm_scoremat.data)
    
    def _traverse_optimal_seq(self, score_mat, backpointer, bestscore_tag, T):
        """decoding a sequence based on backpointer and score matrices
        """
        Y_decoded = [bestscore_tag]
        Y_score = [score_mat[T, bestscore_tag].data[0]]
        counter = 0
#         print("Y_decoded: \n", Y_decoded)
        for t in reversed(range(2, T+1)):
#             print("t: ", t)
#             print("backpointer[{}, {}]={}".format(t, Y_decoded[counter], backpointer[t, Y_decoded[counter]]))
            btag = backpointer[t, Y_decoded[counter]]
#             print("btag: ", btag)
            Y_decoded.append(btag)
            Y_score.append(score_mat[t-1, btag].data[0])
            counter += 1
#             print(btag)
        Y_decoded.reverse()
        Y_score = Y_score[::-1]
#         print("Y_score: \n", Y_score)
#         print("Y_decoded: \n", Y_decoded)

        return(Y_decoded, Y_score)
    
    def decode_guided(self, seq_potential, labels):
        """ decode **only** the last event of a sequence using viterbi decoder
        
            Args:
                seq_potential: computed sequence potential of shape (seqlen, 1,tagset)
            
            TODO:
                - implement this method !!                
        """
        raise(Exception("decode_guided is not implemented yet !!"))

    
    def compute_loss(self, seq, labels, mode='forward_vectors'):
        """ compute CRF loss
        
            Args:
                seq: variable of size (batch,seqlen,input_dim)
                labels: long tensor corresponding to the labels/tags id
        """
        seq_potential = self.compute_potential(seq)
        if(mode == 'forward_vectors'):
            expected_score, __ = self.compute_forward_vec(seq_potential)
        elif(mode == 'structured_perceptron'):
            __, __, expected_score, __  = self.decode(seq_potential, method='viterbi')
        refscore  = self.compute_refscore(seq_potential, labels)
        diff = expected_score - refscore
        return(diff)

    def compute_loss_seqpot(self, seq_potential, labels, mode='forward_vectors'):
        """ compute CRF loss using sequence potential as input
            (i.e. sequence was processed using :func:`self.compute_potential`)
        
            Args:
                seq_potential: computed potential at every time step using :func:`self.compute_potential`.
                               It has the shape (seqlen, 1, tagset_size)
                labels: long tensor corresponding to the labels/tags id
        """
        if(mode == 'forward_vectors'):
            expected_score, __ = self.compute_forward_vec(seq_potential)
        elif(mode == 'structured_perceptron'):
            __, __, expected_score, __  = self.decode(seq_potential, method='viterbi')
        refscore  = self.compute_refscore(seq_potential, labels)
#         print(expected_score)
#         print(expected_score.requires_grad)
        diff = expected_score - refscore
        return(diff)
    
    def forward(self, seq, method='viterbi'):
        """ perform forward pass
        
            Args:
                seq: Variable of shape (1,seqlen,input_dim)
                
        """
        seq_potential = self.compute_potential(seq)
        y_decoded, y_score, optimalscore, score_mat = self.decode(seq_potential, method=method)
        return(y_decoded, y_score, optimalscore, score_mat)
    
if __name__ == "__main__":
    pass

