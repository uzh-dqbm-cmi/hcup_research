'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
import os
import pickle
from collections import namedtuple
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, \
                            precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
dataset_dir = os.path.join(project_dir, 'dataset')

PADDSYMB_INDX = 1000
IS_CUDA = torch.cuda.is_available()

# number of workers in dataloader 
NUM_WORKERS = 0
# if(os.name == 'nt'): # windows os
#     NUM_WORKERS = 0
# else:
#     NUM_WORKERS = 5

mp = mp.get_context('spawn') # setup context 
NUM_PROCESSORS = mp.cpu_count() - 1

def get_tensordtypes(to_gpu):
    fdtype = torch.FloatTensor
    ldtype = torch.LongTensor
    if(IS_CUDA and to_gpu):
        fdtype = torch.cuda.FloatTensor
        ldtype = torch.cuda.LongTensor
    return(fdtype, ldtype)

def max_argmax_var(var, dim=0):
    """ return the index of maximum value in a variable as a scalar
    """
    maxscore, maxscore_indx = torch.max(var, dim=dim)
    return(maxscore, maxscore_indx)


def argmax_var(var, dim=0):
    """ return the index of maximum value in a variable as a scalar
    """
    __, maxscore_indx = torch.max(var, dim=dim)
    return(maxscore_indx)

def logsumexp_var(var, dim=0):
    if(dim == 0):
        view_a = 1
        view_b = -1
    else:
        view_a = -1
        view_b = 1
    max_score = var.gather(dim, argmax_var(var, dim=dim).view(view_a, view_b))
#     print("max_score: ", max_score)
#     print("var-max_score: \n", var - max_score)
#     print("torch.exp(var-max_score): \n", torch.exp(var - max_score))
#     print("torch.sum(torch.exp(var - max_score)):\n", torch.sum(torch.exp(var - max_score), dim=dim, keepdim=True))
#     print("max_score + torch.log(torch.sum(torch.exp(var - max_score), dim=dim, keepdim=True)): \n", max_score + torch.log(torch.sum(torch.exp(var - max_score), dim=dim, keepdim=True)))
    return(max_score + torch.log(torch.sum(torch.exp(var - max_score), dim=dim, keepdim=True)))


def compute_numtrials(prob_interval_truemax, prob_estim):
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)

class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass
    @staticmethod
    def dump_data(data, file_name, mode = "wb"):
        """dump data by pickling 
        
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f) 
    @staticmethod
    def read_data(file_name, mode = "rb"):
        """read dumped/pickled data
        
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)
    
    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
        
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)
    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
        
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line
                
def create_directory(folder_name, directory = "current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
    
       Args:
           folder_name: string representing the name of the folder to be created
       
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)

 
VectorizedData = namedtuple('VectorizedData', ['seqs_tensor', 'labels_tensor', 'seqslen_tensor', 'indexevent_tensor'])

class PatientSeqDataset(Dataset):
    """A class representing a Dataset composed of sequences
      
       Args:
           pdt_object: instance of :class:`PatientDataTensor`
           to_gpu: bool, check if to copy tensors to gpu device
      
    """
    def __init__(self, pdt_object, to_gpu=True):
        self.num_samples = pdt_object.num_samples
        self.vecdata = self.to_torchtensor(pdt_object, to_gpu)
         
    def to_torchtensor(self, pdt_object, to_gpu):
        """turn data that is instance of :class:`PatientDataTensor` into torch Dataset
          
           Args:
               data: instance of :class:`PatientDataTensor`
          
        """
        fdtype, ldtype = get_tensordtypes(to_gpu)
        seqs_tensor = torch.from_numpy(pdt_object.seq_tensor[:self.num_samples]).type(fdtype) # double tensor representing the features
        labels_tensor = torch.from_numpy(pdt_object.label_tensor[:self.num_samples]).type(ldtype).squeeze(-1)
        indexevent_tensor = torch.from_numpy(pdt_object.indexevent_tensor[:self.num_samples]).type(ldtype).squeeze(-1)
        seqslen_tensor = torch.from_numpy(pdt_object.seqlen_tensor[:self.num_samples]).type(ldtype).squeeze(-1)
        vecdata = VectorizedData(seqs_tensor, labels_tensor, seqslen_tensor, indexevent_tensor)
        return vecdata
      
    def __getitem__(self, indx):
        vecdata = self.vecdata
        return(vecdata.seqs_tensor[indx], vecdata.labels_tensor[indx], vecdata.seqslen_tensor[indx], vecdata.indexevent_tensor[indx])
  
    def __len__(self):
        return(self.num_samples)
    

class PatientDataTensorMemmap:
    def __init__(self, idx_mapper, dtype, paddsymbol):
        self.idx_mapper = idx_mapper
        self.idx_mapper_inverse = {numcode:pid for pid, numcode in self.idx_mapper.items()}
        self.dsettype = dtype
        self.paddsymbol = paddsymbol
        
    def memmap_arrays(self, X_tensor, Y_tensor, E_tensor, T_tensor, fpath):
        # create a memmap numpy arrays
        tensor_info = ['seqtensor_info', 'labeltensor_info',
                       'indexeventtensor_info', 'seqlentensor_info']
        arrays = [X_tensor, Y_tensor, E_tensor, T_tensor]
        array_names = ['seq_tensor', 'label_tensor', 'indexevent_tensor', 'seqlen_tensor']
        for i, (arr, arr_name) in enumerate(zip(arrays, array_names)):
            tmparr = np.memmap(os.path.join(fpath, arr_name+'.dat'), dtype=arr.dtype, mode='w+', shape=arr.shape)
            tmparr[:] = arr[:]
            setattr(self, tensor_info[i], (arr.dtype, arr.shape))
        self.num_samples = X_tensor.shape[0]
        self.input_dim = X_tensor.shape[-1]

    def read_fromdisk(self, fpath, memmap = True):
        # to refactor this function/process
        # due to issues with multiprocessing and size of seq_tensor
        # loading seq_tensor is delayed until it is called within the spawned child processes
        array_names = ['seq_tensor', 'label_tensor', 'indexevent_tensor', 'seqlen_tensor']
        tensor_info = [self.seqtensor_info, self.labeltensor_info, 
                       self.indexeventtensor_info, self.seqlentensor_info]
        for arr_info, arr_name in zip(tensor_info, array_names):
            arr = np.memmap(os.path.join(fpath, arr_name+'.dat'), dtype=arr_info[0], mode = 'r', shape=arr_info[1])
            if(not memmap):
                arr = np.asarray(arr)
            setattr(self, arr_name, arr)

        if(hasattr(self, 'fpath')):
            del self.fpath

            
class PatientSeqDatasetMemmap(Dataset):
    """A class representing a Dataset composed of sequences
      
       Args:
           pdt_object: instance of :class:`PatientDataTensorMemmap`
           to_gpu: bool, check if to copy tensors to gpu device
      
    """
    def __init__(self, pdt_object, to_gpu=True):
        self.pdt_object = pdt_object
        self.num_samples = pdt_object.num_samples
        self.fdtype, self.ldtype = get_tensordtypes(to_gpu)
        
    def _to_torchtensor(self, indx):
        """turn data that is instance of :class:`PatientDataTensor` into torch Dataset
          
           Args:
               data: instance of :class:`PatientDataTensor`
          
        """
        pdt_object = self.pdt_object
        fdtype, ldtype = self.fdtype, self.ldtype
        seqs_tensor = torch.from_numpy(pdt_object.seq_tensor[:self.num_samples][indx]).type(fdtype) # double tensor representing the features
        labels_tensor = torch.from_numpy(pdt_object.label_tensor[:self.num_samples][indx]).type(ldtype).squeeze(-1)
        indexevent_tensor = torch.from_numpy(pdt_object.indexevent_tensor[:self.num_samples][indx]).type(ldtype).squeeze(-1)
        seqslen_tensor = torch.from_numpy(np.array([pdt_object.seqlen_tensor[:self.num_samples][indx]])).type(ldtype).squeeze(-1)
        return(seqs_tensor, labels_tensor, seqslen_tensor, indexevent_tensor, indx)
    
    def __getitem__(self, indx):
        sample = self._to_torchtensor(indx)
        return(sample)
    
    def __len__(self):
        return(self.num_samples)
   
def restrict_grad_(mparams, mode, limit):
    """clamp/clip a gradient in-place
    """
    if(mode == 'clip_norm'):
        __, maxl = limit
        nn.utils.clip_grad_norm(mparams, maxl, norm_type=2) # l2 norm clipping
    elif(mode == 'clamp'): # case of clamping
        minl, maxl = limit
        for param in mparams:
            if param.grad is not None:
                param.grad.data.clamp_(minl, maxl)


def generate_operation_pipeline_cnn(input_shape):
    # convolution operation
    # Input ->[[Conv->BatchNorm?->NonLinFunc->Dropout?]*I->Pool?]*J
    C, H, W = input_shape # C: channel index, H: y-dimension of the matrix, W: x-dimension of the matrix
    J_loop = range(7,8)
    pool_options = ('MaxPool', 'AvgPool')
    I_loop = range(1, 4)
    kernel_sizes = (3,)
    batchnorm_options = (1, 0)
    nonlin_funcs = (nn.ReLU, nn.Tanh)
    conv_dropout_options = (0, )
    fc_dropout_options = (0, 0.35, 0.5)
    fc_loop = range(1,3)
    start_numfilters = (128, 256)
    fc_embeds = (3,4,5)
    options = [J_loop, pool_options, I_loop, kernel_sizes, batchnorm_options, nonlin_funcs, conv_dropout_options, 
               fc_dropout_options, fc_loop, start_numfilters, fc_embeds]
    configs = list(itertools.product(*options))
    
    op_pipeline = []
    op_size = []
    for config in configs:
        pool_kernelsize = 2
        pool_strideh = 1
        pool_stridew = 2
        pool_pad = 0
        conv_stride = 1
        j, pool_option, i, conv_kernelsize, batch_norm, nonlin_func, conv_dropout, fc_dropout, k, num_filters, fc_embed = config
        accum = []
        size = []
        prev_inchannel = C
        h = H
        w= W
        conv_pad = (conv_kernelsize-1)//2
        for cj in range(j):
            if(j-cj <= 2):
                pool_strideh = 2
            if(num_filters > 32): # keep decreasing number of filters until it reaches 32
                num_filters = int(num_filters/2)
            for ci in range(i):
                accum.append(('Conv', prev_inchannel, num_filters, conv_kernelsize, conv_stride, conv_pad, not batch_norm))
                h = int(np.floor((h+2*conv_pad-conv_kernelsize)/conv_stride + 1))
                w = int(np.floor((w+2*conv_pad-conv_kernelsize)/conv_stride + 1))
                prev_inchannel = num_filters
                if(batch_norm):
                    accum.append(('BatchNorm2d', num_filters))
                accum.append(('NonLinearFunc', nonlin_func))
                if(conv_dropout):
                    accum.append(('Dropout2d', conv_dropout))
            for ci in range(i):
                size.append((h, w))
            accum.append((pool_option, pool_kernelsize, (pool_strideh, pool_stridew), pool_pad))
            h = int(np.floor((h+2*pool_pad-pool_kernelsize)/pool_strideh + 1))
            w = int(np.floor((w+2*pool_pad-pool_kernelsize)/pool_stridew + 1))
            size.append((h, w))
        if(h<conv_kernelsize or w<conv_kernelsize): # make sure that we have features to map (i.e. valid dimensions)
            continue
        # fc operation
        # ->[FC->BatchNorm?->NonLinFunc->Dropout?]*K
        accum_fc = []
        size_fc = []
        in_features = int(num_filters*h*w)
        for ck in range(k):
            if(ck==0):
                denom= fc_embed # initially embed the unrolled feature vector using fc_embed
            else:
                denom = 2 # divide the subsequent vector dimension by half
            out_features=in_features//denom
            accum_fc.append(('Linear', in_features, out_features, not batch_norm))
            size_fc.append((in_features, out_features))
            in_features = out_features
            if(batch_norm):
                accum_fc.append(('BatchNorm1d', out_features))
            accum_fc.append(('NonLinearFunc', nonlin_func))
            if(fc_dropout):
                accum_fc.append(('Dropout', fc_dropout))
        accum_hiddentoTag = [('Linear', out_features, 2, True)]
        size_hiddentoTag = [(out_features, 2)]
        op_pipeline.append((accum, accum_fc, accum_hiddentoTag))
        op_size.append((size, size_fc, size_hiddentoTag))
    return(op_pipeline, op_size)


def generate_operation_pipeline_cnnwide(input_shape):
    # convolution operation regex-like declaration
    # Input ->[Conv->BatchNorm?->NonLinFunc->Dropout?]->Pool
    C, H, W = input_shape # C: channel index, H: y-dimension of the matrix, W: x-dimension of the matrix
    numkernels_types = range(2,4)
    kernel_sizes = (2,3,5)
    numfilter_options= (16,32,64) # number of filters to be used for every kernel type
    batchnorm_options = (1,0)
    nonlin_funcs = (nn.Tanh,nn.ReLU)
    conv_dropout_options = (0, )
    convpad_flags = (0,1)
    pool_options = ('MaxPool', 'AvgPool')
    fc_dropout_options = (0, 0.35, 0.5)
    fc_loop = range(1,3)
    fc_embeds = range(1,5)
    options = [numkernels_types, numfilter_options, batchnorm_options, nonlin_funcs, conv_dropout_options, convpad_flags,
               pool_options, fc_dropout_options, fc_loop, fc_embeds]
    configs = list(itertools.product(*options))
    
    op_pipeline = []
    op_size = []
    for config in configs:
        num_kerneltypes, numfilters, batch_norm, nonlin_func, conv_dropout, convpad_flag, pool_option, fc_dropout, k, fc_embed = config
        accum_conv = []
        size_conv = []
        for i in range(num_kerneltypes):
            accum = []
            size = []
            prev_inchannel = C
            h = H
            w= W
            conv_ksizeh = kernel_sizes[i]
            conv_ksizew = w
            if(convpad_flag):
                conv_padh = (conv_ksizeh-1)//2
            else:
                conv_padh = 0
            conv_padw = 0
            conv_stride = 1
            accum.append(('Conv', prev_inchannel, numfilters, (conv_ksizeh, conv_ksizew), conv_stride, (conv_padh, conv_padw), not batch_norm))
            h = int(np.floor((h+2*conv_padh-conv_ksizeh)/conv_stride + 1))
            w = int(np.floor((w+2*conv_padw-conv_ksizew)/conv_stride + 1)) # in this context w will be always equal to 1
            if(batch_norm):
                accum.append(('BatchNorm2d', numfilters))
            accum.append(('NonLinearFunc', nonlin_func))
            if(conv_dropout):
                accum.append(('Dropout2d', conv_dropout))

            size.append((h, w))
            pool_ksizeh = h
            pool_ksizew = w
            pool_stride=1
            pool_pad= 0
            accum.append((pool_option, (pool_ksizeh, pool_ksizew), pool_stride, pool_pad))
            accum_conv.append(accum)
            size_conv.append(size)

        # fc operation
        # ->[FC->BatchNorm?->NonLinFunc->Dropout?]*K
        accum_fc = []
        size_fc = []
        in_features = int(numfilters*num_kerneltypes)
        for ck in range(k):
            if(ck==0):
                denom= fc_embed # initially embed the unrolled feature vector using fc_embed
            else:
                denom = 2 # divide the subsequent vector dimension by half
            out_features=in_features//denom
            accum_fc.append(('Linear', in_features, out_features, not batch_norm))
            size_fc.append((in_features, out_features))
            in_features = out_features
            if(batch_norm):
                accum_fc.append(('BatchNorm1d', out_features))
            accum_fc.append(('NonLinearFunc', nonlin_func))
            if(fc_dropout):
                accum_fc.append(('Dropout', fc_dropout))
        # final mapp to classes
        accum_hiddentoTag = [('Linear', out_features, 2, True)]
        size_hiddentoTag = [(out_features, 2)]
        op_pipeline.append((accum_conv, accum_fc, accum_hiddentoTag))
        op_size.append((size_conv, size_fc, size_hiddentoTag))
    return(op_pipeline, op_size)

def get_model_numparams(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def generate_opeartion_pipeline_nn(input_dim):
    num_classes = 2
    I_loop = range(1,6)
    batchnorm_options = (0,1)
    nonlin_funcs = (nn.ReLU,nn.Tanh)
    fc_dropout_options = (0, 0.15, 0.3, 0.45)
    embed_dim = (2,3,4)
    options = [I_loop, embed_dim, batchnorm_options, nonlin_funcs, fc_dropout_options]
    configs = list(itertools.product(*options))
    
    op_pipeline = []
    op_size = []
    for config in configs:
        i, embed_dim, batch_norm, nonlin_func, fc_dropout = config
        # fc operation
        # ->[FC->BatchNorm?->NonLinFunc->Dropout?]*K
        accum_fc = []
        size_fc = []
        in_features = input_dim
        for ci in range(i):
            out_features=in_features//embed_dim
            accum_fc.append(('Linear', in_features, out_features, not batch_norm))
            size_fc.append((in_features, out_features))
            in_features = out_features
            if(batch_norm):
                accum_fc.append(('BatchNorm1d', out_features))
            accum_fc.append(('NonLinearFunc', nonlin_func))
            if(fc_dropout):
                accum_fc.append(('Dropout', fc_dropout))
            in_features = out_features
        if(in_features < num_classes or out_features < num_classes):
            continue
        accum_hiddentoTag = [('Linear', out_features, num_classes, True)]
        size_hiddentoTag = [(out_features, num_classes)]
        op_pipeline.append((accum_fc, accum_hiddentoTag))
        op_size.append((size_fc, size_hiddentoTag))
    return(op_pipeline, op_size)

ParamCNNConfig= namedtuple("ParamCNNConfig", ['op_pipeline', 'weight_decay', 'optimizer', 'batch_size'])


def get_config_nn(input_shape, prob_interval_truemax, prob_estim):
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    op_pipeline, __= generate_opeartion_pipeline_nn(input_shape)
    config_options = []
    l2_reg = (1e-3,1e-2,1e-1)
    optimizer_opts = (optim.Adadelta, optim.Adam)
    batch_size = (64,128)
    options = [op_pipeline, l2_reg, optimizer_opts, batch_size]
    for config in itertools.product(*options):
        config_options.append(ParamCNNConfig(*config))
    if(num_trials > len(config_options)):
        num_trials = len(config_options) # case where the options are fewer than number of estimated trials -- the case of crf only
    indxs = np.random.choice(len(config_options), size=num_trials, replace=False)
    target_options = []
    for indx in indxs:
        target_options.append(config_options[indx])
    return(target_options)

def get_config_cnn(input_shape, cnn_type, prob_interval_truemax, prob_estim):
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    if(cnn_type == 'CNN_Labeler'):
        op_pipeline, __= generate_operation_pipeline_cnn(input_shape)
    else:
        op_pipeline, __= generate_operation_pipeline_cnnwide(input_shape)

    config_options = []
    l2_reg = (1e-3,1e-2,1e-1)
    optimizer_opts = (optim.Adadelta, optim.Adam)
    batch_size = (8,16)
    options = [op_pipeline, l2_reg, optimizer_opts, batch_size]
    for config in itertools.product(*options):
        config_options.append(ParamCNNConfig(*config))
    if(num_trials > len(config_options)):
        num_trials = len(config_options) # case where the options are fewer than number of estimated trials -- the case of crf only
    indxs = np.random.choice(len(config_options), size=num_trials, replace=False)
    target_options = []
    for indx in indxs:
        target_options.append(config_options[indx])
    return(target_options)

ParamConfig= namedtuple("ParamConfig", ['input_dim', 'embed_dim', 'hidden_dim',
                                        'num_hiddenlayers', 'pdropout', 'batch_norm', 
                                        'nonlinear_func','weight_decay', 'optimizer', 'batch_size', 
                                        'alpha_param', 'sampling_func', 'rnn_class','interm_dim'])

def get_config(input_dim, prob_interval_truemax, prob_estim, 
               alpha_param = False, sampling_func = False, 
               rnn_class = True, crf_mode=None):
    """ generate all combination of hyperparameters and select uniformly randomly based on determined number of trials
    """
    
    input_dims = (input_dim,)
    embed_dims = (0, input_dim//3, input_dim//2, input_dim)
    nonlinear_func = (F.relu, F.tanh)
    batch_norm = (0,)
    if(crf_mode in {'crf', 'crf_nn'}):
        hidden_dims = (None, )
        num_hiddenlayers = (None, )
        pdropout = (None, )
        target_indx = 1 # case of CRF with nn model, intermediate dimension is function of embed dimension
        if(crf_mode == 'crf'):        
            embed_dims = (0, )
            nonlinear_func = (None,)
            target_indx = None # case of CRF only model, there is no intermediate embedding

    else:
        hidden_dims = (8, 16, 32,64,128)
        num_hiddenlayers = (1,2)
        pdropout = (0.15, 0.35, 0.5)
        target_indx = 2 # case of RNN is included in the model, intermediate dimension is function of hidden dimension

    l2_reg = (1e-1, 1e-2, 1e-3)
    batch_size = (1, 16, 32, 64)
    optimizer_opts = (optim.Adadelta, optim.Adam)
    if(alpha_param):
        alpha_param = (0.65, 0.8,0.95)
    else:
        alpha_param = (None, )
    if(sampling_func):
        sampling_func = ('linear', 'exponential', 'sigmoid')
    else:
        sampling_func = (None, )
    if(rnn_class):
        rnn_class = (nn.RNN, nn.GRU, nn.LSTM)
    else:
        rnn_class = (None, )
        
    options = [input_dims, embed_dims, hidden_dims,
               num_hiddenlayers, pdropout, batch_norm, nonlinear_func, 
               l2_reg, optimizer_opts, batch_size, 
               alpha_param, sampling_func, rnn_class]
    
    config_options = []
            
    for config in itertools.product(*options):
        config_lst = list(config)
        if(crf_mode == 'crf'):
            tmp = config_lst + [0]
            config_options.append(ParamConfig(*tmp))
        else:
            for div in range(0, 3):
                target_dim = config_lst[target_indx]
                if(div):
                    tmp = config_lst + [target_dim//div]
                else:
                    tmp = config_lst + [div]
                config_options.append(ParamConfig(*tmp))
    print("total number of options: ", len(config_options))
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    if(num_trials > len(config_options)):
        num_trials = len(config_options) # case where the options are fewer than number of estimated trials -- the case of crf only
    indxs = np.random.choice(len(config_options), size=num_trials, replace=False)
    target_options = []
    for indx in indxs:
        target_options.append(config_options[indx])
    return(target_options)

def scheduled_sampling(mode, numbatches_per_epoch, num_epochs, **kwargs):
    max_ep = kwargs.get('max_epsilon', 0.99)
    min_ep = kwargs.get('min_epsilon', 0.2)
    damp_factor = kwargs.get('damping_factor', 2)
    if(mode == 'linear'):
        # compute slope
        slope = (min_ep *(1/damp_factor) - max_ep)/(num_epochs*numbatches_per_epoch-1)
        intercept = max_ep
        def linear_schedule(ibatch):
            return(slope*ibatch+intercept)
        return(linear_schedule)
    elif(mode == 'exponential'):
        # y=ae^{bx}
        b = (np.log(min_ep/max_ep))/((num_epochs*numbatches_per_epoch-1))
        def exponential_schedule(ibatch):
            return(max_ep*np.exp(damp_factor*b*ibatch))
        return(exponential_schedule)
    elif(mode == 'sigmoid'):
        # y = eta/(eta+aexp(bx))
        eta = kwargs.get('eta', 1)
        a = (1-max_ep)*eta/max_ep
        b = np.log(((1-min_ep)*eta)/(a*min_ep))/(num_epochs*numbatches_per_epoch-1)
        def sigmoid_schedule(ibatch):
            return(eta/(eta+a*np.exp(damp_factor*b*ibatch)))
        return(sigmoid_schedule)

def regenerate_dataset(dset_dict, to_gpu, memmap):
    target_dsets = {}
    if(memmap):
        pclass = PatientSeqDatasetMemmap
    else:
        pclass = PatientSeqDataset
        
    for dsettype in dset_dict:
        pdt = dset_dict[dsettype]
#         pdt.update_seqtensor(memmap=memmap) # due to issues with size of seq_tensor and 
                                            # multiprocessing, this is handled within each spawned process
        pdt.read_fromdisk(pdt.fpath, memmap=memmap)
        target_dsets[dsettype] = pclass(pdt, to_gpu=to_gpu)
    return(target_dsets)


def load_dataset(dataset_path, dsettypes = [], memmap = True):
    dataset_tuple = ReaderWriter.read_data(dataset_path)  
    base_dir = os.path.dirname(dataset_path)
    target_dsets = {}
    if(not dsettypes):
        dsettypes = list(dataset_tuple.keys())
    input_shape = 0
    fname='pdtm_object.pkl'
    for dsettype in dsettypes:
        print(dsettype)
        foldername = "{}_pdtm".format(dsettype)
        target_dir = os.path.join(base_dir, foldername)
        target_fpath = os.path.join(target_dir, fname)
        pdtm_obj = ReaderWriter.read_data(target_fpath)
        pdtm_obj.fpath = target_dir # keep the directory where to find the memory mapped data on disk
#         pdtm_obj.read_fromdisk(target_dir, memmap=memmap)
        if(not input_shape):
            input_shape = pdtm_obj.seqtensor_info[-1] # hack to get input shape --- 
        target_dsets[pdtm_obj.dsettype] = pdtm_obj
    return(target_dsets, input_shape)


"""
 :func:`metric_report` should merge with :func:`metric_report_levent` function --- the latter is special case of the former
 TODO:
    refactor both functions to be one and update the functions in :module:`train_eval` module that call them
"""

def metric_report(pred_target, ref_target, prob_score, indexevent_indic, epoch, outlog, paddsymbol, map_dict={}, plot_roc=True):
    outcome_lst = []
#     print("map_dict: \n", map_dict)
#     print(ref_target.shape)
#     print(pred_target.shape)
#     print(indexevent_indic.shape)
#     print(prob_score.shape)
#     print("ref_target \n", ref_target)
#     print("pred_target \n", pred_target)
#     print("indexevent_indic \n", indexevent_indic)
#     print("probscore \n", prob_score)
    for i, arr in enumerate((ref_target, pred_target, prob_score, indexevent_indic)):
        tmp = arr.cpu().numpy()
        if(i == 0):
            cond = tmp!=paddsymbol # indices with values not equal to paddsymbol
        outcome_lst.append(tmp[cond])
#     print("outcome_lst[-1] \n", outcome_lst[-2])
    indx_cond = outcome_lst[-1] == 1 # indices of index events 
#     print(indx_cond)
    if(map_dict): 
        outcome_lst[0] = np.vectorize(map_dict.get)(outcome_lst[0])
#     print("updated ref_target \n", outcome_lst[0])
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Classification report on all events:" + lsep
    report += str(classification_report(outcome_lst[0], outcome_lst[1])) + lsep
    report += "weighted f1:" + lsep
    report += str(f1_score(outcome_lst[0], outcome_lst[1], average='weighted')) + lsep
    report += "micro f1:" + lsep
    report += str(f1_score(outcome_lst[0], outcome_lst[1], average='micro')) + lsep
    report += "-"*30 + lsep
    report += "Classification report on index events:" + lsep
    report += str(classification_report(outcome_lst[0][indx_cond], outcome_lst[1][indx_cond])) + lsep
    report += "weighted f1:" + lsep
    weighted_f1 = f1_score(outcome_lst[0][indx_cond], outcome_lst[1][indx_cond], average='weighted')
    report += str(weighted_f1) + lsep
    report += "micro f1:" + lsep
    micro_f1 = f1_score(outcome_lst[0][indx_cond], outcome_lst[1][indx_cond], average='micro')
    report += str(micro_f1) + lsep
    except_raised = False
    try:
        auc_score = roc_auc_score(outcome_lst[0][indx_cond], outcome_lst[2][indx_cond])
        avg_precrecall = average_precision_score(outcome_lst[0][indx_cond], outcome_lst[2][indx_cond])
    except Exception:
        print("Error AUC computation !!")
        print("ref_target: ", outcome_lst[0][indx_cond])
        print("prob_score: ", outcome_lst[2][indx_cond])
        error_desc = ''
        if(len(np.unique(outcome_lst[0][indx_cond]))==1):
            labels_str = ":".join([str(elm) for elm in np.unique(outcome_lst[0][indx_cond]).tolist()])
            error_desc += "Can't compute AUC for ref_target array having one label type: " + labels_str + lsep
            print(error_desc)
        error_count = np.isnan(outcome_lst[2][indx_cond]).sum()
        if(error_count):
            error_desc += "Can't compute AUC for NaN probability score -- total number of NaN values: {}".format(error_count) + lsep
            print(error_desc)
        # default to 0
        auc_score = 0
        avg_precrecall = 0
        except_raised = True
        report += "Error AUC computation !!" + lsep
        report += error_desc
#         report += "Ref target unique values:" + lsep
#         report += ":".join([str(elm) for elm in np.unique(outcome_lst[0][indx_cond]).tolist()]) + lsep
#         report += "Prob score unique values:" + lsep
#         report += ":".join([str(elm) for elm in np.unique(outcome_lst[2][indx_cond]).tolist()]) + lsep
    finally:
        report += "auc:" + lsep
        report += str(auc_score) + lsep
        report += "average precision recall:" + lsep
        report += str(avg_precrecall) + lsep
        ReaderWriter.write_log(report, outlog)
        # plot roc curve
        if(plot_roc and not except_raised):
            outdir = os.path.dirname(outlog)
            plot_roc_curve(outcome_lst[0][indx_cond], outcome_lst[2][indx_cond], 'allindexevents',outdir)
            plot_precision_recall_curve(outcome_lst[0][indx_cond], outcome_lst[2][indx_cond], 'allindexevents',outdir)
    return(weighted_f1, auc_score)


def metric_report_levent(pred_target_levent, ref_target_levent, prob_score_levent, outlog, paddsymbol, map_dict={}, plot_roc=True):
    outcome_lst = []
#     print(ref_target_levent.shape)
#     print(pred_target_levent.shape)
#     print(prob_score_levent.shape)
#     print("ref_target_levent \n", ref_target_levent)
#     print("pred_target_levent \n", pred_target_levent)
#     print("probscore_levent \n", prob_score_levent)
    for i, arr in enumerate((ref_target_levent, pred_target_levent, prob_score_levent)):
        tmp = arr.cpu().numpy()
        if(i == 0):
            cond = tmp!=paddsymbol # indices with values not equal to paddsymbol
        outcome_lst.append(tmp[cond])
    if(map_dict): 
        outcome_lst[0] = np.vectorize(map_dict.get)(outcome_lst[0])
#     print("updated ref_target \n", outcome_lst[0])
    lsep = "\n"
    report = "Classification report on last event:" + lsep
    report += str(classification_report(outcome_lst[0], outcome_lst[1])) + lsep
    report += "weighted f1:" + lsep
    weighted_f1 = f1_score(outcome_lst[0], outcome_lst[1], average='weighted')
    report += str(weighted_f1) + lsep
    report += "micro f1:" + lsep
    report += str(f1_score(outcome_lst[0], outcome_lst[1], average='micro')) + lsep
    except_raised = False
    try:
        auc_score = roc_auc_score(outcome_lst[0], outcome_lst[2])
        avg_precrecall = average_precision_score(outcome_lst[0], outcome_lst[2])
        
    except Exception:
        print("Error AUC computation !!")
        print("ref_target: ", outcome_lst[0])
        print("prob_score: ", outcome_lst[2])
        error_desc = ''
        if(len(np.unique(outcome_lst[0]))==1):
            labels_str = ":".join([str(elm) for elm in np.unique(outcome_lst[0]).tolist()])
            error_desc += "Can't compute AUC for ref_target array having one label type: " + labels_str + lsep
            print(error_desc)
        error_count = np.isnan(outcome_lst[2]).sum()
        if(error_count):
            error_desc += "Can't compute AUC for NaN probability score -- total number of NaN values: {}".format(error_count) + lsep
            print(error_desc)
        # default to 0
        auc_score = 0
        avg_precrecall = 0
        except_raised = True
        report += "Error AUC computation !!" + lsep
        report += error_desc  
    finally:
        report += "auc:" + lsep
        report += str(auc_score) + lsep
        report += "average precision recall:" + lsep
        report += str(avg_precrecall) + lsep
        ReaderWriter.write_log(report, outlog)
        # plot roc curve
        if(plot_roc  and not except_raised):
            outdir = os.path.dirname(outlog)
            plot_roc_curve(outcome_lst[0], outcome_lst[2], 'lastindexevent',outdir)
            plot_precision_recall_curve(outcome_lst[0], outcome_lst[2], 'lastindexevent',outdir)

    return(weighted_f1, auc_score)

def plot_precision_recall_curve(ref_target, prob_poslabel, figname, outdir):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    thresholds[0]=1
    plt.figure(figsize=(9,6))
    plt.plot(pr, rec, 'bo', label='Precision vs Recall')
#     plt.plot(np.arange(0,len(thresholds)), thresholds, 'r-', label='thresholds')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. recall curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('precisionrecall_curve_{}'.format(figname) + ".pdf")))
    plt.close()
    
def plot_roc_curve(ref_target, prob_poslabel, figname, outdir):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    thresholds[0]=1
    plt.figure(figsize=(9,6))
    plt.plot(fpr, tpr, 'bo', label='TPR vs FPR')
    plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('roc_curve_{}'.format(figname) + ".pdf")))
    plt.close()

def plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir):
    dsettypes =  epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9,6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r', 
                 epoch_loss_avgsamples[dsettype], 'b')
        #'epoch raw loss', 'epoch batch average loss', 'epoch training samples average loss'
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss', 'epoch training samples average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()
        
def plot_scheduler(track_scheduler, wrk_dir):
    plt.figure(figsize=(9,6))
    plt.plot(track_scheduler, 'r')
    plt.xlabel("number of updates")
    plt.ylabel("probability of choosing ground-truth label")
    plt.savefig(os.path.join(wrk_dir, os.path.join("scheduler_plot.pdf")))
    plt.close()        
        
BestModel= namedtuple("BestModel", ['model_prefix', 'model_statedict_pth', 'model_config_pth'])

def retrieve_bestmodels(target_dir, target_folder = 'best_model'):
    bestmodels_lst = []
    if(target_folder == 'best_model'):
        target_dir = os.path.join(target_dir, 'best_model') # update target dir
        for __, __, files in os.walk(target_dir):  
            for filename in files: # might have multiple best models
                model_prefix = filename.split('_bestmodel.pkl')
                if(len(model_prefix)>1):
                    model_prefix = model_prefix[0] # get the prefix
                    model_statedict = os.path.join(target_dir, '{}_bestmodel.pkl'.format(model_prefix))
                    model_config = os.path.join(target_dir, '{}_config.pkl'.format(model_prefix))
                    bestmodels_lst.append(BestModel(model_prefix, model_statedict, model_config))
    else: # general case
        parent_folder = os.path.basename(target_dir)
        for path, folder_names, file_names in os.walk(target_dir):
#             print(path)
#             print(folder_names)
#             print(file_names)
            curr_foldername = os.path.basename(path)
            if(curr_foldername not in {parent_folder, 'best_model'} and curr_foldername[0]!='.'):
                model_prefix = curr_foldername
                model_statedict = os.path.join(path, 'bestmodel.pkl')
                model_config = os.path.join(path, 'config.pkl')
                bestmodels_lst.append(BestModel(model_prefix, model_statedict, model_config))
#             print("-"*25)
    return(bestmodels_lst)

if __name__ == "__main__":
    configs = get_config(500, 0.03, 0.99)
    print(configs)
    assert len(configs) == 152
    