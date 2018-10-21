'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
# -*- coding: utf-8 -*-

import datetime
import shutil
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
import pandas as pd
from model_rnn import RNN_Labeler
from model_rnnss import RNNSS_Labeler
from model_cnn import CNN_Labeler
from model_cnnwide import CNNWide_Labeler
from model_nn import NN_Labeler
from model_rnncrf_pair import RNNCRF_Pair_Labeler
from model_rnncrf_unary import RNNCRF_Unary_Labeler
from model_crf_pair import CRF_Pair_Labeler
from model_crf import CRF_Labeler

from utilities import *

def construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path):
    transmatrix_flag = options.get('transmatrix_flag', None)
    stopstate_symb = options.get('stopstate_symb', None)
    bidirection = options.get('bidirection', False)
    if(model_class not in {'NN_Labeler', 'CNN_Labeler', 'CNNWide_Labeler'}):
        args = [config.input_dim, config.hidden_dim, y_codebook]
        kwargs = {'embed_dim': config.embed_dim,
                'interm_dim': config.interm_dim,
                'num_hiddenlayers': config.num_hiddenlayers,
                'bidirection': bidirection,
                'pdropout': config.pdropout,
                'rnn_class': config.rnn_class,
                'nonlinear_func': config.nonlinear_func,
                'to_gpu':to_gpu
                }
    if(model_class in {'RNN_Labeler', 'RNNSS_Labeler'}):
        loss_mode = options.get('loss_mode')
        model_name = "{}_lossmode_{}".format(model_class, loss_mode)
        if(model_class == 'RNNSS_Labeler'):
            kwargs['startstate_symb'] = "__START__"
    elif (model_class.startswith('RNNCRF')):
        kwargs.update({'startstate_symb': "__START__", 'stopstate_symb': stopstate_symb})
        if('Pair_Labeler' in model_class):
            kwargs['transmatrix_flag'] = transmatrix_flag
        model_name = model_class
    elif(model_class.startswith('CRF')):
        crf_type = options.get('crf_type')
        args = [config.input_dim, y_codebook]
        kwargs = {'embed_dim':config.embed_dim,
                  'interm_dim':config.interm_dim,
                  'nonlinear_func':config.nonlinear_func,
                  'startstate_symb':"__START__",
                  'stopstate_symb':stopstate_symb,
                  'to_gpu':to_gpu
                  }
        prefix, suffix = model_class.split("CRF")
        if(crf_type == 'only_crf'):
            kwargs['embed_dim'] = 0
            kwargs['interm_dim'] = 0
            model_name = "CRF_Only" + suffix
        else:
            model_name = "CRF_NN" + suffix
    elif(model_class.startswith('CNN')):
        conv_pipeline, fc_pipeline, hiddtotag_pipeline = config.op_pipeline
        args = [y_codebook, conv_pipeline, fc_pipeline, hiddtotag_pipeline]
        kwargs = {}
        # cnn_type = options.get('cnn_type')
        model_name = model_class # {CNN_Labeler, CNNWide_Labeler}
    elif(model_class.startswith('NN')):
        fc_pipeline, hiddtotag_pipeline = config.op_pipeline
        args = [y_codebook, fc_pipeline, hiddtotag_pipeline]
        kwargs = {}
        model_name = model_class

    # call a class using its name (string)
    model = globals()[model_class](*args, **kwargs)
    if(modelstatedict_path):
        # load the state dictionary of the model (if specified)
        model.load_state_dict(torch.load(modelstatedict_path))
    if(IS_CUDA and to_gpu):
        model.cuda()
    return(model, model_name)

def construct_load_dataloaders(target_dsets, config, wrk_dir, options):
    """construct dataloaders for the dataset
       
       Args:
            target_dsets:
            config:
            wrk_dir:
            options:
    """
    featimp_inspect = options.get('featimp_inspect', False)
    # setup the dsettypes -- this guarantees to have train (if found) as first element
    available_dsettypes = set(target_dsets.keys())
    if('train' in available_dsettypes):
        dsettypes = ['train']
    else:
        dsettypes = []
    
    for elem in (available_dsettypes - set(dsettypes)):
        dsettypes.append(elem)
                
    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    epoch_loss_avgsamples = {}
    flog_out = {}
    score_dict = {}
    featimp_analysis = {}
    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
        else:
            shuffle = False
        data_loaders[dsettype] = DataLoader(target_dsets[dsettype],
                                            batch_size=config.batch_size,
                                            shuffle=shuffle, num_workers=NUM_WORKERS)
        if(wrk_dir):
            epoch_loss_avgbatch[dsettype] = []
            epoch_loss_avgsamples[dsettype] = []
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
            score_dict[dsettype] = (0, 0.0, 0.0, 0.0, 0.0, 0.0) # (best_epoch, avg_weighted_f1, 
                                                                #  weighted_f1_indexevents, weighted_f1_lastevent,
                                                                #  auc_indexevents, auc_lastevent)
            if(featimp_inspect):
                featimp_analysis = {dsettype:pd.DataFrame()}
        # print(data_loaders)
    if(wrk_dir):
        res = (data_loaders, dsettypes, epoch_loss_avgbatch, epoch_loss_avgsamples, flog_out, score_dict, featimp_analysis)
    else:
        res = (data_loaders, dsettypes)
    return(res)

def rnncrf_modelinspect(config, model_class, target_dsets, y_codebook, num_epochs, wrk_dir, rank, to_gpu, memmap, modelstatedict_path=None, options={}):
    """Inspecting feature attribution/importance using switch on/off approach as reported in Xx et al. <ref>__

       TODO:
            restructure the switch on/off analysis
    """
    pid = "{}-{}".format(os.getpid(), rank)
    print("process id", pid)
    
    fdtype, ldtype = get_tensordtypes(to_gpu)
    if(IS_CUDA and to_gpu): # to integrate it with get_tensordtypes in `utilities module`
        bdtype = torch.cuda.ByteTensor
    else:
        bdtype = torch.ByteTensor
    map_dict = options.get('map_dict', {})
    if(map_dict): # target label (i.e. readmission label)
        poslabel = options.get('poslabel')
    decoder_type = options.get('decoder_type')
    fold_id = options.get('fold_id')
    scoremat_option = options.get('scoremat_option')
    model, model_name = construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path)
    num_params = get_model_numparams(model)
    print("model name: ", model_name)
    print("num params: ", num_params)
    print('model:\n ', model)
    model.map_dict = map_dict
    # load dataset
    target_dsets = regenerate_dataset(target_dsets, to_gpu, memmap)
    data_loaders, dsettypes = construct_load_dataloaders(target_dsets, config, None, options)
    # write description of the current run
    ReaderWriter.write_log("pid: {}".format(pid) + "\n" + "number of params: {}".format(num_params) + "\n" + str(model) + "\n" + str(config) + "\n", os.path.join(wrk_dir, 'description.txt'))
    ReaderWriter.dump_data(config, os.path.join(wrk_dir, 'config.pkl'))
    # get on/off features
    switch_onoff_features = options.get('switch_onoff_features')
    switch_onoff_set = set(switch_onoff_features.keys())
    
    for epoch in range(num_epochs):
        print("epoch: {}, pid: {}, model_name: {}".format(epoch, pid, model_name))
        for dsettype in dsettypes:
            data_loader = data_loaders[dsettype]
            idx_mapper_inverse = target_dsets[dsettype].pdt_object.idx_mapper_inverse
            records = []
            for i_batch, sample_batch in enumerate(data_loader):
                #print("i_batch: ", i_batch)
                #print("sample_batch \n", sample_batch)
                seqs_batch, labels_batch, seqs_len, indexevent_batch, patientsindx_batch  = sample_batch
                seqs_len = seqs_len.view(-1) # make sure it is a column vector size (n,)
                batch_size = seqs_batch.size(0)
#                 print("batch_size: ", batch_size)
                for bindx in range(batch_size):
                    #print("bindx: ", bindx)
                    pindx = patientsindx_batch[bindx]
#                     print("pindx ", pindx)
                    patient_id = idx_mapper_inverse[pindx]
#                     print("patient_id ", patient_id)
                    seqlen = seqs_len[bindx]
                    seq  = seqs_batch[bindx, :seqlen,:]
                    ref_labels = labels_batch[bindx, :seqlen]
                    indexevent_indx = indexevent_batch[bindx,:seqlen].type(bdtype)
                    num_indexevents = indexevent_batch[bindx,:seqlen].sum()
#                     print("ref_labels: \n", ref_labels)
#                     print("indexevent_indx: \n", indexevent_indx)
#                     print("ref_labels[indexevent_indx]: \n",ref_labels[indexevent_indx])
#                     print("seq: \n", seq)
                    nonzero_indx = torch.nonzero(seq)
                    
                    for indx_i, indx_j in nonzero_indx:
                        if(indx_j in switch_onoff_set):
#                             print("indx_i, indx_j: ", indx_i, indx_j)
#                             print("original value (root feature): \n", seq[indx_i, indx_j])
                            ftensor_indices = torch.Tensor(switch_onoff_features[indx_j]).type(ldtype)
#                             print("tensor_indices: ", ftensor_indices)
#                             print("seq original value: ", seq[indx_i][ftensor_indices])
#                             print("index_select: ", torch.index_select(seq[indx_i], 0, tensor_indices))
#                             print("index_slice: ", seq[indx_i][tensor_indices])

                            on_val = seq[indx_i][ftensor_indices]
                            seq[indx_i][ftensor_indices] = 0 # switch off the value
#                             print("seq updated value: \n", seq[indx_i][ftensor_indices])
                            current_seq = autograd.Variable(torch.unsqueeze(seq, 0))
                            seq_potential = model.compute_potential(current_seq)
                            pred_tagsindx, pred_score, optimal_seqscore, score_mat = model.decode(seq_potential, method=decoder_type)
                            if(scoremat_option == 'normalized_alpha'):
                                score_mat = model.compute_normalized_alpha(seq_potential)
                            pred_tagsindx_tensor = torch.Tensor(pred_tagsindx).type(ldtype)
#                             print("pred_tagsindx: \n", pred_tagsindx)
#                             print("prob_score: \n", pred_score)
#                             print("optimal_seqscore: ", optimal_seqscore)
                            seq_decerror = (pred_tagsindx_tensor == ref_labels).sum()/seqlen
#                             print(pred_tagsindx_tensor == ref_labels)
#                             print("seq_decerror \n", seq_decerror)
                            seq_decerror_indxevent = (pred_tagsindx_tensor[indexevent_indx] == pred_tagsindx_tensor[indexevent_indx]).sum()/num_indexevents
#                             print(pred_tagsindx_tensor[indexevent_indx] == pred_tagsindx_tensor[indexevent_indx])
#                             print("seq_decerror index events \n", seq_decerror_indxevent)
    
                            if(len(map_dict)):
                                prob_score = score_mat[:, 1] + score_mat[:, poslabel]
                            else:
                                prob_score = score_mat[:, 1]
    
                            record = (patient_id, pindx, seqlen, indx_i, indx_j, 0, ref_labels[indx_i],
                                      pred_tagsindx[indx_i], prob_score[indx_i], indexevent_indx[indx_i],
                                      seq_decerror_indxevent, seq_decerror, optimal_seqscore, model_name, fold_id)
                    
#                             print("record: \n", record)
                            records.append(record)
                            seq[indx_i][ftensor_indices] = on_val # put back the value
#                             print("seq original value: \n", seq[indx_i][ftensor_indices])
#                             print("-"*25)
#                     print("*"*25)
                    # to add records for the original sequence
                    # NOTE TO SELF: this should be optimized !!!
                    current_seq = autograd.Variable(torch.unsqueeze(seq, 0))
                    seq_potential = model.compute_potential(current_seq)
                    pred_tagsindx, pred_score, optimal_seqscore, score_mat = model.decode(seq_potential, method=decoder_type)
                    if(scoremat_option == 'normalized_alpha'):
                        score_mat = model.compute_normalized_alpha(seq_potential)
                    pred_tagsindx_tensor = torch.Tensor(pred_tagsindx).type(ldtype)
#                     print("pred_tagsindx: \n", pred_tagsindx)
#                     print("prob_score: \n", pred_score)
#                     print("optimal_seqscore: ", optimal_seqscore)
                    seq_decerror = (pred_tagsindx_tensor == ref_labels).sum()/seqlen
#                     print(pred_tagsindx_tensor == ref_labels)
#                     print("seq_decerror \n", seq_decerror)
                    seq_decerror_indxevent = (pred_tagsindx_tensor[indexevent_indx] == pred_tagsindx_tensor[indexevent_indx]).sum()/num_indexevents
#                     print(pred_tagsindx_tensor[indexevent_indx] == pred_tagsindx_tensor[indexevent_indx])
#                     print("seq_decerror index events \n", seq_decerror_indxevent)
                    for indx_i, indx_j in nonzero_indx:
                        if(indx_j in switch_onoff_set):                            
                            record = (patient_id, pindx, seqlen, indx_i, indx_j, seq[indx_i, indx_j], ref_labels[indx_i],
                                      pred_tagsindx[indx_i], prob_score[indx_i], indexevent_indx[indx_i],
                                      seq_decerror_indxevent, seq_decerror, optimal_seqscore, model_name, fold_id)
                            records.append(record)
                                     
            records = pd.DataFrame(records) # construct dataframe 
            # xi_value represents the value of the feature
            records.columns = ['pid', 'pindx', 'seq_len', 'time',
                               'xi_indx', 'xi_value', 'ref_target',
                               'pred_target', 'prob_target1', 'index_event',
                               'seq_decerror_indxevent', 'seq_decerror', 'optimal_seqscore',
                               'model_name', 'fold_id']
            
            ReaderWriter.dump_data(records, os.path.join(wrk_dir, 'featcontrib_switchonoff_df_{}_{}.pkl'.format(fold_id, dsettype)))

def rnncrf_run(config, model_class, target_dsets, y_codebook, num_epochs, wrk_dir,
               rank, to_gpu, memmap, modelstatedict_path=None, options={}):
    """train/eval/test RNNCRF/CRF/Neural CRF models
    """
    pid = "{}-{}".format(os.getpid(), rank) # process id description
    print("process id ", pid)
    fdtype, ldtype = get_tensordtypes(to_gpu)
    featimp_inspect = options.get('featimp_inspect', False)
    dec_outdir = options.get('dec_outdir')
    fold_id = options.get('fold_id')
    map_dict = options.get('map_dict', {})
    if(map_dict): # target label (i.e. readmission label)
        poslabel = options.get('poslabel')
    scoremat_option = options.get('scoremat_option')
    decoder_type = options.get('decoder_type')
    # define and init model
    model, model_name = construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path)
    num_params = get_model_numparams(model)
    print("model name: ", model_name)
    print("num params: ", num_params)
    print('model:\n ', model)
    model.map_dict = map_dict
    # setup optimizer
    reg_type = options.get('reg_type')
    if(reg_type == 'l2'):
        optimizer = config.optimizer(model.parameters(), weight_decay=config.weight_decay)

    rgrad_mode = options.get('restrict_grad_mode')
    rgrad_limit = options.get('restrict_grad_limit')
    target_dsets = regenerate_dataset(target_dsets, to_gpu, memmap)
    # load dataloaders with the relevant entities
    data_loaders, dsettypes, epoch_loss_avgbatch, epoch_loss_avgsamples, flog_out, score_dict, featimp_analysis = construct_load_dataloaders(target_dsets, config, wrk_dir, options)
    # write description of the current training run   
    ReaderWriter.write_log("pid: {}".format(pid) + "\n" + "number of params: {}".format(num_params) + "\n" + str(model) + "\n" + str(config) + "\n", os.path.join(wrk_dir, 'description.txt'))
    ReaderWriter.dump_data(config, os.path.join(wrk_dir, 'config.pkl'))
    targettag_indx = 1 # target label index

    for epoch in range(num_epochs):
        print("epoch: {}, pid: {}, model_name: {}".format(epoch, pid, model_name))
        for dsettype in dsettypes:
            pred_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            indexevent_indic = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            pred_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            
            prob_score = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
            prob_score_levent = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
            
            pindx = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            seqslen = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            
            data_loader = data_loaders[dsettype]
            idx_mapper_inverse = target_dsets[dsettype].pdt_object.idx_mapper_inverse
            epoch_loss = 0.

            if(dsettype == 'train'):
                model.train()
            else:
                model.eval()

            for i_batch, sample_batch in enumerate(data_loader):
                #print("i_batch: ", i_batch)
                #print("sample_batch \n", sample_batch)
                seqs_batch, labels_batch, seqs_len, indexevent_batch, patientsindx_batch = sample_batch
                seqs_len = seqs_len.view(-1) # make sure it is a column vector size (n,)
                patientsindx_batch = patientsindx_batch.type(ldtype)

                batch_size = seqs_batch.size(0)
#                 print("batch_size: ", batch_size)
                # reorder sequences in descending order
#                 seqs_len, perm_idx = seqs_len.sort(0, descending=True)
#                 seqs_len = seqs_len.type(ldtype)
#                 perm_idx = perm_idx.type(ldtype)
# #                 print("perm_idx ", perm_idx)
# #                 print("seqs_len ", seqs_len)
#                 seqs_batch = seqs_batch[perm_idx]
#                 labels_batch = labels_batch[perm_idx]
#                 indexevent_batch = indexevent_batch[perm_idx]
                # wrap with autograd.Variable
                data_var_batch = autograd.Variable(seqs_batch, requires_grad=featimp_inspect)
#                 print("data_var_batch \n {}".format(data_var_batch))
#                 print("seqs len \n", seqs_len)
#                 print("labels_batch \n", labels_batch)
#                 print("indexevent_batch \n", indexevent_batch)
                model.zero_grad()
                loss = autograd.Variable(torch.zeros(1).type(fdtype), requires_grad=True)
                for bindx in range(batch_size):
#                     print("bindx: ", bindx)
                    current_seq = torch.unsqueeze(data_var_batch[bindx, :seqs_len[bindx], :], 0)
#                     seq_potential = model.compute_potential(current_seq, procseq_option=procseq_option)
                    seq_potential = model.compute_potential(current_seq)

#                     print("current seq \n", current_seq)
                    ref_labels = labels_batch[bindx, :seqs_len[bindx]]
                    loss = loss + model.compute_loss_seqpot(seq_potential, ref_labels, mode='forward_vectors')
                    ref_target = torch.cat((ref_target, ref_labels))
                    ref_target_levent = torch.cat((ref_target_levent, ref_labels[-1:])) 
                    indexevent_indic = torch.cat((indexevent_indic, indexevent_batch[bindx, :seqs_len[bindx]]))
                    
                    if(decoder_type in {'posterior_prob', 'viterbi'}):
                        labels = []
                    elif(decoder_type == 'guided_viterbi'):
                        labels = ref_labels
                    # by default the score_mat returned is normalized 
                    pred_tagsindx, __, __, score_mat = model.decode(seq_potential, labels=labels, method=decoder_type)
                    # if(scoremat_option == 'normalized_alpha'):
                    #     score_mat = model.compute_normalized_alpha(seq_potential)
#                     print("pred_tagsindx: \n", pred_tagsindx)
                    pred_target = torch.cat((pred_target, torch.Tensor(pred_tagsindx).type(ldtype)))
                    pred_target_levent = torch.cat((pred_target_levent, pred_target[-1:]))
#                     print('ref_labels: \n', ref_labels)
#                     print("pred_target \n", pred_target)
#                     print("pred_target_levent \n", pred_target_levent)
#                     print("score_mat: \n", score_mat)
#                     print("score_mat poslabel: \n", score_mat.data[1:-1, pos_label])
#                     print("score_mat last event poslabel: \n", score_mat.data[-2:-1, pos_label])
#                     print("score_mat \n", score_mat)
#                     print("score_mat[:,1] \n", score_mat[:,1])
#                     print("prob_score \n", prob_score)
                    if(not len(map_dict)):
                        prob_score = torch.cat((prob_score, score_mat[:, targettag_indx]))
                        prob_score_levent = torch.cat((prob_score_levent, score_mat[-1:, targettag_indx]))
                    else:
                        prob_score = torch.cat((prob_score, score_mat[:, targettag_indx] + score_mat[:, poslabel]))
                        prob_score_levent = torch.cat((prob_score_levent, score_mat[-1:, targettag_indx] + score_mat[-1:, poslabel]))                  

#                     print("Y_decoded: \n", pred_tagsindx)
#                     print("Y_labells: \n", ref_labels)
                    # compare alpha/beta
#                     model.compare_alpha_beta(seq_potential)
                    if(featimp_inspect): # using gradient for the analysis of feature attribution
                        loss.backward()
                        print("analyzing feature contribution")
                        df = pd.DataFrame()
                        pindx = patientsindx_batch[bindx]
                        pid = idx_mapper_inverse[pindx]
                        seq_input = current_seq.data
                        seq_input_grad = current_seq.grad.data
                        ref_t = ref_labels
                        pred_t = pred_tagsindx
                        seqlen = seqs_len[bindx]
                        seq_indexevent = indexevent_batch[bindx,:seqs_len[bindx]]
                        # print("seq_input \n", seq_input)
                        # print("seq_input_grad \n", seq_input_grad)
                        # print("ref_t \n", ref_t)
                        # print("pred_t \n", pred_t)
                        # print("seq_indexevent \n", seq_indexevent)
                        tdf = pd.DataFrame()
                        input_dim = seq_input.size(-1)
                        tdf['pid'] = [pid]*input_dim*seqlen
                        tdf['xi_grad'] = seq_input_grad.view(-1).numpy()
                        tdf['xi_value'] = seq_input.view(-1).numpy()
                        tdf['index_event'] = seq_indexevent.numpy().repeat(input_dim)
                        tdf['ref_target'] = ref_t.repeat(input_dim)
                        tdf['pred_target'] = pred_t.repeat(input_dim)
                        tdf['time'] = np.arange(seqlen).repeat(input_dim)
                        tdf['xi_indx'] = np.tile(np.arange(input_dim),seqlen)
                        tdf['seq_len'] = [seqlen]*input_dim*seqlen
                        df = pd.concat([df, tdf], ignore_index=True, axis=0)   
                        featimp_analysis[dsettype] = pd.concat([featimp_analysis[dsettype], df], axis=0, ignore_index=True)
                                    
                pindx = torch.cat((pindx, patientsindx_batch))    
                seqslen = torch.cat((seqslen, seqs_len))
                    
                if(dsettype == 'train' and not featimp_inspect):
                    loss.backward()
                    # apply grad clipping
                    if(rgrad_mode):
                        restrict_grad_(model.parameters(), rgrad_mode, rgrad_limit)
                    optimizer.step()
                    
                epoch_loss += loss.data[0]
                
            if(dsettype == 'test' and dec_outdir):        
                write_seqprediction_df(ref_target, pred_target, prob_score,
                                       indexevent_indic, seqslen, pindx,
                                       idx_mapper_inverse,
                                       model_name, fold_id, dec_outdir)
                    
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss/len(data_loader.dataset))
            # classification report
            weighted_f1_indexevent, auc_indexevent = metric_report(pred_target, ref_target, prob_score, indexevent_indic, 
                                                                   epoch+1, flog_out[dsettype], PADDSYMB_INDX, map_dict=map_dict)
            weighted_f1_levent, auc_levent = metric_report_levent(pred_target_levent, ref_target_levent, prob_score_levent, 
                                                                  flog_out[dsettype], PADDSYMB_INDX, map_dict=map_dict)
            avg_f1 = (weighted_f1_indexevent + weighted_f1_levent)/2   
            perf = auc_levent         
            if(perf > score_dict[dsettype][-1]):
                score_dict[dsettype] = (epoch, avg_f1, weighted_f1_indexevent,weighted_f1_levent, auc_indexevent, auc_levent)
                if(dsettype == 'validation'):
                    torch.save(model.state_dict(), os.path.join(wrk_dir, 'bestmodel.pkl'))

    if(num_epochs>1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir)
    dump_scores(score_dict, wrk_dir)
    if(featimp_inspect):
        ReaderWriter.dump_data(featimp_analysis, os.path.join(wrk_dir, 'featanalysis_dict.pkl'))
    ReaderWriter.write_log("finished {}\n".format(os.path.basename(wrk_dir)), os.path.join(os.path.dirname(wrk_dir), 'out.log'), mode='a')

def rnn_run(config, model_class, target_dsets, y_codebook, num_epochs, wrk_dir,
            rank, to_gpu, memmap, modelstatedict_path=None, options={}):
    """train/eval/test RNN model
    """
    pid = "{}-{}".format(os.getpid(), rank) # process id description
    print("process id ", pid)
    fdtype, ldtype = get_tensordtypes(to_gpu)
    featimp_inspect = options.get('featimp_inspect', False)
    dec_outdir = options.get('dec_outdir')
    # print("decoded_outputdir ", dec_outdir)
    fold_id = options.get('fold_id')
    class_weights = options.get('class_weights')
    if(class_weights):
        class_weights = torch.Tensor(list(class_weights)).type(fdtype) # update class weights to float tensor
    else:
        class_weights = torch.Tensor([1,1]).type(fdtype) # weighting all casess equally
    print("class_weights: ", class_weights)
    loss_mode = options.get('loss_mode')
    print("loss_mode ", loss_mode)
    # define and init model
    model, model_name = construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path)
    num_params = get_model_numparams(model)
    print("model name: ", model_name)
    print("num params: ", num_params)
    print('model:\n ', model)
    # setup optimizer
    reg_type = options.get('reg_type')
    if(reg_type == 'l2'):
        optimizer = config.optimizer(model.parameters(), weight_decay=config.weight_decay)

    loss_func = torch.nn.NLLLoss(ignore_index=PADDSYMB_INDX, weight=class_weights, size_average=True)
    rgrad_mode = options.get('restrict_grad_mode')
    rgrad_limit = options.get('restrict_grad_limit')
    target_dsets = regenerate_dataset(target_dsets, to_gpu, memmap)
    # load dataloaders with the relevant entities
    data_loaders, dsettypes, epoch_loss_avgbatch, epoch_loss_avgsamples, flog_out, score_dict, featimp_analysis = construct_load_dataloaders(target_dsets, config, wrk_dir, options)
    # write description of the current training run   
    ReaderWriter.write_log("pid: {}".format(pid) + "\n" + "number of params: {}".format(num_params) + "\n" + str(model) + "\n" + str(config) + "\n", os.path.join(wrk_dir, 'description.txt'))

    alpha = config.alpha_param
    print("original alpha: ", alpha)
    if(loss_mode == 'LastHF'): # total loss is only last event loss
        alpha = 1 # (1-alpha)index_event_loss + alpha*last_event_loss
    # print("alpha: ", alpha)
    targettag_indx = 1
    # dump the model configuration on disk
    ReaderWriter.dump_data(config, os.path.join(wrk_dir, 'config.pkl'))

    for epoch in range(num_epochs):
        print("epoch: {}, pid: {}, model_name: {}".format(epoch, pid, model_name))
        for dsettype in dsettypes:
            pred_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            prob_score = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
            indexevent_indic = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            pred_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            prob_score_levent = torch.Tensor([PADDSYMB_INDX]).type(fdtype)

            pindx = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            seqslen = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            
            data_loader = data_loaders[dsettype]
            epoch_loss = 0.
            if(dsettype == 'train'):
                model.train()
            else:
                model.eval()
                
            for i_batch, sample_batch in enumerate(data_loader):
#                 print("i_batch: ", i_batch)
#                 print("sample_batch \n", sample_batch)
                seqs_batch, labels_batch, seqs_len, indexevent_batch, patientsindx_batch = sample_batch
                seqs_len = seqs_len.view(-1) # make sure it is a column vector size (n,)
                patientsindx_batch = patientsindx_batch.type(ldtype)
#                 batch_size = seqs_batch.size(0)
#                 print("batch_size: ", batch_size)
                # clear gradients
                model.zero_grad()
                # reorder sequences in descending order
                seqs_len, perm_idx = seqs_len.sort(0, descending=True)
                seqs_len = seqs_len.type(ldtype)
                perm_idx = perm_idx.type(ldtype)
                seqs_batch = seqs_batch[perm_idx]
                labels_batch = labels_batch[perm_idx]
                indexevent_batch = indexevent_batch[perm_idx]
#                 print(config.batch_size)
#                 print("patientsindx_batch size ", patientsindx_batch.size(), type(patientsindx_batch))
#                 print("permindx size ", perm_idx.size(), type(perm_idx))
#                 print('perm_idx \n', perm_idx)
                patientsindx_batch = patientsindx_batch[perm_idx]
                # max sequence length
                max_seqlen = seqs_len[0]
                # wrap with autograd.Variable
                data_var_batch = autograd.Variable(seqs_batch, requires_grad=featimp_inspect)
                target_var_batch = autograd.Variable(labels_batch)
#                 print("data_var_batch \n", data_var_batch)
#                 print("target_var_batch \n", target_var_batch)
#                 print("seqs len \n", seqs_len)
#                 print("max sequence len: ", max_seqlen)
                # forward pass.
                tag_logprobs, tag_prob = model(data_var_batch, seqs_len, tagprob=True)
#                 print(tag_prob.shape)
#                 print(tag_prob)
                __, pred_tagsindx = torch.max(tag_logprobs, 1)
#                 print("tag_logprobs \n", tag_logprobs)
#                 print("pred_tagsindx \n", pred_tagsindx)
                
                pindx = torch.cat((pindx, patientsindx_batch))    
                seqslen = torch.cat((seqslen, seqs_len))
                # case of max sequence length in the batch is smaller than the max sequence in dataset
                if(max_seqlen < seqs_batch.size(1)): 
#                     print("max_seqlen < seqs_batch.size(1)")
                    target_classes = target_var_batch[:,:max_seqlen].contiguous().view(-1)
                    indexevent_flags = indexevent_batch[:,:max_seqlen].contiguous().view(-1)
                else:
#                     print("max_seqlen >= seqs_batch.size(1)")
                    target_classes = target_var_batch.view(-1)
                    indexevent_flags = indexevent_batch.view(-1)

#                 print("target_classes reshaped \n", target_classes)
#                 print("indexevents reshaped \n", indexevent_flags)

                # loss_0 refers to loss computed from non-HF events
                # loss_1 refers to loss computed from HF events (to consider renaming these variables)
                if(loss_mode == 'Convex_HF_NonHF'):
                    loss_0 = autograd.Variable(torch.zeros(1).type(fdtype), requires_grad=True)
                    cond_0 = np.where(indexevent_flags.cpu().numpy() == 0)[0]
                    if(cond_0.size):
                        cond_0 = torch.from_numpy(cond_0).type(ldtype)
#                         print("cond_0 \n", cond_0)
                        loss_0 = loss_0 + loss_func(tag_logprobs[cond_0], target_classes[cond_0])
#                         print("loss_0: ", loss_0)

                loss_1 = autograd.Variable(torch.zeros(1).type(fdtype), requires_grad=True)
                cond_1 = np.where(indexevent_flags.cpu().numpy() == 1)[0]

                if(cond_1.size):
                    cond_1 = torch.from_numpy(cond_1).type(ldtype)
#                     print("cond_1 \n", cond_1)
#                     print("tag_logprobs[cond_1]: \n", tag_logprobs[cond_1])
                    loss_1 = loss_1 + loss_func(tag_logprobs[cond_1], target_classes[cond_1])
#                     print("loss_1: ", loss_1)
#                 loss = (1-beta)*((1-alpha)*loss_0 + alpha*loss_1) + beta*loss_levent

                # identify last events indices
                # ------------------------------------
                offset = max_seqlen - seqs_len
#                 print("offset \n", offset)
                if(offset.size(0)>1): 
                    offset_shift = torch.zeros(offset.size(0)).type(ldtype)
                    offset_shift[1:] = offset[0:-1]
                else: # case where the batch is of size 1
                    offset_shift = offset
#                 print("offset shift \n", offset_shift)
                levent_indx = seqs_len + offset_shift
#                 print("levent_indx \n", levent_indx)
                levent_indx = torch.cumsum(levent_indx,0) - 1
#                 print("levent_indx \n", levent_indx)
#                 print("target_classes last event \n", target_classes[levent_indx])
                # ------------------------------------

                # compute loss from last events (i.e. target last HF events)
                if(loss_mode == 'Convex_HF_LastHF' or loss_mode == 'LastHF'):
                    loss_levent = loss_func(tag_logprobs[levent_indx], target_classes[levent_indx]) # compute last event loss
                    loss = (1-alpha)*loss_1 + alpha*loss_levent
                    
                elif(loss_mode == 'Uniform_HF'):
                    loss = loss_1
                    
                elif(loss_mode == 'Convex_HF_NonHF'):
                    loss = (1-alpha)*loss_0 + alpha*loss_1
                    
                if(featimp_inspect):
                    loss.backward()
                    print("analyzing feature contribution")
                    f_df = create_featanalysis_df(data_var_batch, target_classes.data, pred_tagsindx.data,
                                                  indexevent_flags, seqs_len,
                                                  patientsindx_batch,
                                                  target_dsets[dsettype].pdt_object.idx_mapper_inverse)
                    featimp_analysis[dsettype] = pd.concat([featimp_analysis[dsettype], f_df], axis=0, ignore_index=True)
                    
                elif(dsettype == 'train'):
                    # backward step
                    loss.backward()
                    # apply grad clipping
                    if(rgrad_mode):
                        restrict_grad_(model.parameters(), rgrad_mode, rgrad_limit)
                    # optimzer step -- update gradients
                    optimizer.step()

                epoch_loss += loss.data[0]
                
#                 print(tag_prob[levent_indx,])
#                 print(tag_prob.data[levent_indx,:])
#                 tmp = tag_prob.data[levent_indx,:]
#                 print(tmp[:, 1])
#                 print(tag_prob.data[levent_indx,:][:,1])
#                 print(tag_prob.data[:,1])
#                 print(target_classes.data[levent_indx])
                pred_target = torch.cat((pred_target, pred_tagsindx.data))
                ref_target = torch.cat((ref_target, target_classes.data))
                indexevent_indic = torch.cat((indexevent_indic, indexevent_flags))
                prob_score = torch.cat((prob_score, tag_prob.data[:, targettag_indx]))

                pred_target_levent = torch.cat((pred_target_levent, pred_tagsindx.data[levent_indx]))
                ref_target_levent = torch.cat((ref_target_levent, target_classes.data[levent_indx]))
                prob_score_levent = torch.cat((prob_score_levent, tag_prob.data[levent_indx,:][:,targettag_indx]))
                    
            if(dsettype == 'test' and dec_outdir):        
                write_seqprediction_df(ref_target, pred_target, prob_score, 
                                       indexevent_indic, seqslen, pindx, 
                                       target_dsets[dsettype].pdt_object.idx_mapper_inverse,
                                       model_name, fold_id, dec_outdir)
                
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss/len(data_loader.dataset))
#             print("average epoch loss (over batches): ", epoch_loss_avgbatch[dsettype][-1])
#             print("average epoch loss (over train samples): ", epoch_loss_avgsamples[dsettype][-1])
            # classification report
            weighted_f1_indexevent, auc_indexevent = metric_report(pred_target, ref_target, prob_score, indexevent_indic, epoch+1, flog_out[dsettype], PADDSYMB_INDX)
            weighted_f1_levent, auc_levent = metric_report_levent(pred_target_levent, ref_target_levent, prob_score_levent, flog_out[dsettype], PADDSYMB_INDX)
            avg_f1 = (weighted_f1_indexevent + weighted_f1_levent)/2     
            perf = auc_levent       
            if(perf > score_dict[dsettype][-1]):
                score_dict[dsettype] = (epoch, avg_f1, weighted_f1_indexevent, weighted_f1_levent, auc_indexevent, auc_levent)
                if(dsettype == 'validation'):
                    torch.save(model.state_dict(), os.path.join(wrk_dir, 'bestmodel.pkl'))
    if(num_epochs>1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir)

    dump_scores(score_dict, wrk_dir)

    if(featimp_inspect):
        ReaderWriter.dump_data(featanalysis, os.path.join(wrk_dir, 'featanalysis_dict.pkl'))
    ReaderWriter.write_log("finished {}\n".format(os.path.basename(wrk_dir)), os.path.join(os.path.dirname(wrk_dir), 'out.log'), mode='a')

def rnnss_run(config, model_class, target_dsets, y_codebook, num_epochs, wrk_dir,
              rank, to_gpu, memmap, modelstatedict_path=None, options={}):
    """train/eval/test RNN model
    """
    pid = "{}-{}".format(os.getpid(), rank) # process id description
    print("process id ", pid)
    fdtype, ldtype = get_tensordtypes(to_gpu)
    dec_outdir = options.get('dec_outdir')
    # print("decoded_outputdir ", dec_outdir)
    fold_id = options.get('fold_id')
    class_weights = options.get('class_weights')
    if(class_weights):
        # assign the same weight for 0 label/state to __START__ state
        class_weights = torch.Tensor(list(class_weights)+[class_weights[0]]).type(fdtype) 
    else:
        class_weights = torch.Tensor([1,1,1]).type(fdtype)

    print("class_weights: ", class_weights)
    loss_mode = options.get('loss_mode')
    print("loss_mode ", loss_mode)
    # define and init model
    model, model_name = construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path)
    num_params = get_model_numparams(model)
    print("model name: ", model_name)
    print("num params: ", num_params)
    print('model:\n ', model)
    # setup optimizer
    reg_type = options.get('reg_type')
    if(reg_type == 'l2'):
        optimizer = config.optimizer(model.parameters(), weight_decay=config.weight_decay)

    loss_func = torch.nn.NLLLoss(ignore_index=PADDSYMB_INDX, weight=class_weights, size_average=True)
    rgrad_mode = options.get('restrict_grad_mode')
    rgrad_limit = options.get('restrict_grad_limit')
    target_dsets = regenerate_dataset(target_dsets, to_gpu, memmap)
    # load dataloaders with the relevant entities
    data_loaders, dsettypes, epoch_loss_avgbatch, epoch_loss_avgsamples, flog_out, score_dict, featimp_analysis = construct_load_dataloaders(target_dsets, config, wrk_dir, options)
    # write description of the current training run   
    ReaderWriter.write_log("pid: {}".format(pid) + "\n" + "number of params: {}".format(num_params) + "\n" + str(model) + "\n" + str(config) + "\n", os.path.join(wrk_dir, 'description.txt'))

    alpha = config.alpha_param
    print("original alpha: ", alpha)
    if(loss_mode == 'LastHF'): # total loss is only last event loss
        alpha = 1 # (1-alpha)index_event_loss + alpha*last_event_loss
    # print("alpha: ", alpha)
    targettag_indx = 1

    schedule_counter = 0
    track_scheduler = []
    if('train' in dsettypes):
        scheduler = scheduled_sampling(config.sampling_func, len(data_loaders['train']), num_epochs)

    # dump the model configuration on disk
    ReaderWriter.dump_data(config, os.path.join(wrk_dir, 'config.pkl'))  

    for epoch in range(num_epochs):
        print("epoch: {}, pid: {}, model_name: {}".format(epoch, pid, model_name))
        for dsettype in dsettypes:
            pred_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            indexevent_indic = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            pred_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            prob_score = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
            prob_score_levent = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
            
            pindx = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            seqslen = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            
            data_loader = data_loaders[dsettype]
            epoch_loss = 0.
            if(dsettype == 'train'):
                model.train()
            else:
                model.eval()
            for i_batch, sample_batch in enumerate(data_loader):
#                 print("i_batch: ", i_batch)
#                 print("sample_batch \n", sample_batch)
                seqs_batch, labels_batch, seqs_len, indexevent_batch, patientsindx_batch= sample_batch
                seqs_len = seqs_len.view(-1) # make sure it is a column vector size (n,)
                patientsindx_batch = patientsindx_batch.type(ldtype)
                batch_size = seqs_batch.size(0)

                model.zero_grad()

                tag_logprobs = autograd.Variable(torch.Tensor([PADDSYMB_INDX]).type(fdtype)).expand(1, len(y_codebook))
                tag_probs = torch.Tensor([PADDSYMB_INDX]).type(fdtype)
                target_classes = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
                indexevent_flags = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
                pred_tagsindx = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
#                 a = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
#                 b = torch.Tensor([PADDSYMB_INDX]).type(ldtype)

                pindx = torch.cat((pindx, patientsindx_batch))    
                seqslen = torch.cat((seqslen, seqs_len))
                if(dsettype == 'train'):
                    e_i = scheduler(schedule_counter)
                    track_scheduler.append(e_i)
                    schedule_counter+=1
                    model.train()
                else:
                    e_i = 0 # use always the predicted states from the model
                    model.eval()
                for bindx in range(batch_size):
                    # we will pass each sequence independently
#                     print("bindx: ", bindx)
                    current_seq = torch.unsqueeze(seqs_batch[bindx, :seqs_len[bindx],:], 0)
                    labels = labels_batch[bindx, :seqs_len[bindx]]
                    index_events = indexevent_batch[bindx, :seqs_len[bindx]]
#                     print("current seq \n", current_seq)
#                     print("labels \n", labels)
#                     print("index_events \n", index_events)
                    logscores, probscores, bestlogscores_indx = model.forward(current_seq, labels, e_i, tagprob=True)
                    tag_logprobs = torch.cat((tag_logprobs, logscores))
                    pred_tagsindx = torch.cat((pred_tagsindx, bestlogscores_indx))
                    target_classes = torch.cat((target_classes, labels))
                    indexevent_flags = torch.cat((indexevent_flags, index_events))
#                     print("probscores: \n", probscores)
#                     print("probscores.data[1:, pos_label]: \n",probscores.data[:, pos_label])
#                     print("probscores.data[-1:, pos_label]: \n", probscores.data[-1:, pos_label])
                    tag_probs = torch.cat((tag_probs, probscores.data[:, targettag_indx]))

#                     a = torch.cat((a, pred_tagsindx[-1:]))
#                     b = torch.cat((b, target_classes[-1:]))
#                     print("tag_logprobs \n", tag_logprobs)
#                     print("target_classes \n", target_classes)
#                     print("pred_tagsindx \n", pred_tagsindx)
#                     print("indexevent_flags \n", indexevent_flags)
#                     print("a \n", a[1:])
#                     print("b \n", b[1:])
#                     print("-"*20)

                # wrap target_classess with variable
                target_classes = autograd.Variable(target_classes)

                # loss_0 refers to loss computed from non-HF events
                # loss_1 refers to loss computed from HF events (to consider renaming these variables)
                if(loss_mode == 'Convex_HF_NonHF'):
                    loss_0 = autograd.Variable(torch.zeros(1).type(fdtype), requires_grad=True)
                    cond_0 = np.where(indexevent_flags.cpu().numpy() == 0)[0]
                    if(cond_0.size):
                        cond_0 = torch.from_numpy(cond_0).type(ldtype)
#                         print("cond_0 \n", cond_0)
                        loss_0 = loss_0 + loss_func(tag_logprobs[cond_0], target_classes[cond_0])
#                         print("loss_0: ", loss_0)

                loss_1 = autograd.Variable(torch.zeros(1).type(fdtype), requires_grad=True)
                cond_1 = np.where(indexevent_flags.cpu().numpy() == 1)[0]

                if(cond_1.size):
                    cond_1 = torch.from_numpy(cond_1).type(ldtype)
#                     print("cond_1 \n", cond_1)
#                     print("tag_logprobs[cond_1]: \n", tag_logprobs[cond_1])
                    loss_1 = loss_1 + loss_func(tag_logprobs[cond_1], target_classes[cond_1])
#                     print("loss_1: ", loss_1)
#                 loss = (1-beta)*((1-alpha)*loss_0 + alpha*loss_1) + beta*loss_levent

                # identify last events indices
                # ------------------------------------
                levent_indx = torch.cumsum(seqs_len,0).type(ldtype)

                # compute loss from last events (i.e. target last HF events)
                if(loss_mode == 'Convex_HF_LastHF' or loss_mode == 'LastHF'):
                    loss_levent = loss_func(tag_logprobs[levent_indx], target_classes[levent_indx]) # compute last event loss
                    loss = (1-alpha)*loss_1 + alpha*loss_levent
                    
                elif(loss_mode == 'Uniform_HF'):
                    loss = loss_1
                    
                elif(loss_mode == 'Convex_HF_NonHF'):
                    loss = (1-alpha)*loss_0 + alpha*loss_1
                    
                if(dsettype == 'train'):
                    # backward step
                    loss.backward()
                    # apply grad clipping
                    if(rgrad_mode):
                        restrict_grad_(model.parameters(), rgrad_mode, rgrad_limit)
                    # optimzer step -- update gradients
                    optimizer.step()

                epoch_loss += loss.data[0]

                pred_target = torch.cat((pred_target, pred_tagsindx))
                ref_target = torch.cat((ref_target, target_classes.data))
                indexevent_indic = torch.cat((indexevent_indic, indexevent_flags))
                prob_score = torch.cat((prob_score, tag_probs)) 
#                 print("prob score: \n", prob_score)
#                 print("tag_probs: \n", tag_probs)
#                 print("seqlen: \n", seqs_len)
#                 print(tag_probs[levent_indx])
                pred_target_levent = torch.cat((pred_target_levent, pred_tagsindx[levent_indx]))
                ref_target_levent = torch.cat((ref_target_levent, target_classes.data[levent_indx]))
                prob_score_levent = torch.cat((prob_score_levent, tag_probs[levent_indx])) 

#                 print("levent_indx \n", levent_indx)
#                 print("pred_target_levent \n", pred_tagsindx[levent_indx])
#                 print("ref_target_levent \n", target_classes.data[levent_indx])
#                 assert torch.eq(a[1:], pred_tagsindx[levent_indx]).all()
#                 assert torch.eq(b[1:], target_classes.data[levent_indx]).all()

                    
            if(dsettype == 'test' and dec_outdir):        
                write_seqprediction_df(ref_target, pred_target, prob_score, 
                                       indexevent_indic, seqslen, pindx, 
                                       target_dsets[dsettype].pdt_object.idx_mapper_inverse,
                                       model_name, fold_id, dec_outdir)
                
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss/len(data_loader.dataset))
#             print("average epoch loss (over batches): ", epoch_loss_avgbatch[dsettype][-1])
#             print("average epoch loss (over train samples): ", epoch_loss_avgsamples[dsettype][-1])

            # classification report
            weighted_f1_indexevent, auc_indexevent = metric_report(pred_target, ref_target, prob_score, indexevent_indic, epoch+1, flog_out[dsettype], PADDSYMB_INDX)
            weighted_f1_levent, auc_levent = metric_report_levent(pred_target_levent, ref_target_levent, prob_score_levent, flog_out[dsettype], PADDSYMB_INDX)
            avg_f1 = (weighted_f1_indexevent + weighted_f1_levent)/2
            perf = auc_levent
            if(perf > score_dict[dsettype][-1]):
                score_dict[dsettype] = (epoch, avg_f1, weighted_f1_indexevent,weighted_f1_levent, auc_indexevent, auc_levent)
                if(dsettype == 'validation'):
                    torch.save(model.state_dict(), os.path.join(wrk_dir, 'bestmodel.pkl'))    

    if(num_epochs>1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir)
        plot_scheduler(track_scheduler, wrk_dir)
    dump_scores(score_dict, wrk_dir)
    ReaderWriter.write_log("finished {}\n".format(os.path.basename(wrk_dir)), os.path.join(os.path.dirname(wrk_dir), 'out.log'), mode='a')
    
def cnn_nn_run(config, model_class, target_dsets, y_codebook, num_epochs, wrk_dir,
            rank, to_gpu, memmap, modelstatedict_path=None, options={}):
    """train/eval/test CNN and CNNWide models
    """
    
    pid = "{}-{}".format(os.getpid(), rank) # process id description
    print("process id ", pid)
    fdtype, ldtype = get_tensordtypes(to_gpu)
    dec_outdir = options.get('dec_outdir')
    # print("decoded_outputdir ", dec_outdir)
    fold_id = options.get('fold_id')
    class_weights = options.get('class_weights')
    if(class_weights):
        class_weights = torch.Tensor(list(class_weights)).type(fdtype) # update class weights to float tensor
    else:
        class_weights = torch.Tensor([1,1]).type(fdtype) # weighting all casess equally
    print("class_weights: ", class_weights)

    # define and init model
    model, model_name = construct_load_model(config, model_class, y_codebook, options, to_gpu, modelstatedict_path)
    num_params = get_model_numparams(model)
    print("model name: ", model_name)
    print("num params: ", num_params)
    # print('model:\n ', model)
    # setup optimizer
    reg_type = options.get('reg_type')
    if(reg_type == 'l2'):
        optimizer = config.optimizer(model.parameters(), weight_decay=config.weight_decay)

    loss_func = torch.nn.NLLLoss(ignore_index=PADDSYMB_INDX, weight=class_weights, size_average=True)
    rgrad_mode = options.get('restrict_grad_mode')
    rgrad_limit = options.get('restrict_grad_limit')
    target_dsets = regenerate_dataset(target_dsets, to_gpu, memmap)
    # load dataloaders with the relevant entities
    data_loaders, dsettypes, epoch_loss_avgbatch, epoch_loss_avgsamples, flog_out, score_dict, featimp_analysis = construct_load_dataloaders(target_dsets, config, wrk_dir, options)
    # write description of the current training run   
    ReaderWriter.write_log("pid: {}".format(pid) + "\n" + "number of params: {}".format(num_params) + "\n" + str(model) + "\n" + str(config) + "\n", os.path.join(wrk_dir, 'description.txt'))
    # target label (i.e. 1)
    targettag_indx = 1
    if(dec_outdir):
        out_df = pd.DataFrame()
    ReaderWriter.dump_data(config, os.path.join(wrk_dir, 'config.pkl'))

    for epoch in range(num_epochs):
        print("epoch: {}, pid: {}, model_name: {}".format(epoch, pid, model_name))
        for dsettype in dsettypes:
            pred_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            ref_target_levent = torch.Tensor([PADDSYMB_INDX]).type(ldtype)
            prob_score_levent = torch.Tensor([PADDSYMB_INDX]).type(fdtype)

            data_loader = data_loaders[dsettype]
            epoch_loss = 0.
            if(dsettype == 'train'):
                model.train()
            else:
                model.eval()
                
            for i_batch, sample_batch in enumerate(data_loader):
                seqs_batch, labels_batch, seqs_len, indexevent_batch, patientsindx_batch = sample_batch
                seqs_len = seqs_len.view(-1) # make sure it is a column vector size (n,)
                patientsindx_batch = patientsindx_batch.type(ldtype)
                # clear gradients
                model.zero_grad()
                if(model_name == 'NN_Labeler'):
                    seqs_batch = seqs_batch.gather(1, seqs_len.view(-1, 1, 1).expand(seqs_batch.size(0), 1, seqs_batch.size(2))-1)
                    data_var_batch = autograd.Variable(seqs_batch.squeeze(1))
                else:
                    # wrap with autograd.Variable
                    data_var_batch = autograd.Variable(seqs_batch.unsqueeze(1))

                # select the last index event labels batch
                target_classes  = labels_batch.gather(1, seqs_len.view(-1,1)-1).view(-1)
                target_classes = autograd.Variable(target_classes)
#                 target_classes = target_classes.view(-1)
#                 print("data_var_batch \n", data_var_batch)
#                 print("target_var_batch \n", labels_batch)
#                 print("target classes \n", target_classes)
#                 print("seqs len \n", seqs_len)
                # forward pass
                tag_logprobs, tag_prob = model(data_var_batch, tagprob=True)
#                 print(tag_prob.shape)
#                 print(tag_prob)
                __, pred_tagsindx = torch.max(tag_logprobs, 1)
#                 print("tag_logprobs \n", tag_logprobs)
#                 print("pred_tagsindx \n", pred_tagsindx)
                
                loss = loss_func(tag_logprobs, target_classes)
#                 print("loss\n", loss)
                epoch_loss += loss.data[0]
                    
                if(dsettype == 'train'):
                    # backward step
                    loss.backward()
                    # apply grad clipping
                    if(rgrad_mode):
                        restrict_grad_(model.parameters(), rgrad_mode, rgrad_limit)
                    # optimzer step -- update gradients
                    optimizer.step()
          
                pred_target_levent = torch.cat((pred_target_levent, pred_tagsindx.data))
                ref_target_levent = torch.cat((ref_target_levent, target_classes.data))
                prob_score_levent = torch.cat((prob_score_levent, tag_prob.data[:,targettag_indx]))

                if(dsettype == 'test' and dec_outdir):
                    # get predictions of last event
                    pred_df = generate_predictions_df(target_classes.data, 
                                                      pred_tagsindx.data, 
                                                      tag_prob.data,
                                                      seqs_len,
                                                      patientsindx_batch,
                                                      target_dsets[dsettype].pdt_object.idx_mapper_inverse, 
                                                      model_name, fold_id)
                    out_df = pd.concat([out_df, pred_df], axis=0, ignore_index=True)

            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss/len(data_loader.dataset))
            # classification report
            weighted_f1_levent, auc_levent = metric_report_levent(pred_target_levent, ref_target_levent, prob_score_levent, flog_out[dsettype], PADDSYMB_INDX)
            avg_f1 = None # None is converted automatically to np.nan when using it in np.array   
            perf = auc_levent       
            if(perf > score_dict[dsettype][-1]):
                score_dict[dsettype] = (epoch, avg_f1, None, weighted_f1_levent, None, auc_levent)
                if(dsettype == 'validation'):
                    torch.save(model.state_dict(), os.path.join(wrk_dir, 'bestmodel.pkl'))
    if(num_epochs>1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir)
    # dump performance scores of the model on disk
    dump_scores(score_dict, wrk_dir)

    if(dec_outdir):
        out_path = os.path.join(dec_outdir, "{}_{}.txt".format(model_name, fold_id))
        dump_df(out_df, out_path)
    ReaderWriter.write_log("finished {}\n".format(os.path.basename(wrk_dir)), os.path.join(os.path.dirname(wrk_dir), 'out.log'), mode='a')


def dump_scores(score_dict, wrk_dir):
    for dsettype in score_dict:
        ReaderWriter.dump_data(score_dict[dsettype], os.path.join(wrk_dir, 'bestscore_{}.pkl'.format(dsettype)))
  
# get scores
def get_scores_fromdisk(target_dir, dsettype):
    parent_folder = os.path.basename(target_dir)
    scores = {}
    for path, folder_names, file_names in os.walk(target_dir):
        curr_foldername = os.path.basename(path)
        if(curr_foldername not in {parent_folder, 'best_model'} and curr_foldername[0]!='.'):
#             print(path)
#             print(folder_names)
#             print(file_names)
            target_path = os.path.join(path, 'bestscore_{}.pkl'.format(dsettype))
            scores[curr_foldername] = ReaderWriter.read_data(target_path)
#             print("-"*25)
    return(scores)

def get_models_dir(target_dir):
    parent_folder = os.path.basename(target_dir)
    model_dirs = []
    for path, folder_names, file_names in os.walk(target_dir):
        curr_foldername = os.path.basename(path)
        if(curr_foldername not in {parent_folder, 'best_model'} and curr_foldername[0]!='.'):
            model_dirs.append(os.path.join(project_dir, parent_folder, curr_foldername))
    return(model_dirs)

def get_perfscores(model_dirs, dsettype, outlog, wrk_dir):
    num_models = len(model_dirs)
    bestscores_dict = {}
    targetmetrix_indx = -1 # target metric score
    # (best_epoch, avg_weighted_f1, weighted_f1_indexevents, weighted_f1_lastevent, auc_indexevents, auc_lastevent)
    scores = np.zeros((num_models, 6)) 
    for i in range(num_models):
        target_path = os.path.join(model_dirs[i], 'bestscore_{}.pkl').format(dsettype)
        if(os.path.isfile(target_path)):
            res = ReaderWriter.read_data(target_path)
            for j in range(len(res)):
                scores[i, j] = res[j]
            model_name = os.path.basename(model_dirs[i])
            bestscores_dict[model_name] = scores[i, ]
        else:
            print(target_path, " is not found !!")

    line = "best scores in each trial\n {}\n".format(scores)
    if(wrk_dir):
        bestmodel_dir = create_directory("best_model", wrk_dir)
        
    bestscore_indxs = np.where(scores[:,targetmetrix_indx] == np.max(scores[:,targetmetrix_indx]))[0].tolist()
    for bestscore_indx in bestscore_indxs:
        model_name = os.path.basename(model_dirs[bestscore_indx])
        scoretuple = scores[bestscore_indx, ]
        line += "best score achieved in {} : {}\n".format(model_name, scoretuple)
        
        if(wrk_dir):
            for fname in ('bestmodel.pkl', 'config.pkl'):
                src = os.path.join(model_dirs[bestscore_indx], fname)
                dst = os.path.join(bestmodel_dir, '{}_{}'.format(os.path.basename(model_dirs[bestscore_indx]), fname))
                shutil.copyfile(src, dst)
    if(outlog):
        ReaderWriter.write_log(line, outlog, mode='a')
    return(bestscores_dict)
   
def compute_avgperf_acrossfolds(scores):
    res = {}
    last_indxevent_wf1 = []
    all_indxevent_wf1 = []
    all_auc = []
    last_auc = []
    for scorearray in scores.values():
        last_indxevent_wf1.append(scorearray[-3])
        all_indxevent_wf1.append(scorearray[-4])
        last_auc.append(scorearray[-1])
        all_auc.append(scorearray[-2])
    res['last_indxevent'] = [(np.mean(last_indxevent_wf1), np.std(last_indxevent_wf1)),
                             (np.mean(last_auc), np.std(last_auc))]
    res['all_indexevent'] = [(np.mean(all_indxevent_wf1), np.std(all_indxevent_wf1)),
                             (np.mean(all_auc), np.std(all_auc))]
    return(res)


def generate_predictions_df(ref_target, pred_target, prob_target, 
                            seqs_len, patients_indx, idx_mapper_inverse,
                            model_name, fold_id):
#     print("ref_target.shape ", ref_target.shape)
#     print("pred_target.shape ", pred_target.shape)
#     print("prob_target.shape ", prob_target.shape)
    ref_target = ref_target.cpu().numpy()
#     print("ref_target values: ", np.unique(ref_target))
#     cond = ref_target != PADDSYMB_INDX
#     ref_target = ref_target[cond]
    pred_target = pred_target.cpu().numpy()
#     print("pred_target values: ", np.unique(pred_target))
#     pred_target = pred_target[cond]
    prob_target = prob_target.cpu().numpy()
#     prob_target = prob_target[prob_target != PADDSYMB_INDX]
    
    pindx = patients_indx.cpu().numpy()
    pid = np.vectorize(idx_mapper_inverse.get)(pindx)
    df = pd.DataFrame()
    df['pid'] = pid
    df['pindx'] = pindx
    df['seq_len'] = seqs_len
    df['ref_target'] = ref_target
    df['pred_target'] = pred_target
    df['prob_target0'] = prob_target[:,0]
    df['prob_target1'] = prob_target[:,1]
    df['model_name'] = model_name
    df['fold_id'] = fold_id
#     print("df \n", df)
    return(df)

def dump_df(df, fpath, sep="\t"):
    f_out = open(fpath, 'a')
    f_out.write(sep.join(df.columns.tolist()) + "\n")
    f_out.close()
    df.to_csv(fpath, mode='a', index=False, header=False, sep=sep, na_rep='NaN')
       
       
def write_seqprediction_df(ref_target, pred_target, prob_target, 
                           indexevents_indic, seqs_len, patients_indx, 
                           idx_mapper_inverse, model_name, fold_id, out_dir):
    ref_target = ref_target.cpu().numpy()
    cond = ref_target != PADDSYMB_INDX
    ref_target = ref_target[cond]
    pred_target = pred_target.cpu().numpy()
    pred_target = pred_target[cond]
    prob_target = prob_target.cpu().numpy()
    prob_target = prob_target[cond]
    indexevents_indic = indexevents_indic.cpu().numpy()
    indexevents_indic = indexevents_indic[cond]
    
#     print("ref_target ", ref_target.shape)
#     print("pred_target ", pred_target.shape)
#     print("prob_target ", prob_target.shape)
#     print("indexevents_indic ", indexevents_indic.shape)
#     print(indexevents_indic)

    pindx = patients_indx.cpu().numpy()
    # ommit first element as it contains PADDSYMB_INDX by default
    pindx = pindx[1:] 
    pid = np.vectorize(idx_mapper_inverse.get)(pindx) # patients nrd_visitlink
#     print("pid ", pid.shape)
#     print("pindx ", pindx.shape)
#     print("pid ", pid)
#     print("pindx ",pindx)
    seqs_len = seqs_len.cpu().numpy()
    seqs_len = seqs_len[seqs_len != PADDSYMB_INDX]
#     print('pindx \n', pindx)
#     print("pid \n", pid)
#     print("seqs_len shape ", seqs_len.shape)
#     print("seqs_len length ", len(seqs_len))
#     print("seqs_len \n", seqs_len)
    pindx_stretch = []
    pid_stretch = []
    seqs_len_stretch = []
    
    for i in range(len(seqs_len)):
        seqlen = seqs_len[i]
        pid_stretch.extend([pid[i]]*seqlen)
        pindx_stretch.extend([pindx[i]]*seqlen)
        seqs_len_stretch.extend([seqlen]*seqlen)   
#     print("pid_stretch: \n", len(pid_stretch))
#     print("pindx_stretch: \n", len(pindx_stretch))
#     print("seqs_len_stretch: \n", len(seqs_len_stretch))
         
    df = pd.DataFrame()
    df['pid'] = pid_stretch
    df['pindx'] = pindx_stretch
    df['seq_len'] = seqs_len_stretch
    df['index_event'] = indexevents_indic
    df['ref_target'] = ref_target
    df['pred_target'] = pred_target
    df['prob_target1'] = prob_target
    df['model_name'] = model_name
    df['fold_id'] = fold_id     
    fpath = os.path.join(out_dir, "{}_{}.txt".format(model_name, fold_id))
    dump_df(df, fpath, sep="\t")
    
def create_featanalysis_df(data_var_batch, ref_target, pred_target,
                           indexevent_batch, seqs_len, patients_indx, idx_mapper_inverse):
    ref_target = ref_target.cpu().numpy()
    ref_target = ref_target[ref_target != PADDSYMB_INDX]
    pred_target = pred_target.cpu().numpy()
    pred_target = pred_target[pred_target != PADDSYMB_INDX]
    
#     print("patient indx \n", patients_indx)
    data_var_grad_batch = data_var_batch.grad
#     print("grad \n", data_var_grad_batch)
    df = pd.DataFrame()
    start = 0
    for bindx, seqlen in enumerate(seqs_len):
#         print("bindx ", bindx)
        pindx = patients_indx[bindx]
#         print("pindx ", pindx)
        # patient's id
        pid = idx_mapper_inverse[pindx]
#         print("pid ", pid)
        seq_input = data_var_batch[bindx, :seqlen, :].data
        seq_input_grad = data_var_grad_batch[bindx, :seqlen, :].data
        stop = seqlen + start
        ref_t = ref_target[start:stop]
        pred_t = pred_target[start:stop]
        start = stop
        seq_indexevent = indexevent_batch[bindx, :seqlen]
#         print("seq_input \n", seq_input)
#         print("seq_input_grad \n", seq_input_grad)
#         print("ref_t \n", ref_t)
#         print("pred_t \n", pred_t)
#         print("seq_indexevent \n", seq_indexevent)
        tdf = pd.DataFrame()
        input_dim = seq_input.size(-1)
        tdf['pid'] = [pid]*input_dim*seqlen
        tdf['xi_grad'] = seq_input_grad.view(-1).numpy()
        tdf['xi_value'] = seq_input.view(-1).numpy()
        tdf['index_event'] = seq_indexevent.numpy().repeat(input_dim)
        tdf['ref_target'] = ref_t.repeat(input_dim)
        tdf['pred_target'] = pred_t.repeat(input_dim)
        tdf['time'] = np.arange(seqlen).repeat(input_dim)
        tdf['xi_indx'] = np.tile(np.arange(input_dim),seqlen)
        tdf['seqlen'] = [seqlen]*input_dim*seqlen
        df = pd.concat([df, tdf], ignore_index=True, axis=0)   
#     print("df \n", df)
    return(df)

if __name__ == '__main__':
    pass