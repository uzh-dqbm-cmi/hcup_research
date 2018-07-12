'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
import torch.nn as nn
import torch.nn.functional as F

class NN_Labeler(nn.Module):
    def __init__(self, y_codebook, fc_pipeline, hiddtotag_pipeline):
        super(NN_Labeler, self).__init__()
        self.y_codebook = y_codebook
        self.tagset_size = len(y_codebook)
        self.fc_pipeline = self.generate_layers(fc_pipeline)
        self.hiddenTotag = self.generate_layers(hiddtotag_pipeline)
#         self._init_weights()
        
    def generate_layers(self, op_pipeline):
        layers = []
        for elm in op_pipeline:
            elm_name = elm[0]
            if(elm_name == 'BatchNorm1d'):
                bnorm_class = nn.BatchNorm1d
                __, num_features = elm
                layers += [bnorm_class(num_features)]
            elif(elm_name == 'NonLinearFunc'):
                __, nonlinear_func = elm
                layers += [nonlinear_func()]
            elif(elm_name == 'Linear'):
                __, in_features, out_features, bias = elm
                layers += [nn.Linear(in_features, out_features, bias=bias)]
            if(elm_name == 'Dropout'):
                dropout_class = nn.Dropout
                __, pdropout = elm
                layers += [dropout_class(p=pdropout)]
        return(nn.Sequential(*layers))
    
    def forward(self, batch_seqs, tagprob=False):
        out = self.fc_pipeline(batch_seqs)
        out = self.hiddenTotag(out)
        tag_prob = None
        if(tagprob):
            tag_prob = F.softmax(out, dim=1)
        tag_scores = F.log_softmax(out, dim=1)
#         print("tag_space \n", out)
#         print()
#         print("tag_prob \n", tag_prob)
#         print("tag_scores log probability \n", tag_scores)
#         print()
        return(tag_scores, tag_prob)
    
    def _init_weights(self):
        for m in self.modules():
            if(isinstance(m, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif(isinstance(m, nn.Linear)):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()