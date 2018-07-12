'''
@author: ahmed allam <ahmed.allam@nih.gov>
'''
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_Labeler(nn.Module):
    def __init__(self, y_codebook, conv_pipeline, fc_pipeline, hiddtotag_pipeline):
        super(CNN_Labeler, self).__init__()
        self.y_codebook = y_codebook
        self.tagset_size = len(y_codebook)
        self.conv_layer = self.generate_layers(conv_pipeline)
        self.fc_pipeline = self.generate_layers(fc_pipeline)
        self.hiddenTotag = self.generate_layers(hiddtotag_pipeline)
#         self._init_weights()
        
    def generate_layers(self, op_pipeline):
        layers = []
        for elm in op_pipeline:
            elm_name = elm[0]
            if(elm_name in {'MaxPool', 'AvgPool'}):
                __, kernel_size, stride, padding = elm
                if(elm_name == 'MaxPool'):
                    pool_class = nn.MaxPool2d
                elif(elm_name == 'AvgPool'):
                    pool_class = nn.AvgPool2d
                layers += [pool_class(kernel_size, stride=stride, padding=padding)]
            elif(elm_name == 'Conv'):
                __, in_channels, out_channels, kernel_size, stride, padding, bias = elm
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
            elif(elm_name in {'BatchNorm1d','BatchNorm2d'}):
                __, num_features = elm
                if(elm_name == 'BatchNorm1d'):
                    bnorm_class = nn.BatchNorm1d
                else:
                    bnorm_class = nn.BatchNorm2d
                layers += [bnorm_class(num_features)]
            elif(elm_name == 'NonLinearFunc'):
                __, nonlinear_func = elm
                layers += [nonlinear_func()]
            elif(elm_name == 'Linear'):
                __, in_features, out_features, bias = elm
                layers += [nn.Linear(in_features, out_features, bias=bias)]
            elif(elm_name in {'Dropout', 'Dropout2d'}):
                __, pdropout = elm
                if(elm_name == 'Dropout'):
                    dropout_class = nn.Dropout
                else:
                    dropout_class = nn.Dropout2d
                layers += [dropout_class(p=pdropout)]
        return(nn.Sequential(*layers))
    
    def forward(self, batch_seqs, tagprob=False):
        out = self.conv_layer(batch_seqs)
        out = out.view(out.size(0), -1)
        out = self.fc_pipeline(out)
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
            if(isinstance(m, nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif(isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif(isinstance(m, nn.Linear)):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()