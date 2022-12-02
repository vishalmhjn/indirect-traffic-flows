import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM_Embedding(nn.Module):
    def __init__(self,
                 emb_dims_dynamic,
                 emb_dims_static,
                no_of_cat_static,
                no_of_continous_static,
                no_of_cat_dyn,
                no_of_continous_dyn,
                 lookback_length, 
                 hidden_size, 
                 lstm_layers, 
                 prediction_horizon,
                 fc1_size,
                fcm_size,
                 dropout):
        '''
        emb_dims: List of two element tuples
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        '''
        super(LSTM_Embedding, self).__init__()
        self.n_features_cat_stat = no_of_cat_static
        self.n_features_cont_stat = no_of_continous_static
        self.n_features_cat_dyn = no_of_cat_dyn
        self.n_features_cont_dyn = no_of_continous_dyn
        self.seq_len = lookback_length
        self.n_hidden = hidden_size # number of hidden states
        self.n_layers = lstm_layers # number of LSTM layers (stacked)
        self.fc1_layer_size = fc1_size
        self.fcm_layer_size = fcm_size
        self.output_size = prediction_horizon
        
        #embedding layer
        self.emb_layers_dyn = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims_dynamic*lookback_length])
        self.emb_layers_static = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims_static])
        no_of_embs = sum([y for x, y in emb_dims_dynamic]) + sum([y for x, y in emb_dims_static])
        self.no_of_embs = no_of_embs
        self.lstm = nn.LSTM(input_size = self.no_of_embs + \
                                               self.n_features_cont_stat + \
                                               self.n_features_cont_dyn, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.fc_1= nn.Linear(self.n_hidden*self.seq_len, self.fc1_layer_size)
        self.fc_m= nn.Linear(self.fc1_layer_size, self.fcm_layer_size)		
        self.fc_2= nn.Linear(self.fcm_layer_size, self.output_size)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x_cat_static, x_cont_static, 
                        x_cat_dynamic, x_cont_dynamic):
        if x_cat_static.shape[1]==0:
            x = [emb_layer(x_cat_dynamic.reshape(x_cat_dynamic.shape[0], -1)[:, i].to(torch.long)) 
                 for i, emb_layer in enumerate(self.emb_layers_dyn)]
            x = torch.cat(x, 1)
        elif x_cat_dynamic.shape[2]!=0:
            x_d = [emb_layer(x_cat_dynamic.reshape(x_cat_dynamic.shape[0], -1)[:, i].to(torch.long)) 
                 for i, emb_layer in enumerate(self.emb_layers_dyn)]
            x_s = [emb_layer(x_cat_static.reshape(x_cat_static.shape[0], -1)[:, i].to(torch.long)) 
                 for i, emb_layer in enumerate(self.emb_layers_static)]
            x_s = torch.cat([*x_s], 1)
            x_s = x_s.unsqueeze(1).expand(-1, self.seq_len, -1)
        
            x_d = torch.cat([*x_d], 1)
            x_d= x_d.reshape(x_d.shape[0], self.seq_len, -1)
            x = torch.cat([x_s, x_d], 2)
        else:
            x_s = [emb_layer(x_cat_static.reshape(x_cat_static.shape[0], -1)[:, i].to(torch.long)) 
                 for i, emb_layer in enumerate(self.emb_layers_static)]
            x_s = torch.cat([*x_s], 1)
            x_s = x_s.unsqueeze(1).expand(-1, self.seq_len, -1)
            x = x_s

        x_cont_static = x_cont_static.unsqueeze(1).expand(-1, self.seq_len, -1)
        x = torch.cat([x, x_cont_static, x_cont_dynamic], 2)
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.lstm(x.float(),self.hidden)
        x = lstm_out.contiguous().view(batch_size,-1)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc_1(x)
        x = nn.functional.relu(x)
        x = self.fc_m(x)
        x = nn.functional.relu(x)
        return self.fc_2(x)