
import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import InferSent

#from models import autoencoder
import discrete_encoders as DisEnc


model_name = "bEncoder"

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='./dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--word_emb_path", type=str, default="./dataset/GloVe/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default = 20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam,lr=0.00005", help="adam or sgd,lr=0.1")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--struct_coef", type=float, default=1e-5, help="coefficient for structural loss")


parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
#parser.add_argument("--sent_emb_dim", type=int, default=512, help="encoder output dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default= 0, help="GPU ID")
parser.add_argument("--seed", type=int, default= 2019 , help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

parser.add_argument("--dis_emb_dim", type=int, default = 1024, help = 'discrete_embedding_dim')

params, _ = parser.parse_known_args()

ae_lr = params.lr
# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

infersent_net = InferSent(config_nli_model)
print(infersent_net)

infersent_net.load_state_dict(torch.load('./encoder/infersent1.pkl'))
infersent_net.cuda()

for parameters_infer in infersent_net.parameters():
    parameters_infer.requires_grad =False


ae_model = DisEnc.LinearAutoEncoder(params.dis_emb_dim).cuda()

print(ae_model)

def cos_distance(a,b):
    return (1.-torch.nn.functional.cosine_similarity(a,b))

def hamming_distance(a,b):
    #return (a-b).abs().sum()
    return torch.nn.functional.pairwise_distance(a,b)

def mse(a,b):
    return ((a-b)*(a-b)).mean() 


optimizer = torch.optim.Adam(ae_model.parameters(), lr=ae_lr, weight_decay=1e-5)

"""
TRAIN
"""
val_mse_best = 100000
adam_stop = False
stop_training = False



def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    ae_model.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]

   # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
   #     and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
#    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        if params.batch_size > len(s1)-stidx:
            sub_batch_size = (len(s1)-stidx)//2
        else:
            sub_batch_size = params.batch_size // 2
        
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + sub_batch_size],
                                    word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + sub_batch_size],
                                    word_vec, params.word_emb_dim)
        s3_batch, s3_len = get_batch(s1[stidx + sub_batch_size :stidx + 2 * sub_batch_size],
                                    word_vec, params.word_emb_dim)
        s4_batch, s4_len = get_batch(s2[stidx + sub_batch_size :stidx + 2 * sub_batch_size],
                                    word_vec, params.word_emb_dim)

        s1_batch, s2_batch, s3_batch, s4_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda()), Variable(s3_batch.cuda()), Variable(s4_batch.cuda())

        # model forward
        cont_code1 = infersent_net((s1_batch, s1_len))
        cont_code2 = infersent_net((s2_batch, s2_len))
        
        cont_code3 = infersent_net((s3_batch, s3_len))
        cont_code4 = infersent_net((s4_batch, s4_len))

        output1 = ae_model(cont_code1)
        output2 = ae_model(cont_code2)
        output3 = ae_model(cont_code3)
        output4 = ae_model(cont_code4)

        discrete_code1 = ae_model.encode(cont_code1)
        discrete_code2 = ae_model.encode(cont_code2)
        discrete_code3 = ae_model.encode(cont_code3)
        discrete_code4 = ae_model.encode(cont_code4)
        
        cont_code_dist_1 = cos_distance(cont_code1,cont_code2)
        cont_code_dist_2 = cos_distance(cont_code3,cont_code4)
        discrete_code_dist1 = hamming_distance(discrete_code1,discrete_code2) # or cos
        discrete_code_dist2 = hamming_distance(discrete_code3,discrete_code4) # or cos

        struct_sign = ((cont_code_dist_1 -cont_code_dist_2) > 0.).float().cuda()
        struct_loss = (F.relu( (discrete_code_dist2-discrete_code_dist1)*(2*struct_sign-1.).detach()) ** 2).mean()
        recons_loss = (mse(cont_code1,output1) +mse(cont_code2,output2) +mse(cont_code3,output3) +mse(cont_code4,output4))/4.
        
        loss = params.struct_coef * struct_loss + recons_loss

        all_costs.append([struct_loss.sqrt().data.cpu().detach().numpy(),recons_loss.sqrt().data.cpu().detach().numpy()])
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        
        # optimizer step
        optimizer.step()
        #optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) % 100 == 0:
            losses = np.mean(all_costs, axis = 0)
            print('{0} at epoch {1} ; struct_loss {2}; recons_loss {3}'.format(stidx,epoch,losses[0],losses[1] ))
            all_costs = []
            
    train_acc = (np.mean(all_costs))
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch,(train_acc)))
    
    evaluate(epoch)
    return 0



def evaluate(epoch, eval_type='valid', final_eval=False):
    infersent_net.eval()
    ae_model.eval()
    correct = 0.
    global stop_training, adam_stop 

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = test['s1'] if eval_type == 'valid' else test['s1']
    s2 = test['s2'] if eval_type == 'valid' else test['s2']
    target = test['label'] if eval_type == 'valid' else test['label']

    all_costs = []

    
    for stidx in range(0, len(s1), params.batch_size):
        if params.batch_size > len(s1)-stidx:
            sub_batch_size = (len(s1)-stidx)//2
        else:
            sub_batch_size = params.batch_size // 2
        
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + sub_batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + sub_batch_size],
                                     word_vec, params.word_emb_dim)
        s3_batch, s3_len = get_batch(s1[stidx + sub_batch_size :stidx + 2 * sub_batch_size],
                                     word_vec, params.word_emb_dim)
        s4_batch, s4_len = get_batch(s2[stidx + sub_batch_size :stidx + 2 * sub_batch_size],
                                     word_vec, params.word_emb_dim)

        s1_batch, s2_batch, s3_batch, s4_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda()), Variable(s3_batch.cuda()), Variable(s4_batch.cuda())
        # tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        # model forward
        cont_code1 = infersent_net((s1_batch, s1_len))
        cont_code2 = infersent_net((s2_batch, s2_len))
        
        cont_code3 = infersent_net((s3_batch, s3_len))
        cont_code4 = infersent_net((s4_batch, s4_len))

        output1 = ae_model(cont_code1)
        output2 = ae_model(cont_code2)
        output3 = ae_model(cont_code3)
        output4 = ae_model(cont_code4)

        discrete_code1 = ae_model.encode(cont_code1)
        discrete_code2 = ae_model.encode(cont_code2)
        discrete_code3 = ae_model.encode(cont_code3)
        discrete_code4 = ae_model.encode(cont_code4)
        
        cont_code_dist_1 = cos_distance(cont_code1,cont_code2)
        cont_code_dist_2 = cos_distance(cont_code3,cont_code4)
        discrete_code_dist1 = hamming_distance(discrete_code1,discrete_code2) # or cos
        discrete_code_dist2 = hamming_distance(discrete_code3,discrete_code4) # or cos

        struct_sign = ((cont_code_dist_1 -cont_code_dist_2) > 0.).float().cuda()
        struct_loss = F.relu( (discrete_code_dist2-discrete_code_dist1)*(2*struct_sign-1.).detach()).mean()
        recons_loss = (mse(cont_code1,output1) +mse(cont_code2,output2) +mse(cont_code3,output3) +mse(cont_code4,output4))/4.
    
        # loss    
        all_costs.append([struct_loss.data.cpu().detach().numpy(),recons_loss.sqrt().data.cpu().detach().numpy()])
        
    losses = np.mean(all_costs, axis = 0)
    print('valid set at epoch{0} ; struct_loss {1}; recons_loss {2}'.format(epoch,losses[0],losses[1] ))

    return 0


"""
Train model on Natural Language Inference task
"""
epoch = 1


while  epoch <= params.n_epochs:# and (not stop_training):
    train_acc = trainepoch(epoch)
#    evaluate(epoch, 'valid')
    epoch += 1

