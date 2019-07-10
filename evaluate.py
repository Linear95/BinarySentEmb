
"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

#from __future__ import absolute_import, division, unicode_literals

import sys
import os
import numpy as np
import torch
import logging

import discrete_encoders as DisEnc
import config

# Set PATHs
PATH_TO_INFERSENT = '../../InferSent-master/'         # path to Infersent
PATH_TO_SENTEVAL = '../../SentEval-master/'           # path to SentEval
PATH_TO_DATA = PATH_TO_SENTEVAL + 'data'              # path to transfer task datasets
PATH_TO_W2V = './dataset/GloVe/glove.840B.300d.txt'   # path to GloVe word embedding
PATH_TO_CONT_ENCODER = './encoder/infersent1.pkl'
PATH_TO_B_ENCODER = './encoder/bEncoder2048.pkl'


#assert os.path.isfile(INFERSENT_PATH) and os.path.isfile(PATH_TO_W2V), \    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval.engine as engine_cosine
#import senteval.engine_hamming as engine_hamming

sys.path.insert(0, PATH_TO_INFERSENT)
from models import InferSent

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

    
def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    embeddings = torch.from_numpy(embeddings).float().cuda()
    return params.autoencoder.encode(embeddings).data.cpu().numpy()


def hamming_similarity(s1,s2):
    return -np.sum(np.abs(s1-s2),axis = -1)


# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5,
                   'classifier' :{'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                  'tenacity': 5, 'epoch_size': 4}
                }
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(PATH_TO_CONT_ENCODER))
    model.set_w2v_path(PATH_TO_W2V)

    model_name = config.encoder_type
    if config.encoder_type == 'AE':
        dis_encoder = DisEnc.LinearAutoEncoder(config.dim)
        dis_encoder.load_state_dict(torch.load(PATH_TO_B_ENCODER))
        model_name = model_name + '_' + config.model_name #+'V'+str(config.INFERSENT_VERSION)
    elif config.encoder_type == 'PCA':
        dis_encoder = DisEnc.PCAEncoder(config.dim,config.PCA_LOAD_PATH)
    elif config.encoder_type == 'Random':
        dis_encoder = DisEnc.RandomEncoder(config.dim,config.RAN_LOAD_PATH)
    elif config.encoder_type == 'Id':
        dis_encoder = DisEnc.IdEncoder()
    elif config.encoder_type == 'HT':
        dis_encoder = DisEnc.HTEncoder(config.RAN_LOAD_PATH)


    print('testing '+model_name)
    
    params_senteval['infersent'] = model.cuda()
    params_senteval['autoencoder'] = dis_encoder.cuda()
    params_senteval['similarity'] = hamming_similarity

    if config.sim_type == 'cosine':
        se = engine_cosine.SE(params_senteval, batcher, prepare)
    elif config.sim_type == 'hamming':
        se = engine_hamming.SE(params_senteval, batcher, prepare)

    results = se.eval(['MR', 'CR','STS12', 'STS13', 'STS14', 'STS15', 'STS16','MRPC','SICKRelatedness','STSBenchmark','SICKEntailment','SICKRelatedness', 'MPQA', 'SUBJ', 'SST2', 'SST5']#,  'MRPC',

    print(results)
