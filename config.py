lty = 'linear'
fc_dim = 2048
random_seed = 2019
struct_coef = 0.8 #0.1/discrete_dim
recons_coef = 1.
RAN_LOAD_PATH = '../nonlinear_ae/'
PCA_LOAD_PATH = RAN_LOAD_PATH
INFERSENT_PATH = './../../SentEval-master/examples/infersent1.pkl'
INFERSENT_VERSION = 1 # version of InferSent
dim = 2048
model_name = 'bEncoder2048.pkl'
#'linear_d2048_f2048epoch10l0.8n7ran2018MR_MRPC_best'#
PATH_TO_AE ='./encoder/'
encoder_type = 'AE'#'AE' #'PCA','Random' 'Id' 'HT'
sim_type = 'hamming'#'cosine' #'hamming'

transfer_tasks = ['MRPC','SICKRelatedness','STSBenchmark','SICKEntailment','SICKRelatedness','STS12', 'STS13', 'STS14', 'STS15', 'STS16','MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5']#,  'MRPC',
                      #'SICKEntailment', 'SICKRelatedness', 'STSBenchmark'] # 'TREC',
                      # 'Length', 'WordContent', 'Depth', 'TopConstituents',
                      # 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      # 'OddManOut', 'CoordinationInversion']

# transfer_tasks = ['MRPC','SICKRelatedness','STSBenchmark','SICKEntailment','STS14','MR','CR', 'MPQA', 'SUBJ', 'SST2']
                      #'SICKEntailment', 'SICKRelatedness', 'STSBenchmark'] # 'TREC',
                      # 'Length', 'WordContent', 'Depth', 'TopConstituents',
                      # 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      # 'OddManOut', 'CoordinationInversion']

# transfer_tasks = [ 'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
