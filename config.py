import argparse

parser = argparse.ArgumentParser()
# My arguments
parser.add_argument('--ds', type=int, default=1, help='dataset id')
parser.add_argument("--unsup", action="store_true", default=False, help="unsup")
parser.add_argument("--train_ratio", type=int, default=100, help="train_ratio")
parser.add_argument("--dual_no_time", action="store_true", default=False, help="use relation only")
parser.add_argument("--time_no_agg", action="store_true", default=False, help="Not aggregate time feature")
parser.add_argument("--sub_exps", action="store_true", default=False, help="test sim for every matrix")
parser.add_argument("--train_anyway", action="store_true", default=False, help="training over again")
parser.add_argument("--sep_eval", action="store_true", default=False, help="eval time sense/not sense separately")
# parser.add_argument('--lang', type=str, default='fr', help='dataset language (fr, de)')
# parser.add_argument('--k', type=int, default=1, help='mini-batch number')
# parser.add_argument('--it_round', type=int, default=1)
# parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
# parser.add_argument('--model', type=str, default='dual-large')
# parser.add_argument("--save_folder", type=str, default='tmp/')
# parser.add_argument("--step", type=int, default=1)
# parser.add_argument("--max_sinkhorn_sz", type=int, default=33000,
#                     help="max matrix size to run Sinkhorn iteration"
#                          ", if the matrix size is higher than this value"
#                          ", it will calculate kNN search without normalizing to avoid OOM"
#                          ", default is set for 33000^2 (for RTX3090)."
#                          " could be set to higher value in case there is GPU with larger memory")
# parser.add_argument("--gcn_max_iter", type=int, default=-1, help="max iteration of GCN for partition")
# parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
# parser.add_argument("--faiss_gpu", action="store_true", default=True, help="whether to use FAISS GPU")
# parser.add_argument("--norm", action="store_true", default=True, help="whether to normalize embeddings")


global_args = parser.parse_args()
print('IMPORTANT! current ARGs are', global_args)
unsup = global_args.unsup
which_file = global_args.ds
train_ratio = float(global_args.train_ratio) / 100

# filename = 'data/YAGO-WIKI50K/'

if which_file == 0:
    filename = 'data/ICEWS05-15/'
elif which_file == 1:
    filename = 'data/YAGO-WIKI50K/'
else:
    filename = 'data/YAGO-WIKI20K/'

