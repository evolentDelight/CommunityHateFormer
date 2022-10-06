from model import Graphormer
from data import GraphDataModule, get_dataset
import pytorch_lightning as pl
from argparse import ArgumentParser
from pprint import pprint

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser = Graphormer.add_model_specific_args(parser)
parser = GraphDataModule.add_argparse_args(parser)
args = parser.parse_args()
args.max_steps = args.tot_updates + 1
if not args.test and not args.validate:
    print(args)
pl.seed_everything(args.seed)
prefix = '/home/l2hebert/dev/Graphormer/exps/pcba/'

all_model = prefix + 'reddit-first/1/lightning_logs/checkpoints/last.ckpt'
the_donald = prefix + 'reddit-donald/1/lightning_logs/checkpoints/last.ckpt'
politics = prefix + 'reddit-politics/1/lightning_logs/checkpoints/last.ckpt'
iama = prefix + 'reddit-iama/1/lightning_logs/checkpoints/last.ckpt'
amItheasshole = prefix + 'reddit-amItheasshole/1/lightning_logs/checkpoints/last.ckpt'

data = [(all_model, None), (the_donald, 'The_Donald'), (politics, 'politics'), (iama, 'IAmA'), (amItheasshole, 'AmItheAsshole')]
dm = GraphDataModule(dataset_name='REDDIT', batch_size=16, seed=1)
for i in range(len(data)):
    print(f"===== starting to evaluate model {data[i][1]} =====")

    model = Graphormer.load_from_checkpoint(
            data[i][0],
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name='REDDIT',
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )

    for k in range(i, len(data)):
        print(f'** evaluating on {data[k][1]} **')
        dm.change_subreddit(data[k][1])        

        trainer = pl.Trainer(precision=32, max_epochs=10, gpus=1)

        trainer.test(model, dm)
