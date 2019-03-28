import argparse
import os
import time
import torch
torch.backends.cudnn.enabled = False
import numpy as np
from spatialdata import SpatialMNISTDataset
from datasets import ShapeNet15kPointClouds
from spatialmodel import Statistician
from spatialplot import grid, visualize_point_clouds
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from metrics.evaluation_metrics_pytorch import compute_all_gen_metrics, _MMD_EMD_CD_one_to_one_
import scipy

# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')
parser.add_argument('--log-name', required=False, type=str, default=None,
                    help='Log-name for shits')
parser.add_argument('--cates', required=False, type=str, default=['airplane'], nargs='+',
                    help='Categories.')

# optional
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size (of datasets) for training (default: 64)')
parser.add_argument('--c-dim', type=int, default=64,
                    help='dimension of c variables (default: 64)')
parser.add_argument('--n-hidden-statistic', type=int, default=3,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=256,
                    help='dimension of hidden layers in statistic network (default: 256)')
parser.add_argument('--n-stochastic', type=int, default=3,
                    help='number of z variables in hierarchy (default: 3)')
parser.add_argument('--z-dim', type=int, default=2,
                    help='dimension of z variables (default: 2)')
parser.add_argument('--n-hidden', type=int, default=3,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 3)')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 256)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for training (default: 100)')
parser.add_argument('--viz-interval', type=int, default=1,
                    help='number of epochs between visualizing context space '
                         '(default: 1)')
parser.add_argument('--val-interval', type=int, default=1,
                    help='number of epochs between evaluation on validation set'
                         '(default: 1)')
parser.add_argument('--save-interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
args = parser.parse_args()
assert args.output_dir is not None
# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")
if args.log_name is None:
    args.log_name = "run_%s"%time_stamp
else:
    args.log_name += "_%s"%time_stamp
os.makedirs(os.path.join(args.output_dir, args.log_name, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, args.log_name, 'figures'), exist_ok=True)
log_path = os.path.join('runs', args.log_name)
os.makedirs(log_path, exist_ok=True)
writer = SummaryWriter(log_dir=log_path)


def evaluate(model, test_loader, writer, epoch, verbose=False):
    with torch.no_grad():
        sample_pcls = []
        recon_pcls = []
        ref_pcls = []
        for batch in test_loader:
            inputs = batch['train_points'].cuda().contiguous()
            B, N = inputs.size(0), inputs.size(1)
            model.eval()
            recons = model.sample_conditioned(inputs)
            samples = model.sample_unconditioned(N)
            samples = samples[:B]
            reference = batch['test_points'].cuda()

            # Denormalize
            m, s = data['mean'].float(), data['std'].float()
            m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
            recons = recons * s + m
            samples = samples * s + m
            reference = reference * s + m

            sample_pcls.append(samples)
            recon_pcls.append(recons)
            ref_pcls.append(reference)

        sample_pcls = torch.cat(sample_pcls, dim=0)
        recon_pcls = torch.cat(recon_pcls, dim=0)
        ref_pcls = torch.cat(ref_pcls, dim=0)

    gen_res = compute_all_gen_metrics(
        sample_pcls, ref_pcls, args.batch_size, gpu=None, accelerated_cd=True, verbose=verbose)
    if writer is not None:
        for k, v in gen_res.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            writer.add_scalar('val_gen/%s'%k, v, epoch)

    rec_res = _MMD_EMD_CD_one_to_one_(
        sample_pcls, ref_pcls, args.batch_size, gpu=None, accelerated_cd=True, verbose=verbose, reduced=True)
    if writer is not None:
        for k, v in rec_res.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            writer.add_scalar('val_rec/%s'%k, v, epoch)

    return gen_res, rec_res


def run(model, optimizer, loaders, datasets):
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders
    test_batch = next(iter(test_loader))

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    val_interval = args.epochs if args.val_interval == -1 else args.val_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # initial weighting for loss terms is (1 + alpha)
    alpha = 1

    # gen, rec = evaluate(model, test_loader, writer, -1, verbose=True)

    # main training loop
    tbar = tqdm(range(args.epochs))
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for bidx, batch in enumerate(train_loader):
            inputs = batch['train_points'].cuda().contiguous()
            vlb, loss = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            step = bidx + epoch * len(train_loader)
            writer.add_scalar("train/vlb", vlb, step)
            writer.add_scalar("train/loss", loss, step)
            running_vlb += vlb

        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        if (epoch + 1) % val_interval == 0:
            gen, rec = evaluate(model, test_loader, writer, epoch, verbose=False)

        # # show samples conditioned on test batch at intervals
        # model.eval()
        # with torch.no_grad():
        #     if (epoch + 1) % viz_interval == 0:
        #         inputs = Variable(test_batch[0].cuda())
        #         samples = model.sample_conditioned(inputs)
        #         filename = time_stamp + '-{}.png'.format(epoch + 1)
        #         save_path = os.path.join(args.output_dir, args.log_name, 'figures/' + filename)
        #         ret = grid(inputs, samples, save_path=save_path, ncols=10)
        #         writer.add_image("val/image", ret, epoch)

        if (epoch + 1) % args.viz_interval == 0:
            # Sample for visualization
            model.eval()
            samples = model.sample_conditioned(inputs)
            rets = []
            for idx in range(min(10, inputs.size(0))):
                ret = visualize_point_clouds(samples[idx], inputs[idx], idx,
                        pert_order=train_loader.dataset.display_axis_order)
                rets.append(ret)
            ret = np.concatenate(rets, axis=1)
            scipy.misc.imsave(
                    os.path.join(log_path, 'tr_vis_conditioned_epoch%d.png'%epoch),
                    ret.transpose((1,2,0)))
            if writer is not None:
                writer.add_image('tr_vis/conditioned', ret, epoch)

            # Sampling reconstruction (only for non AE case)
            B, N = inputs.size(0), inputs.size(1)
            ttl = min(10, B)
            samples = model.sample_unconditioned(N)
            samples = samples[:ttl, ...]
            rets = []
            for idx in range(ttl):
                ret = visualize_point_clouds(samples[idx], inputs[idx], idx,
                        pert_order=train_loader.dataset.display_axis_order)
                rets.append(ret)
            ret = np.concatenate(rets, axis=1)
            scipy.misc.imsave(
                    os.path.join(log_path,
                        'tr_vis_conditioned_epoch%d.png'%epoch),
                    ret.transpose((1,2,0)))
            if writer is not None:
                writer.add_image('tr_vis/sampled', ret, epoch)


        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            filename = time_stamp + '-{}.m'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, args.log_name, 'checkpoints/' + filename)
            model.save(optimizer, save_path)

    # we're already in eval mode, but let's be explicit
    model.eval()
    with torch.no_grad():
        # summarize test batch at end of training
        n = 10  # number of datasets to summarize
        inputs = Variable(test_batch[0].cuda())
        print("Summarizing...")
        summaries = model.summarize_batch(inputs[:n], output_size=6)
        print("Summary complete!")

        # # plot summarized datasets
        # samples = model.sample_conditioned(inputs)
        # filename = time_stamp + '-summary.png'
        # save_path = os.path.join(args.output_dir, 'figures/' + filename)
        # grid(inputs, samples, summaries=summaries, save_path=save_path, ncols=n)


def main():
    tr_max_sample_points = 2048
    te_max_sample_points = tr_max_sample_points
    normalize_per_shape = False
    normalize_per_dataset = True
    normalize_std_per_axis = False
    shapenet_uniform_split = False
    print("Dataset")
    train_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='train',
            tr_sample_size=tr_max_sample_points,
            te_sample_size=te_max_sample_points,
            root_dir=args.data_dir,
            normalize_per_shape=normalize_per_shape,
            normalize_per_dataset=normalize_per_dataset,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=True,
            uniform_split=shapenet_uniform_split)
    test_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='val',
            tr_sample_size=tr_max_sample_points,
            te_sample_size=te_max_sample_points,
            root_dir=args.data_dir,
            normalize_per_shape=normalize_per_shape,
            normalize_per_dataset=normalize_per_dataset,
            normalize_std_per_axis=normalize_std_per_axis,
            all_points_mean=(train_dataset.all_points_mean if normalize_per_dataset else None),
            all_points_std=(train_dataset.all_points_std if normalize_per_dataset else None),
            uniform_split=shapenet_uniform_split
    )

    # train_dataset = SpatialMNISTDataset(data_dir=args.data_dir, split='train')
    # test_dataset = SpatialMNISTDataset(data_dir=args.data_dir, split='test')

    datasets = (train_dataset, test_dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)

    # hardcoded sample_size and n_features when making Spatial MNIST dataset
    sample_size = tr_max_sample_points
    n_features = 3
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.relu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run(model, optimizer, loaders, datasets)


if __name__ == '__main__':
    main()
