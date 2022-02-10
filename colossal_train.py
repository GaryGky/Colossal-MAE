import argparse
import datetime
import os
import time
from pathlib import Path

import colossalai
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader

import models_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # parser.add_argument('--batch_size', default=64, type=int,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    #
    # parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
    #                     help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/imagenet/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def forward_loss(engine, model, imgs, pred, mask):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = model.patchify(imgs)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6) ** .5

    loss = engine.criterion(pred, target)
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def main(args):
    # ./config.py refers to the config file we just created in step 1
    colossalai.launch_from_torch(config='./colossal-ai/config.py')
    logger = get_dist_logger()
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    # if True:  # args.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #
    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    train_dataloader = get_dataloader(dataset=dataset_train,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    # define the model
    model = models_mae.__dict__[gpc.config.MODEL](norm_pix_loss=gpc.config.NORM_PIX_LOSS)

    # build criterion
    criterion = torch.nn.MSELoss()

    # DDP 的操作

    # model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    #
    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    #
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256
    #
    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)
    #
    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)
    #
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LR, betas=(0.9, 0.95))
    print(optimizer)
    # loss_scaler = NativeScaler()
    #
    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # build engine
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                         optimizer,
                                                                         criterion,
                                                                         train_dataloader,
                                                                         )

    print(f"Start training for {gpc.config.NUM_EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(gpc.config.NUM_EPOCHS):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        # train_stats = train_one_epoch(
        #     model, data_loader_train,
        #     optimizer, device, epoch, loss_scaler,
        #     log_writer=log_writer,
        #     args=args
        # )
        engine.train()
        for img, label in train_dataloader:
            img = img.cuda()

            # set gradiant to zero
            engine.zero_grad()

            # run forward
            outputs = engine(img)

            pred, mask = outputs
            # forward loss
            loss = forward_loss(engine, model, img, pred, mask)
            # backward and update parameters
            engine.backward(loss)
            engine.step()

            logger.info(
                f"Epoch {epoch} - train loss: {loss.item():.5},  lr: {optimizer.param_groups[0]['lr']:.5g}", ranks=[0])

        # save model
        # if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              'epoch': epoch, }

        # log print
        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
