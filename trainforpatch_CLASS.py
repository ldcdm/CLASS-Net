import os
import random
import time
from itertools import cycle

import monai
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference

from monai.metrics import DiceMetric
from monai.transforms import Compose, EnsureType, AsDiscrete, Activations, AsDiscreted, SaveImaged, EnsureTyped, \
    Invertd, KeepLargestConnectedComponent, SaveImage
from torch.optim import lr_scheduler

from utils import ramps

from utils.dataloader_noresize import MedData_train, MedData_test, Data_fetcher
from utils.makeonehot import make_one_hot

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import argparse
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torchio

from tqdm import tqdm
from hparam_CLASS_pdc_CNN_2_507la8 import hparams as hp
# from hparam_test import hparams as hp
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir_test = hp.output_dir_test

unlabeled_image_dir = hp.unlabeled_image_dir
unlabeled_label_dir = hp.unlabeled_label_dir
labeled_image_dir = hp.labeled_image_dir
labeled_label_dir = hp.labeled_label_dir
test_image_dir = hp.test_image_dir
test_label_dir = hp.test_label_dir
val_image_dir = hp.val_image_dir
val_label_dir = hp.val_label_dir


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--best-checkpoint-file', type=str, default=hp.best_checkpoint_file,
                        help='Store the best checkpoint')
    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs_interval', type=int, default=hp.epochs_interval,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--labeled_bs', type=int, default=hp.labeled_bs, help='labeled-batch-size')
    training.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')
    return parser


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return hp.consistency * ramps.sigmoid_rampup(epoch, hp.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def create_model(ema=False):
    # Network definition

    from models.three_d.vnet3d_pdc_CNN2 import VNet
    model = VNet(n_channels=1, n_classes=hp.out_class, normalization='batchnorm',has_dropout=True,dc=True).to(device)

    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, torch.nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, torch.nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def worker_init_fn(worker_id):
    random.seed(1337 + worker_id)

def seed_torch(args):
    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    monai.utils.set_determinism(seed=args.seed, additional_settings=None)

def train():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    seed_torch(args)

    os.makedirs(args.output_dir, exist_ok=True)

    model1 = create_model()
    # ema_model = create_model(ema=True)
    # model2 = UNet3D(in_channels=hp.out_class,out_channels=hp.in_class,init_features=16).to(device)

    model1 = kaiming_normal_init_weight(model1)
    # model2 = kaiming_normal_init_weight(model2)

    model1 = torch.nn.DataParallel(model1, device_ids=devicess)
    # ema_model = torch.nn.DataParallel(ema_model, device_ids=devicess)

    # model2 = torch.nn.DataParallel(model2, device_ids=devicess)

    # para = list(model1.parameters()) + list(model2.parameters())
    para = model1.parameters()
    optimizer1 = torch.optim.SGD(para, lr=args.init_lr, momentum=0.9, weight_decay=1e-4)

    lr_scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1,
                                               T_max=args.epochs,
                                               eta_min=1e-4)


    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)
        model1.load_state_dict(ckpt["model1"])
        optimizer1.load_state_dict(ckpt["optim1"])
        lr_scheduler1.load_state_dict(ckpt["scheduler1"])
        # model2.load_state_dict(ckpt["model2"])
        # ema_model.load_state_dict(ckpt["ema_model"])
        for state in optimizer1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        elapsed_epochs = ckpt["epoch"]
        best_metric = ckpt["best_metric"]
        best_metric_epoch = ckpt["best_metric_epoch"]
    else:
        elapsed_epochs = 0
        best_metric = -1
        best_metric_epoch = -1

    model1.cuda()


    from utils.loss import DC_and_CE_loss,ContrastiveLoss,weightbceloss,entropy_loss
    criterion = DC_and_CE_loss(soft_dice_kwargs={},ce_kwargs={}).cuda()
    cont_criterion = ContrastiveLoss(batch_size=4,temperature=0.1)
    cons_criterion = weightbceloss().cuda()
    entro = entropy_loss
    mt_criterion = torch.nn.MSELoss(reduction='sum').cuda()

    Train_dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_channel = DiceMetric(include_background=True, reduction="mean_batch")

    writer = SummaryWriter(args.output_dir)

    batch_size = hp.batch_size
    labeled_bs = hp.labeled_bs

    un_util = MedData_train(unlabeled_image_dir, unlabeled_label_dir, True, batch_size=batch_size - labeled_bs,contrast=True)
    un_loader = un_util.train_loader

    la_util = MedData_train(labeled_image_dir, labeled_label_dir, False, batch_size=labeled_bs,contrast=True)
    la_loader = la_util.train_loader

    val_util = MedData_test(val_image_dir, val_label_dir)
    val_loader = val_util.test_loader

    fetcher = Data_fetcher(la_loader,un_loader,0,2,hp.iters_perepoch)

    model1.train()
    # ema_model.train()
    # model2.train()

    max_epochs = args.epochs

    ites_per_epoch = fetcher.length

    epochs = max_epochs - elapsed_epochs
    iteration = elapsed_epochs * ites_per_epoch
    num_iters = iteration

    print("max epochs", max_epochs)
    print("now iters", num_iters, "now epoch", elapsed_epochs+1)


    epoch_loss_values = []
    metric_values = []
    post_ori = Compose([
        EnsureType(),
    ])
    post_output = Compose([
        EnsureType(),
        AsDiscrete(argmax=True, to_onehot=hp.out_class),
        KeepLargestConnectedComponent(is_onehot=True,applied_labels=[1,2,3])
    ])
    post_label = Compose([
        EnsureType(),
        AsDiscrete(to_onehot=hp.out_class),
        # KeepLargestConnectedComponent(is_onehot=True,applied_labels=[1,2,3])
    ])


    for epoch in range(1, epochs+1):
        print("epoch:" + str(epoch))
        time1 = time.time()
        epoch += elapsed_epochs

        epoch_loss = 0
        step = 0

        pbar = tqdm(range(ites_per_epoch), dynamic_ncols=True)
        l1 = 0
        l2 = 0
        l3 = 0
        # save_ = 1
        for i in pbar:
            pbar.set_description(desc='Train_Epoch: {}/{}'.format(epoch, epochs+elapsed_epochs))
        # for i in range(ites_per_epoch):
        #
        #     print(f"Batch: {i}/{ites_per_epoch} epoch {epoch}")

            optimizer1.zero_grad()
            model1.train()
            # model2.train()
            batch1,batch2 = fetcher.fetch()

            input1, label1 = (
                batch1["image"].to(device),
                batch1["label"].to(device),
            )
            input2, label2 = (
                batch2["image"].to(device),
                batch2["label"].to(device),
            )
            input1_aug, label1_aug = (
                batch1["image_aug"].to(device),
                batch1["label_aug"].to(device),
            )
            input2_aug, label2_aug = (
                batch2["image_aug"].to(device),
                batch2["label_aug"].to(device),
            )

            input = torch.cat([input1,input2,input1_aug,input2_aug],dim=0)
            label = torch.cat([label1,label2,label1_aug,label2_aug], dim=0)

            x = input

            x = x.type(torch.FloatTensor).cuda()
            y_all = label
            y = y_all.type(torch.FloatTensor).cuda()

            outputs = model1(x,False,True)
            feat1,feat2,feat3,outputs1 = outputs

            supervised_loss = criterion(outputs1[:labeled_bs], y[:labeled_bs])
            supervised_loss_aux = criterion(feat3[:labeled_bs], y[:labeled_bs])

            cont1_loss = cont_criterion(feat1[:batch_size],feat1[batch_size:])
            cont_loss = cont1_loss

            consistency_weight = get_current_consistency_weight(epoch)

            l1+=(cont_loss.detach().cpu()/200)

            # loss = supervised_loss + consistency_weight*(supervised_loss_aux + cont_loss)
            loss = supervised_loss + consistency_weight * (cont_loss)

            y_pred = make_one_hot(torch.argmax(torch.softmax(outputs1[:labeled_bs].detach(), dim=1), dim=1, keepdim=True),hp.out_class)
            y_ = make_one_hot(y[:labeled_bs].long(),hp.out_class)

            # print(y_pred.shape,y_.shape)
            Train_dice_metric(y_pred=y_pred, y=y_)


            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            epoch_loss += loss.item()

            num_iters += 1
            step += 1

            ## log
            writer.add_scalar('Training/Loss', loss.item(), num_iters)

        lr_scheduler1.step()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        Train_dice = Train_dice_metric.aggregate().item()
        Train_dice_metric.reset()
        # print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        print(f"epoch {epoch} average loss: {epoch_loss:.4f} average dice: {Train_dice:.4f} ",l1,l2,l3)
        writer.add_scalar('Training/Train_dice', Train_dice, epoch)

        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model1": model1.state_dict(),
                "optim1": optimizer1.state_dict(),
                "scheduler1": lr_scheduler1.state_dict(),
                # "model2": model2.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "best_metric_epoch": best_metric_epoch,
            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        # Save checkpoint
        # if True:
        if epoch == max_epochs or epoch % hp.epochs_interval == 0:
            # if epoch % 1 == 0:
            model1.eval()
            save_flag = 1
            with torch.no_grad():

                for val_data in val_loader:

                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    input_args = {'turnoff_drop': True, 'is_train': False}
                    val_outputs = model1(val_inputs, turnoff_drop=True, is_train=False)

                    val_inputs = [post_ori(i) for i in decollate_batch(val_inputs)]
                    val_outputs = [post_output(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric_channel(y_pred=val_outputs, y=val_labels)
                    if (epoch % 25 == 0) & (save_flag == 1):
                        save_flag = 0
                        # print(val_inputs[0].shape)
                        saver1 = SaveImage(output_dir=output_dir_test, output_ext=".nii.gz", output_postfix="seg",
                                           print_log=False)
                        saver2 = SaveImage(output_dir=output_dir_test, output_ext=".nii.gz", output_postfix="ori",
                                           print_log=False)
                        meta_data = decollate_batch(val_data["image_meta_dict"])
                        for val_ori, val_output, data in zip(val_inputs, val_outputs, meta_data):
                            val_output = torch.argmax(val_output, dim=0, keepdim=True)
                            saver2(val_ori, data)
                            saver1(val_output, data)

                    # print('use time:', time2 - time1, dice_metric.aggregate().item(), dice_metric_channel.aggregate())
                metric_channel = dice_metric_channel.aggregate()
                metric = torch.mean(metric_channel)
                print(metric_channel)
                dice_metric_channel.reset()
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch
                    torch.save(
                        {
                            "model1": model1.state_dict(),
                            "optim1": optimizer1.state_dict(),
                            "scheduler1": lr_scheduler1.state_dict(),
                            # "model2": model2.state_dict(),
                            # "optim2": optimizer2.state_dict(),
                            # "scheduler2": lr_scheduler2.state_dict(),
                            "epoch": epoch,
                            # "scaler": scaler.state_dict(),
                            "best_metric": best_metric,
                            "best_metric_epoch": best_metric_epoch,
                        },
                        os.path.join(args.output_dir, args.best_checkpoint_file),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
            writer.add_scalar('Training/dice', metric, epoch)

        time2 = time.time()
        print('time per epoch', time2 - time1)

    writer.close()


if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
