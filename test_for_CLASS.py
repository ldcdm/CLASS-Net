import os
import random
import time
from itertools import cycle

import monai
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.inferers import sliding_window_inference

from monai.metrics import DiceMetric,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric,MSEMetric,MAEMetric
from monai.transforms import Compose, EnsureType, AsDiscrete, Activations, EnsureTyped, Invertd, AsDiscreted, \
    KeepLargestConnectedComponentd, SaveImage, KeepLargestConnectedComponent, ToDeviced

from utils import ramps

from utils.dataloader_monai import MedData_train, MedData_test
from utils.dataloader_noresize import MedData_val

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import argparse
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torchio
from medpy import metric
from tqdm import tqdm
# from hparam import hparams as hp
from hparam_CLASS_pdc_CNN_2_507la8_test import hparams as hp
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
# val_image_dir = test_image_dir
# val_label_dir = test_label_dir

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
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
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
    #
    # from models.three_d.vnet3d_deep import VNet
    # model = VNet(n_channels=1, n_classes=5, normalization='batchnorm', has_dropout=True,dv=True).to(device)

    # from models.three_d.vnet3d_dc import VNet
    # model = VNet(n_channels=1, n_classes=hp.out_class, normalization='batchnorm',has_dropout=True,dc=True).to(device)

    # from models.three_d.new_vnet3d import VNet
    # model = VNet(n_channels=1, n_classes=5, normalization='batchnorm',has_dropout=True).to(device)

    from models.three_d.vnet3d_pdc_CNN2 import VNet
    model = VNet(n_channels=1, n_classes=hp.out_class, normalization='batchnorm',has_dropout=True).to(device)


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
def test():
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    seed_torch(args)
    os.makedirs(args.output_dir, exist_ok=True)

    model1 = create_model()


    model1 = torch.nn.DataParallel(model1, device_ids=devicess)


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.best_checkpoint_file))

    ckpt = torch.load(os.path.join(args.output_dir, args.best_checkpoint_file),
                      map_location=lambda storage, loc: storage)
    model1.load_state_dict(ckpt["model1"])
    best_metric_epoch = ckpt["best_metric_epoch"]
    print('best_metric_epoch: ',best_metric_epoch)

    model1.cuda()

    model1.eval()

    dice_metric_channel1 = DiceMetric(include_background=False, reduction="mean_batch")

    hausdorff_metric_channel = HausdorffDistanceMetric(include_background=False,reduction='mean_batch',percentile=95)

    # asd_metric_channel = SurfaceDistanceMetric(include_background=False,reduction='mean_batch',symmetric=True)
    val_util = MedData_val(test_image_dir, test_label_dir, )
    val_loader = val_util.val_loader

    # val_util = MedData_test(val_image_dir, val_label_dir, )
    # val_loader = val_util.test_loader
    val_org_transforms = val_util.transforms


    """
    如果验证的时候有变换操作
    """
    post_transforms = Compose(
        [
            # Invertd(
            #     keys="pred",
            #     transform=val_org_transforms,
            #     orig_keys="image",
            #     meta_keys="pred_meta_dict",
            #     orig_meta_keys="image_meta_dict",
            #     meta_key_postfix="meta_dict",
            #     nearest_interp=False,
            #     to_tensor=True,
            #     device="cpu",
            # ),
            # Invertd(
            #     keys="label",
            #     transform=val_org_transforms,
            #     orig_keys="label",
            #     meta_keys="label_meta_dict",
            #     orig_meta_keys="label_meta_dict",
            #     meta_key_postfix="meta_dict",
            #     nearest_interp=False,
            #     to_tensor=True,
            #     device="cpu",
            # ),
            # Invertd(
            #     keys="image",
            #     transform=val_org_transforms,
            #     orig_keys="image",
            #     meta_keys="image_meta_dict",
            #     orig_meta_keys="image_meta_dict",
            #     meta_key_postfix="meta_dict",
            #     nearest_interp=False,
            #     to_tensor=True,
            #     device="cpu",
            # ),
            ToDeviced(keys="pred",device="cpu"),
            AsDiscreted(keys="pred", argmax=True, to_onehot=hp.out_class),
            KeepLargestConnectedComponentd(keys="pred",is_onehot=True, applied_labels=[1, 2, 3]),
            AsDiscreted(keys="label", to_onehot=hp.out_class),
        ]
    )
    saver_ori = SaveImage(output_dir=output_dir_test, output_ext=".nii.gz", output_postfix="ori",print_log=True)
    saver_gt = SaveImage(output_dir=output_dir_test, output_ext=".nii.gz", output_postfix="gt",print_log=True)
    saver_seg = SaveImage(output_dir=output_dir_test, output_ext=".nii.gz", output_postfix="seg",print_log=True)

    with torch.no_grad():
        i = 1

        for val_data in val_loader:
            time1 = time.time()
            val_inputs = val_data["image"].to(device)
            # val_labels = val_data["label"].to(device)

            val_data["pred"] = model1(val_inputs,turnoff_drop=True,is_train=False)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_inputs,val_outputs, val_labels = from_engine(["image", "pred", "label"])(val_data)

            hausdorff_metric_channel(y_pred=val_outputs,y=val_labels)
            dice_metric_channel1(y_pred=val_outputs,y=val_labels)

            time2 = time.time()

            if(i==1):
                meta_data = decollate_batch(val_data[0]["image_meta_dict"])

                metric_channel1 = dice_metric_channel1.aggregate()
                metric_hd1 = hausdorff_metric_channel.aggregate()
                # print(metric_org1,metric_channel1)
                saver_ori(val_inputs[0],{"filename_or_obj":val_data[0]["image_meta_dict"]["filename_or_obj"],"affine":val_data[0]["image_meta_dict"]["affine"]})
                saver_seg(torch.argmax(val_outputs[0],dim=0,keepdim=True),{"filename_or_obj":val_data[0]["image_meta_dict"]["filename_or_obj"],"affine":val_data[0]["image_meta_dict"]["affine"]})
                saver_gt(torch.argmax(val_labels[0],dim=0,keepdim=True),{"filename_or_obj":val_data[0]["image_meta_dict"]["filename_or_obj"],"affine":val_data[0]["image_meta_dict"]["affine"]})
            # metric_channel1 = dice_metric_channel1.aggregate()
            print('data_name:{},dice:{},hd:{}'.format(val_data[0]["image_meta_dict"]["filename_or_obj"],metric_channel1,metric_hd1))

            # i=i+1
            print('use time:{:.4f},'.format(time2 - time1))


        hd_channel = hausdorff_metric_channel.aggregate()
        hd_org = torch.mean(hd_channel)
        # asd_channel = asd_metric_channel.aggregate()
        # asd_org = torch.mean(asd_channel)

        metric_channel1 = dice_metric_channel1.aggregate()
        metric_org1 = torch.mean(metric_channel1)

        hausdorff_metric_channel.reset()
        # asd_metric_channel.reset()
        dice_metric_channel1.reset()
    # print("Metric on original image spacing: ", metric_org)
    # print("channel wise Metric on original image spacing: ", metric_channel)
    print("Metric on original image spacing (no BG) hd: ", hd_org)
    print("channel wise Metric on original image spacing(no BG) hd: ", hd_channel)
    # print("Metric on original image spacing (no BG) asd: ", asd_org)
    # print("channel wise Metric on original image spacing(no BG) asd: ", asd_channel)
    print("Metric on original image spacing (no BG): ", metric_org1)
    print("channel wise Metric on original image spacing(no BG): ", metric_channel1)
    # values /= 50
    # print(values)
    # print(np.mean(values, axis=0))
if __name__ == '__main__':
    if hp.train_or_test == 'test':
        test()
