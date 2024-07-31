class hparams:

    train_or_test = 'train'
    output_dir = 'logs/Knee_semi_supervised_unetr_0'
    aug = True
    ckpt = None
    latest_checkpoint_file = 'checkpoint_latest.pt'#'checkpoint_latest.pt'
    best_checkpoint_file = 'best_metric_model.pt'
    batch_size = 4
    total_epochs = 300
    iters_perepoch = 200

    epochs_interval = 2


    in_class = 1
    out_class = 5
    crop_or_pad_size = 160,384,384 #128,256,256# # if 2D: 256,256,1      #D W H

    patch_size = 96,96,96 #64,192,192 # 96,192,192 # if 2D: 128,128,1              #D W H
    queue_length = 10
    samples_per_volume = 1

    init_lr = 0.1
    # scheduer_step_size = 15
    # scheduer_gamma = 0.8

    """半监督分割"""
    consistency = 0.1
    consistency_rampup = 300.0
    consistency_type = 'mse'
    ema_decay = 0.99
    labeled_nums = 4
    labeled_bs = 2

    # for test
    patch_overlap = 0,0,0 # if 2D: 4,4,0

    fold_arch = '*.nii.gz'

    save_arch = '.nii.gz'

    unlabeled_image_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageUn'
    unlabeled_label_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelUn'
    labeled_image_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageLa'
    labeled_label_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelLa'
    test_image_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageTs'
    test_label_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelTs'
    val_image_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/imageVa'
    val_label_dir = r'/home/dluser/dataset/ZDY_Dataset/dataset/OAI-ZIB/labelVa'


    output_dir_test = 'results/Knee_semi_supervised_unetr_0'