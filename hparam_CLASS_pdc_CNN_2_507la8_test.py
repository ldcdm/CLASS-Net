class hparams:

    la_num = 8
    un_num = 467 - la_num
    train_or_test = 'test'
    output_dir = f'/home/dluser/DXD/CLASS-Net/logs/full_size/Knee_CLASS-Net_507NOResize{la_num}'
    aug = None
    ckpt = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    best_checkpoint_file = 'best_metric_model.pt'
    batch_size = 1

    in_class = 1
    out_class = 4
    # patch_size = 80,144,144 #128,256,256# # if 2D: 256,256,1      #D W H
    patch_size = 160, 288, 288  # 128,256,256# # if 2D: 256,256,1      #D W H

    # sample_size = 80,144,144#80,192,192
    sample_size = 160, 288, 288

    patch_overlap = 0,0,0 # if 2D: 4,4,0
    # for train
    queue_length = 10
    samples_per_volume = 1
    total_epochs = 300
    epochs_per_checkpoint = 15


    init_lr = 0.1
    """半监督分割"""
    consistency = 0.1
    consistency_rampup = 300.0
    consistency_type = 'mse'
    ema_decay = 0.99
    labeled_bs = 2


    fold_arch = '*.nii.gz'
    save_arch = '.nii.gz'

    unlabeled_image_dir = f'/home/dluser/dataset/DXD/OAI_507/imageUn{un_num}'
    unlabeled_label_dir = f'/home/dluser/dataset/DXD/OAI_507/labelUn{un_num}'
    labeled_image_dir = f'/home/dluser/dataset/DXD/OAI_507/imageLa{la_num}'
    labeled_label_dir = f'/home/dluser/dataset/DXD/OAI_507/labelLa{la_num}'
    test_image_dir = r'/home/dluser/dataset/DXD/OAI_507/imageTest'
    test_label_dir = r'/home/dluser/dataset/DXD/OAI_507/labelTest'
    val_image_dir = r'/home/dluser/dataset/DXD/OAI_507/imageVal'
    val_label_dir = r'/home/dluser/dataset/DXD/OAI_507/labelVal'

    # output_dir_test = f'results/full_size/Knee_semi_supervised_pdcmt_NOResize{la_num}'
    output_dir_test = f'/home/dluser/DXD/CLASS-Net/results/full_size/Knee_CLASS-Net_507NOResize{la_num}'