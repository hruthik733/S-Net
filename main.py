import argparse
import torch
import datetime
import os
import sys

import torch.nn.functional as F

import pandas as pd
import imageio.v2 as imageio


from Datasets.ISIC2018 import ISIC2018_dataset
from Datasets.Kvasir import Kvasir_dataset
from Datasets.BUSI import BUSI_dataset
from utils1.transform import dataset_transform
from utils1.utils import AverageMeter, Logger, BceDiceLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils1.metric import iou_and_dice, jc, dc, recall, ACC, ravd
from collections import OrderedDict

from Models.Unet import Unet
from Models.S_Net import S_Net

# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend for scripts
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0.000001, lr_max=0.001, warmup=True):
#    warmup_epoch = 5 if warmup else 0
#    if current_epoch < warmup_epoch:
#        lr = lr_max * current_epoch / warmup_epoch
#    else:
#        lr = lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr


def main(args):
    # 定义所用设备是cpu还是gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.data == 'ISIC2018_png_224':
        # # 根据文件路径生成经过排序的图像文件名列表
        # image_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'image'), key=lambda x: x.split('.')[0].split('_')[-1])
        # label_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'label'), key=lambda x: x.split('_')[1])
        # # 生成完整的文件路径
        # image_path_list = [os.path.join(Path_dataset[args.data], 'image', x) for x in image_name_list]
        # label_path_list = [os.path.join(Path_dataset[args.data], 'label', x) for x in label_name_list]
        # path_list = (image_path_list, label_path_list)
        # # 根据所用数据集自动计算数据集大小，并自动拆分出训练集、验证集和测试集
        # index = [i for i in range(len(os.listdir(Path_dataset[args.data] + '/' + 'image')))]
        # train_index, test_and_val_index = train_test_split(index, test_size=0.3, random_state=45)
        # val_index,  test_index = train_test_split(test_and_val_index, test_size=0.66, random_state=45)
        # # 若需要进行交叉验证则则将epoch循环放在以下循环代码内
        # # kf = KFold(n_splits=args.n_splits, shuffle=False)
        # # for train_index, val_index in kf.split(train_and_val_index):
        # #     train_index = [train_and_val_index[i] for i in train_index]
        # #     val_index = [train_and_val_index[i] for i in val_index]
        # # 判断分割结果中的金标准文件夹是否有与测试数量相同的图片
        # if os.path.exists('./segment_result/gold_standard/ISIC2018/image'):
        #     pass
        # else:
        #     os.makedirs('./segment_result/gold_standard/ISIC2018/image')
        #     os.makedirs('./segment_result/gold_standard/ISIC2018/label')
        # if len(os.listdir('./segment_result/gold_standard/ISIC2018/image')) != len(test_index):
        #     for i in test_index:
        #         image_path = image_path_list[i]
        #         label_path = label_path_list[i]
        #         # # 加载保存.pny格式图像
        #         # image = np.load(image_path)
        #         # label = np.load(label_path)
        #         # np.save('./segment_result/gold_standard/' + image_path.split('/')[-1], image)
        #         # np.save('./segment_result/gold_standard/' + label_path.split('/')[-1], label)
        #         # 加载保存.png格式图像
        #         image = imageio.imread(image_path)
        #         label = imageio.imread(label_path)
        #         imageio.imwrite('./segment_result/gold_standard/ISIC2018/image/' + image_path.split('/')[-1], image)
        #         imageio.imwrite('./segment_result/gold_standard/ISIC2018/label/' + label_path.split('/')[-1], label)
        # print(f"训练集数量：{len(train_index)}，训练集索引：{train_index}")
        # print(f"验证集数量：{len(val_index)}，验证集索引：{val_index}")
        # print(f"测试集数量：{len(test_index)}，测试集索引：{test_index}")

        # # 定义训练集验证集，采用传入数据集索引的方式拆分训练验证测试集，方便进行数据打乱。
        # train_dataset = Dataset[args.data](path_list=path_list, index=train_index,
        #                                    train_type='train', image_size=(224, 224), transform=dataset_transform)
        # val_dataset = Dataset[args.data](path_list=path_list, index=val_index,
        #                                  train_type='val', image_size=(224, 224), transform=dataset_transform)
        # test_dataset = Dataset[args.data](path_list=path_list, index=test_index,
        #                                   train_type='test', image_size=(224, 224), transform=dataset_transform)

        base_path = './data/isic-dataset'
        img_dirs = {
            'train': os.path.join(base_path, 'data_train'),
            'val': os.path.join(base_path, 'data_val'),
            'test': os.path.join(base_path, 'data_test')
        }
        mask_dirs = {
            'train': os.path.join(base_path, 'mask_train'),
            'val': os.path.join(base_path, 'mask_val'),
            'test': os.path.join(base_path, 'mask_test')
        }

        # Get image and label paths
        def get_sorted_paths(img_dir, mask_dir):
            images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
            masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
            return images, masks

        train_imgs, train_masks = get_sorted_paths(img_dirs['train'], mask_dirs['train'])
        val_imgs, val_masks = get_sorted_paths(img_dirs['val'], mask_dirs['val'])
        test_imgs, test_masks = get_sorted_paths(img_dirs['test'], mask_dirs['test'])

        path_list = {
            'train': (train_imgs, train_masks),
            'val': (val_imgs, val_masks),
            'test': (test_imgs, test_masks)
        }

        # Create gold standard directory for test set
        gold_img_dir = './segment_result/gold_standard/ISIC2018/image'
        gold_lbl_dir = './segment_result/gold_standard/ISIC2018/label'
        os.makedirs(gold_img_dir, exist_ok=True)
        os.makedirs(gold_lbl_dir, exist_ok=True)

        if len(os.listdir(gold_img_dir)) != len(test_imgs):
            for img_path, lbl_path in zip(test_imgs, test_masks):
                image = imageio.imread(img_path)
                label = imageio.imread(lbl_path)
                imageio.imwrite(os.path.join(gold_img_dir, os.path.basename(img_path)), image)
                imageio.imwrite(os.path.join(gold_lbl_dir, os.path.basename(lbl_path)), label)

        print(f"训练集数量：{len(train_imgs)}")
        print(f"验证集数量：{len(val_imgs)}")
        print(f"测试集数量：{len(test_imgs)}")

        # Create dataset instances
        train_dataset = Dataset[args.data](path_list=path_list['train'], index=list(range(len(train_imgs))),
                                        train_type='train', image_size=(224, 224), transform=dataset_transform)
        val_dataset = Dataset[args.data](path_list=path_list['val'], index=list(range(len(val_imgs))),
                                        train_type='val', image_size=(224, 224), transform=dataset_transform)
        test_dataset = Dataset[args.data](path_list=path_list['test'], index=list(range(len(test_imgs))),
                                        train_type='test', image_size=(224, 224), transform=dataset_transform)


    elif args.data == 'Kvasir_png_224':
        # 根据文件路径生成经过排序的图像文件名列表
        image_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'images'),
                                 key=lambda x: x.split('.')[0])
        label_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'masks'), key=lambda x: x.split('.')[0])
        # 生成完整的文件路径
        image_path_list = [os.path.join(Path_dataset[args.data], 'images', x) for x in image_name_list]
        label_path_list = [os.path.join(Path_dataset[args.data], 'masks', x) for x in label_name_list]
        path_list = (image_path_list, label_path_list)
        # 根据所用数据集自动计算数据集大小，并自动拆分出训练集、验证集和测试集
        index = [i for i in range(len(image_name_list))]
        train_index, test_and_val_index = train_test_split(index, test_size=0.3, random_state=45)
        val_index, test_index = train_test_split(test_and_val_index, test_size=0.66, random_state=45)
        if os.path.exists('./segment_result/gold_standard/Kvasir/image'):
            pass
        else:
            os.makedirs('./segment_result/gold_standard/Kvasir/image')
            os.makedirs('./segment_result/gold_standard/Kvasir/label')
        if len(os.listdir('./segment_result/gold_standard/Kvasir/image')) != len(test_index):
            for i in test_index:
                image_path = image_path_list[i]
                label_path = label_path_list[i]
                # # 加载保存.pny格式图像
                # image = np.load(image_path)
                # label = np.load(label_path)
                # np.save('./segment_result/gold_standard/' + image_path.split('/')[-1], image)
                # np.save('./segment_result/gold_standard/' + label_path.split('/')[-1], label)
                # 加载保存.png格式图像
                image = imageio.imread(image_path)
                label = imageio.imread(label_path)
                imageio.imwrite('./segment_result/gold_standard/' + 'Kvasir/image/' + image_path.split('/')[-1], image)
                imageio.imwrite('./segment_result/gold_standard/' + 'Kvasir/label/' + label_path.split('/')[-1], label)
        print(f"训练集数量：{len(train_index)}，训练集索引：{train_index}")
        print(f"验证集数量：{len(val_index)}，验证集索引：{val_index}")
        print(f"测试集数量：{len(test_index)}，测试集索引：{test_index}")

        # 定义训练集验证集，采用传入数据集索引的方式拆分训练验证测试集，方便进行数据打乱。
        train_dataset = Dataset[args.data](path_list=path_list, index=train_index,
                                           train_type='train', image_size=(224, 224), transform=dataset_transform)
        val_dataset = Dataset[args.data](path_list=path_list, index=val_index,
                                         train_type='val', image_size=(224, 224), transform=dataset_transform)
        test_dataset = Dataset[args.data](path_list=path_list, index=test_index,
                                          train_type='test', image_size=(224, 224), transform=dataset_transform)

    elif args.data == 'BUSI_png_224':
        # Get sorted lists of images and masks
        image_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'images'),
                            key=lambda x: x.split('.')[0])
        label_name_list = sorted(os.listdir(Path_dataset[args.data] + '/' + 'masks'),
                            key=lambda x: x.split('.')[0])
        
        # Generate full paths
        image_path_list = [os.path.join(Path_dataset[args.data], 'images', x) for x in image_name_list]
        label_path_list = [os.path.join(Path_dataset[args.data], 'masks', x) for x in label_name_list]
        path_list = (image_path_list, label_path_list)
        
        # Split dataset
        index = [i for i in range(len(image_name_list))]
        train_index, test_and_val_index = train_test_split(index, test_size=0.3, random_state=45)
        val_index, test_index = train_test_split(test_and_val_index, test_size=0.66, random_state=45)
        
        # Create directories if they don't exist
        os.makedirs('./segment_result/gold_standard/BUSI_224/image', exist_ok=True)
        os.makedirs('./segment_result/gold_standard/BUSI_224/label', exist_ok=True)
        
        # Process test set
        if len(os.listdir('./segment_result/gold_standard/BUSI_224/image')) != 2 * len(test_index):
            for i in test_index:
                image_path = image_path_list[i]
                label_path = label_path_list[i]
                
                # Read images and masks
                image = imageio.imread(image_path)
                label = imageio.imread(label_path)
                
                # Convert mask to binary (0-255) if needed
                if label.dtype == bool:
                    label = label.astype(np.uint8) * 255
                elif np.max(label) <= 1:  # If values are between 0-1
                    label = (label * 255).astype(np.uint8)
                
                # Save with original filenames
                imageio.imwrite('./segment_result/gold_standard/BUSI_224/image/' + os.path.basename(image_path), image)
                imageio.imwrite('./segment_result/gold_standard/BUSI_224/label/' + os.path.basename(label_path), label)
        
        print(f"训练集数量：{len(train_index)}，训练集索引：{train_index}")
        print(f"验证集数量：{len(val_index)}，验证集索引：{val_index}")
        print(f"测试集数量：{len(test_index)}，测试集索引：{test_index}")
        
        # Create datasets
        train_dataset = Dataset[args.data](path_list=path_list, index=train_index,
                                        train_type='train', image_size=(224, 224), transform=dataset_transform)
        val_dataset = Dataset[args.data](path_list=path_list, index=val_index,
                                    train_type='val', image_size=(224, 224), transform=dataset_transform)
        test_dataset = Dataset[args.data](path_list=path_list, index=test_index,
                                    train_type='test', image_size=(224, 224), transform=dataset_transform)

    # 定义num_workers数量，根据电脑不太不同
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0])
    # debu
    # num_workers = 1

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=3,
                                             shuffle=False,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=1,
                                              shuffle=False,
                                              pin_memory=True)

    # 创建模型
    model = Model[args.id](3, 1)
    # 定义模型并将模型转移到指定的device上
    model.to(device)

    # 模型架构与参数设置记录
    for key, value in vars(args).items():
        print(f"{key:16} {value}")

    # 计算模型可训练参数量并将可训练参数选出
    print("-"*50)
    print(f"Network Architecture of Model {args.id}:")
    print(model)
    print("-"*50)
    num_params = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_params += num_mul
    print(f"Number of trainable parameters {num_params} in Model {args.id}")
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 定义优化器并将可训练参数传入优化器 SGD时要加momentum=args.momentum,
    # optimizer = torch.optim.Adam(
    #     params_to_optimize,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay
    # )
    # AdamW优化器，加入了正则项，可以更好地减小权重参数值，提高模型的泛化能力
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    # 定义学习率变化
    # 余弦退火算法学习率
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    # 根据验证集上的性能指标自动调整学习率
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, min_lr=1e-6, patience=6
    )
    # loss 用于训练网络更新参数，criterion用于验证网络效果选出最好的网络，两个可以一样
    # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice'])  # 列名
    df.to_csv(os.path.join(args.save_path, args.data, args.id, f'train_val_acc-{args.time}.csv'), index=False)  # 路径可以根据需要更改
    train_loss = BceDiceLoss()
    val_loss = BceDiceLoss()
    best_iou = 0
    trigger = 0
    for epoch in range(args.epochs):
        # 余弦学习率的学习率更新
        # adjust_learning_rate(optimizer=optimizer, current_epoch=epoch+1, max_epoch=args.epochs, lr_min=0.000001,
        #                      lr_max=args.lr, warmup=True)
        # 采用余弦退火学习率时每个epoch更新一次学习率
        # lr_scheduler.step()
        # 另一种学习率设置方法
        # lr_ = args.rl * (1.0 - iter_num / max_iterations) ** 0.9
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_
        #
        # iter_num = iter_num + 1

        # 打印学习率
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # 打印训练进度
        print(f"Epoch:{epoch}/{args.epochs}, lr:{optimizer.param_groups[0]['lr']}")
        train_metric = train(model, train_loader, optimizer, criterion=train_loss, device=device)
        print(f"Epoch:{epoch}，train_loss:{train_metric['loss']}")
        val_metric = val(model, val_loader, criterion=val_loss, device=device)
        # 自动根据值指标调整学习率的学习率更新
        lr_scheduler.step(val_metric['iou'])
        print(f"Epoch:{epoch}, val_loss:{val_metric['loss']}，val_iou:{val_metric['iou']}，val_dice:{val_metric['dice']}")
        # 生成保存数据的CSV表格
        data = [epoch, train_metric['loss'], val_metric['loss'], val_metric['iou'], val_metric['dice']]
        data = pd.DataFrame([data])
        data.to_csv(os.path.join(args.save_path, args.data, args.id, f'train_val_acc-{args.time}.csv'), mode='a', header=False, index=False)

        # 根据验证得到的网络分数判断进行最优模型的保存
        if val_metric['iou'] > best_iou:
            trigger = 0
            model_save_path = os.path.join(args.save_path, args.data, args.id)
            if os.path.exists(model_save_path):
                pass
            else:
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'The best model.pth'))
            best_iou = val_metric['iou']
            print(f'=> saved best model——{datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")}')
        else:
            trigger += 1
        if trigger == args.early_stop:
            break
    print(f"Training Done!  Start testing.——{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")
    print("- "*30)

    model_name = os.path.join(model_save_path, 'The best model.pth')
    # 用于单独测试
    # model_name = os.path.join('./save_model/mscasegNext_net', 'The best model.pth')
    model.load_state_dict(torch.load(model_name))
    print(f'The best model has been loaded.——{datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")}')
    test_metric = test(model, args, test_loader, device=device)
    print(f"test_iou:{test_metric['iou']}, test_dice:{test_metric['dice']}.")
    print(f"Test Done!——{datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}")


def train(model, train_loader, optimizer, criterion, device):
    losses = AverageMeter()
    # 将模型置于训练模式
    model.train()
    torch.autograd.set_detect_anomaly(True)
    # 开始循环data_loader训练
    for image, label in tqdm(train_loader, total=len(train_loader)):
        # 将image和label转移到指定的device上
        image, label = image.to(device), label.to(device)
        # 将图像输入模型进行前向传播
        output = model(image)    # 2,1,224,320
        # print(output.shape, label.shape)
        # print('output:', output)
        # print('label:', label)
        loss = criterion(output, label)

        # 优化器梯度清空，反向传播计算梯度，优化器更新梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新参数记录
        losses.update(loss.item(), image.shape[0])

        metric = OrderedDict([
            ('loss', losses.avg)
        ])
    return metric


def val(model, val_loader, criterion, device):
    net_score = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    # 将模型置于评估模式
    model.eval()
    # 开始循环data_loader训练
    for image, label in tqdm(val_loader, total=len(val_loader)):
        # 将image和label转移到指定的device上
        image, label = image.to(device), label.to(device)
        # 输入图像计算输出
        output = model(image)   # 2,1,224,320
        output = output.detach()
        if output.shape != label.shape:
            label = F.interpolate(label, size=output.shape[2:], mode='bilinear', align_corners=True)

        score = criterion(output, label)
        # 针对2分类的一通道网络输出和二通道网络输出
        if output.shape[1] == 1:
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
        elif output.shape[1] == 2:
            output = torch.max(output, 1)[1].unsqueeze(dim=1)
        # iou, dice = iou_and_dice(output, label)
        iou = jc(output, label)
        dice = dc(output, label)
        # 更新参数记录
        net_score.update(val=score.item(), n=image.shape[0])
        ious.update(iou, image.shape[0])
        dices.update(dice, image.shape[0])

    metric = OrderedDict([
        ('loss', net_score.avg),
        ('iou', ious.avg),
        ('dice', dices.avg)
    ])

    return metric


def test(model, args, test_loader, device):
    # losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    jcs = AverageMeter()
    recalls = AverageMeter()
    accs = AverageMeter()
    rvds = AverageMeter()

    # 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df = pd.DataFrame(columns=['num', 'image_name', 'iou', 'dice', 'jc', 'recall', 'ACC', 'RVD'])  # 列名
    df.to_csv(os.path.join(args.save_path, args.data, args.id, f'test_acc-{args.time}.csv'),index=False)
    # 将模型置于评估模式
    model.eval()
    # 开始循环data_loader测试
    for i, (name, image, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        name = name[0]
        # 将image和label转移到指定的device上
        image, label = image.to(device), label.to(device)
        # 输入图像计算输出
        output = model(image)    # 1,1,224,320
        output = output.detach()
        # loss = criterion(output, label)
        # 针对2分类的一通道网络输出和二通道网络输出
        if output.shape[1] == 1:
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
        elif output.shape[1] == 2:
            output = torch.max(output, 1)[1].unsqueeze(dim=1)

            
        label = F.interpolate(label, size=output.shape[2:], mode='bilinear', align_corners=True)
        jc_, dice_ = iou_and_dice(output, label)
        iou = jc(output, label)
        dice = dc(output, label)
        # jc_ = jc(output, label)
        recall_ = recall(output, label)
        acc = ACC(output, label)
        rvd = ravd(output, label)

        data = [i, name, iou, dice, jc_, recall_, acc, rvd]
        data = pd.DataFrame([data])
        data.to_csv(os.path.join(args.save_path, args.data, args.id, f'test_acc-{args.time}.csv'), mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
        # losses.update(loss.item(), image.shape[0])
        ious.update(iou)
        dices.update(dice)
        jcs.update(jc_)
        recalls.update(recall_)
        accs.update(acc)
        rvds.update(rvd)

        # 按照格式保存
        output = output.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        path = os.path.join(args.save_path, args.data, args.id, f'test_result-{args.time}/')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        imageio.imwrite(path + name.split('.')[0] + '.png', (output*255.0).astype('uint8'))
        # np.save('./segment_result/test_result/' + save_name.split('.')[0] + '.npy', output.cpu().numpy())

    metric = OrderedDict([
        # ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('jc', jcs.avg),
        ('recall', recalls.avg),
        ('acc', accs.avg),
        ('rvd', rvds.avg)
    ])

    return metric


if __name__ == '__main__':
    # 定义所用数据集及其路径的字典
    Dataset = {'ISIC2018_png_224': ISIC2018_dataset,
               'Kvasir_png_224': Kvasir_dataset,
               'BUSI_png_224': BUSI_dataset,
               }
    Path_dataset = {'ISIC2018_png_224': './data/ISIC2018_png_224', 'Kvasir_png_224': './data/Kvasir_png_224', 'BUSI_png_224': './data/BUSI_png_224', }
    Model = {'Unet': Unet, 'S_Net': S_Net}
    # 建立参数解析对象的实例
    parser = argparse.ArgumentParser(description='parameter')
    # 添加实例属性
    parser.add_argument('--time', default=f'{str(datetime.datetime.now()).split(".")[0]}',
                        help='now time')
    parser.add_argument('--id', default='S_Net', help='the name of the network')
    parser.add_argument('--data', default='ISIC2018_png_224', help='choose the dataset')
    parser.add_argument('--n_splits', default=5, type=int, help='the times of the cross validation')
    parser.add_argument('--save_path', default="./save_model/", help='the file name of save model')
    parser.add_argument('--epochs', default=120, type=int,
                        help='number of epochs to train (default:10)')
    parser.add_argument('--early_stop', default=150, type=int, help='the number of early stop')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='inital learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')

    # 将添加的属性给args，方便调用
    args = parser.parse_args()
    args.time = f'{str(datetime.datetime.now()).split(".")[0]}'
    save_path = os.path.join(args.save_path, args.data, args.id)
    # path of the training log
    logfile = os.path.join(save_path, f"Log-{args.time}.txt")
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    sys.stdout = Logger(logfile)
    main(args)

