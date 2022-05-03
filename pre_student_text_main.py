import os
import argparse
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import random
from torch.optim.lr_scheduler import (
    LambdaLR,
)


from vilbert.net import CKDH, ImgBackbone, TextBackbone
import logging


from vilbert.vilbert import BertConfig
from vilbert.datasets.hash_dataset import (
    StuDataSet,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    #filename='mir_16.log'
)
logger = logging.getLogger(__name__)


# Classification prediction accuracy
def precision(pred_label, label):
    pred_label[pred_label >= 0.5] = 1
    pred_label[pred_label < 0.5] = 0
    precision = pred_label.mul(label)
    precision = torch.sum(precision, 1)
    precision[precision > 1] = 1
    precision = torch.sum(precision)
    # precision = precision / (label.shape[0])
    return precision

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='number of data loading workers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='input batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='number of epochs to train for')
    parser.add_argument("--lr_scheduler_epoch",
                        default=[45, 80, 150],
                        help="lr change epoch")
    parser.add_argument("--lr_decay_value",
                        default=0.2,
                        type=float,
                        help="lr decay value")
    parser.add_argument('--lr',
                        type=float,
                        default=5,
                        help='learning rate')
    parser.add_argument('--lrMul', type=float, default=100.)
    parser.add_argument('--seed', type=int, default=42,
                        help='manual seed')
    parser.add_argument('--image_dir',
                        default="./data/mir/image/",
                        help='path to image')
    parser.add_argument('--train_list',
                        default='./data/mir_train.txt',
                        help='file of training set')
    parser.add_argument('--test_list',
                        default='./data/mir_test.txt',
                        help='file of testing set')
    parser.add_argument('--tag_train',
                        default='./data/mir_train_tag.txt',
                        help='tag file of training set')
    parser.add_argument('--tag_test',
                        default='./data/mir_test_tag.txt',
                        help='tag file of testing set')
    parser.add_argument('--num_label',
                        type=int,
                        default=38,
                        help='cls numbers')
    parser.add_argument("--output_dir",
                        default="save/pretrain/mir/img/",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--y_dim",
                        default=1386,
                        type=int,
                        help="input tag dim")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_output_dir = args.output_dir
    if not os.path.exists(total_output_dir):
        os.makedirs(total_output_dir)

    bone = TextBackbone(args.y_dim, args.num_label)
    bone.to(device)

    train_dataset = StuDataSet(
        args.image_dir,
        args.train_list,
        args.tag_train,
        transform_pre=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]),
        transform_totensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )

    test_dataset = StuDataSet(
        args.image_dir,
        args.test_list,
        args.tag_test,
        transform_pre=transforms.Compose([
            transforms.Resize((224, 224)),
        ]),
        transform_totensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers)

    optimizer = optim.SGD([{'params': bone.transition.parameters()},
                           {'params': bone.cls.parameters(), 'lr': 0.1}
                           ], lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    lr_reduce_list = np.array(args.lr_scheduler_epoch)
    def v_lr_lambda_fun(epoch):
        return pow(args.lr_decay_value, np.sum(lr_reduce_list <= epoch))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=v_lr_lambda_fun)

    bce_loss = torch.nn.BCELoss()

    logger.info("***** Running training *****")
    logger.info("  output_dir = %s" % (total_output_dir))
    logger.info("  Batch size = %d num_train_epochs = %d" % (args.batch_size, args.epochs))

    for epochId in range(0, args.epochs):
        bone.train()

        all_loss = 0.0

        prec = 0.0

        for step, batch in enumerate(train_dataloader):
            image, tag, label, filename = (
                batch
            )

            image = image.to(device)
            label = label.to(device)
            tag = tag.to(device)

            pre_label = bone(tag)

            err = bce_loss(pre_label, label)

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            all_loss = all_loss + err.item()

            prec = prec + precision(pre_label, label)

        lr_scheduler.step()
        all_loss = all_loss / len(train_dataloader)
        prec = prec / len(train_dataset)

        logger.info('epoch %d total_loss: %.4f prec: %.4f '
                    % (epochId, all_loss, prec))

        # save and test
        if (epochId + 1) % 5 == 0:
            model_to_save = (
                bone.module if hasattr(bone, "module") else bone
            )  # Only save the model it-self
            torch.save(model_to_save.state_dict(), '%s/net_epoch_%d.pth' % (total_output_dir, epochId + 1))

            logger.info("******  Do the evaluation  *********")
            # Do the evaluation

            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=args.workers)

            torch.set_grad_enabled(False)

            bone.eval()

            prec = 0.0

            for step, batch in enumerate(test_dataloader):
                image, tag, label, filename = (
                    batch
                )

                image = image.to(device)
                tag = tag.to(device)
                label = label.to(device)

                label_t = bone(tag)

                prec = prec + precision(label_t, label)

            prec = prec / len(test_dataset)

            logger.info('*** For Test *** epoch: %d prec: %.4f ' % (epochId, prec))

            torch.set_grad_enabled(True)


if __name__ == '__main__':
    main()


