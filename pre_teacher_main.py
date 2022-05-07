# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import random
from io import open

import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW

from vilbert.vilbert import BertConfig
from vilbert.model import VilForCLS, VilWithHash
from vilbert.datasets.hash_dataset import TeaDataset
from vilbert.calc_hr import calc_map

from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# 分类预测精度
def precision(pred_label, label):
    pred_label[pred_label >= 0.5] = 1
    pred_label[pred_label < 0.5] = 0
    precision = pred_label.mul(label)
    precision = torch.sum(precision, 1)
    precision[precision > 1] = 1
    precision = torch.sum(precision)
    # precision = precision / (label.shape[0])
    return precision


# set lr and weight_decay
def set_params(key, value, lr, optimizer_grouped_parameters):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if value.requires_grad:
        if any(nd in key for nd in no_decay):
            optimizer_grouped_parameters += [
                {"params": [value], "lr": lr, "weight_decay": 0.0}
            ]
        if not any(nd in key for nd in no_decay):
            optimizer_grouped_parameters += [
                {"params": [value], "lr": lr, "weight_decay": 0.01}
            ]

    return optimizer_grouped_parameters


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--from_pretrained",
        # default="bert-base-uncased",
        default=".save/net_epoch_60.bin",
        type=str,
        help="pre-trained vilbert model",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, roberta-base",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/bert_base_6layer_6conect.json",
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=20,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--max_region_num",
        default=37,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=250,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr_scheduler_epoch",
        default=[70, 140, 210],
        help="lr change epoch"
    )
    parser.add_argument(
        "--lr_decay_value",
        default=0.2,
        type=float,
        help="lr decay value"
    )
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="The initial learning rate.",
    )
    parser.add_argument(
        "--finetune_lr", default=0.00002, type=float, help="The initial learning rate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--num_label", type=int, default=38, help="code length"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=10,
        help='hyper-param for pairloss'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.01,
        help='hyper-param for quantification loss'
    )

    parser.add_argument('--imgfeature_root',
                        default="./data/mir/feature/",
                        help='path to features extracted by object detection')
    parser.add_argument('--train_list',
                        default='./data/mir_train.txt',
                        help='file of training set')
    parser.add_argument('--database_list',
                        default='./data/mir_database.txt',
                        help='file of database set')
    parser.add_argument('--test_list',
                        default='./data/mir_test.txt',
                        help='file of testing set')
    parser.add_argument('--cap_train',
                        default='./data/mir_train_cap.txt',
                        help='caption file of training set')
    parser.add_argument('--cap_test',
                        default='./data/mir_test_cap.txt',
                        help='caption file of testing set')
    parser.add_argument('--cap_database',
                        default='./data/mir_database_cap.txt',
                        help='caption file of database set')
    parser.add_argument("--output_dir",
                        default="./save/teacher/mir/pretrain/",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    default_gpu = True

    config = BertConfig.from_json_file(args.config_file)
    config.num_label = args.num_label

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_output_dir = args.output_dir
    if not os.path.exists(total_output_dir):
        os.makedirs(total_output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )


    if args.from_pretrained:
        model = VilForCLS.from_pretrained(
            args.from_pretrained, config=config, default_gpu=default_gpu
        )
    else:
        model = VilForCLS(config)

    bert_lr = 0.000002
    optimizer_grouped_parameters = []
    for key, value in dict(model.bert.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, bert_lr, optimizer_grouped_parameters)
    for key, value in dict(model.classification.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)
    for key, value in dict(model.v_classification.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)

    # lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.4)
    lr_reduce_list = np.array([20, 40, 80])
    def lr_lambda_fun(epoch):
        return pow(0.2, np.sum(lr_reduce_list <= epoch))

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    model.cuda()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    padding_index = 0

    bce_loss = torch.nn.BCELoss()

    train_dataset = TeaDataset(
        args.imgfeature_root,
        args.cap_train,
        args.train_list,
        args.max_seq_length,
        args.max_region_num,
        tokenizer,
        padding_index
    )

    test_dataset = TeaDataset(
        args.imgfeature_root,
        args.cap_test,
        args.test_list,
        args.max_seq_length,
        args.max_region_num,
        tokenizer,
        padding_index
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True, num_workers=8)

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d num_train_epochs = %d" % (args.train_batch_size, args.num_train_epochs))
    logger.info("  lr = %f  " % (args.learning_rate))

    for epochId in range(0, args.num_train_epochs):
        model.train()

        img_loss = 0.0
        img_prec = 0.0
        text_prec = 0.0
        text_loss = 0.0
        all_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            image_feat, image_loc, image_mask, input_ids, input_mask, segment_ids, image_id, label, _ = (
                batch
            )

            image_feat = image_feat.to(device)
            image_loc = image_loc.to(device)
            image_mask = image_mask.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label = label.to(device)

            _, _, output_t, output_v = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
            )

            loss_t = bce_loss(output_t, label)
            loss_v = bce_loss(output_v, label)
            loss = loss_t + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            text_loss = text_loss + loss_t.item()
            img_loss = img_loss + loss_v.item()
            all_loss = all_loss + loss.item()

            img_prec = img_prec + precision(output_v, label)
            text_prec = text_prec + precision(output_t, label)

        # lr_scheduler.step()

        text_loss = text_loss / len(train_dataloader)
        img_loss = img_loss / len(train_dataloader)
        all_loss = all_loss / len(train_dataloader)

        img_prec = img_prec / len(train_dataset)
        text_prec = text_prec / len(train_dataset)

        logger.info('epoch %d total_loss: %.4f text_loss: %.4f img_loss: %.4f text_prec: %.4f img_prec: %.4f ' %
                    (epochId, all_loss, text_loss, img_loss, text_prec, img_prec))

        # save and test
        if (epochId + 1) % 5 == 0:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            torch.save(model_to_save.state_dict(), '%s/net_epoch_%d.bin' % (total_output_dir, epochId + 1))

            logger.info("******  Do the evaluation  *********")
            # # Do the evaluation

            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train_batch_size,
                                                          shuffle=False, num_workers=8)

            torch.set_grad_enabled(False)
            model.eval()

            img_prec = 0.0
            text_prec = 0.0

            for step, batch in enumerate(test_dataloader):
                image_feat, image_loc, image_mask, input_ids, input_mask, segment_ids, image_id, label, _ = (
                    batch
                )

                image_feat = image_feat.to(device)
                image_loc = image_loc.to(device)
                image_mask = image_mask.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label = label.to(device)

                _, _, output_t, output_v = model(
                    input_ids,
                    image_feat,
                    image_loc,
                    segment_ids,
                    input_mask,
                    image_mask,
                )

                img_prec = img_prec + precision(output_v, label)
                text_prec = text_prec + precision(output_t, label)

            img_prec = img_prec / len(test_dataset)
            text_prec = text_prec / len(test_dataset)

            logger.info('*** For Test *** model: %s text_prec: %.4f img_prec: %.4f ' %
                        (total_output_dir, text_prec, img_prec))

            torch.set_grad_enabled(True)


if __name__ == "__main__":

    main()
