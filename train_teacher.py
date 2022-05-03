# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import random
from io import open
from time import *

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


# set lr&weight_decay
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
        default="./save/net_epoch.bin",
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
        default=[20, 70, 140, 210],
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
        "--code_length", type=int, default=16, help="code length"
    )
    parser.add_argument(
        "--num_label", type=int, default=38, help="cls numbers"
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
                        help='tag file of training set')
    parser.add_argument('--cap_test',
                        default='./data/mir_test_cap.txt',
                        help='tag file of testing set')
    parser.add_argument('--cap_database',
                        default='./data/mir_database_cap.txt',
                        help='tag file of database set')
    parser.add_argument("--output_dir",
                        default="./save/teacher/mir",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    default_gpu = True

    config = BertConfig.from_json_file(args.config_file)
    config.code_length = args.code_length
    config.num_label = args.num_label

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_output_dir = args.output_dir + '/' + str(args.code_length)
    if not os.path.exists(total_output_dir):
        os.makedirs(total_output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    # Initialize the parameters of the teacher network backbone
    if args.from_pretrained:
        model = VilWithHash.from_pretrained(
            args.from_pretrained, config=config, default_gpu=default_gpu
        )
    else:
        model = VilWithHash(config)

    optimizer_grouped_parameters = []
    for key, value in dict(model.bert.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.finetune_lr, optimizer_grouped_parameters)
    for key, value in dict(model.hash.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)
    for key, value in dict(model.v_hash.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)
    for key, value in dict(model.classifier.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)
    for key, value in dict(model.v_classifier.named_parameters()).items():
        optimizer_grouped_parameters = set_params(key, value, args.learning_rate, optimizer_grouped_parameters)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)

    lr_reduce_list = np.array(args.lr_scheduler_epoch)
    def lr_lambda_fun(epoch):
        return pow(args.lr_decay_value, np.sum(lr_reduce_list <= epoch))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    model.cuda()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    padding_index = 0

    mse_loss = torch.nn.MSELoss()
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

    database_dataset = TeaDataset(
        args.imgfeature_root,
        args.cap_database,
        args.database_list,
        args.max_seq_length,
        args.max_region_num,
        tokenizer,
        padding_index
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True, num_workers=8)

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d num_train_epochs = %d" % (args.train_batch_size, args.num_train_epochs))
    logger.info("  code_length = %d " % (args.code_length))
    logger.info("  lr = %f  finetune_lr = %f" % (args.learning_rate, args.finetune_lr))

    for epochId in range(0, args.num_train_epochs):
        model.train()

        img_loss = 0.0
        text_loss = 0.0
        x_loss = 0.0
        img_clsloss = 0.0
        text_clsloss = 0.0
        codeloss = 0.0
        loss_ban = 0.0
        loss_quan = 0.0
        all_loss = 0.0

        img_prec = 0.0
        text_prec = 0.0

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

            # Get classification prediction and hash code
            _, _, output_t, output_v, code_t, code_v, _, _ = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
            )

            # cal loss
            sim = label.matmul(label.t())
            sim[sim > 1] = 1
            theta_t = 1.0 / 2 * torch.matmul(code_t, code_t.t())
            logloss_t = mse_loss(sim * theta_t, torch.log(1.0 + torch.exp(theta_t))) * args.alpha
            theta_v = 1.0 / 2 * torch.matmul(code_v, code_v.t())
            logloss_v = mse_loss(sim * theta_v, torch.log(1.0 + torch.exp(theta_v))) * args.alpha
            theta_x = 1.0 / 2 * torch.matmul(code_t, code_v.t())
            logloss_x = mse_loss(sim * theta_x, torch.log(1.0 + torch.exp(theta_x))) * args.alpha

            bceloss_t = bce_loss(output_t, label)
            bceloss_v = bce_loss(output_v, label)

            mse_code_x = mse_loss(code_t, code_v)

            zero_vet = torch.zeros(args.code_length).cuda()
            one_mat = torch.ones(label.size(0), args.code_length).cuda()
            err_quan = (mse_loss(code_t.abs(), one_mat) + mse_loss(code_v.abs(), one_mat)) * args.beta
            err_ban = mse_loss(code_t.sum(dim=0).div(label.size(0)), zero_vet) + mse_loss(
                            code_v.sum(dim=0).div(label.size(0)), zero_vet)

            loss = logloss_t + logloss_v + logloss_x + bceloss_t + bceloss_v + mse_code_x + err_quan + err_ban

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            img_loss = img_loss + logloss_v.item()
            text_loss = text_loss + logloss_t.item()
            x_loss = x_loss + logloss_x.item()
            text_clsloss = text_clsloss + bceloss_t.item()
            img_clsloss = img_clsloss + bceloss_v.item()
            codeloss = codeloss + mse_code_x.item()
            loss_quan = loss_quan + err_quan.item()
            loss_ban = loss_ban + err_ban.item()
            all_loss = all_loss + loss.item()

            img_prec = img_prec + precision(output_v, label)
            text_prec = text_prec + precision(output_t, label)

        lr_scheduler.step()

        x_loss = x_loss / len(train_dataloader)
        img_loss = img_loss / len(train_dataloader)
        text_loss = text_loss / len(train_dataloader)
        text_clsloss = text_clsloss / len(train_dataloader)
        img_clsloss = img_clsloss / len(train_dataloader)
        codeloss = codeloss / len(train_dataloader)
        loss_quan = loss_quan / len(train_dataloader)
        loss_ban = loss_ban / len(train_dataloader)
        all_loss = all_loss / len(train_dataloader)

        img_prec = img_prec / len(train_dataset)
        text_prec = text_prec / len(train_dataset)

        logger.info('epoch %d total_loss: %.4f text_loss: %.4f img_loss: %.4f x_loss: %.4f text_bceloss: %.4f '
                    'img_bceloss: %.4f codeloss: %.4f loss_quan: %.4f loss_ban: %.4f text_prec: %.4f img_prec: %.4f '
                    % (epochId, all_loss, text_loss, img_loss, x_loss, text_clsloss, img_clsloss,
                       codeloss, loss_quan, loss_ban, text_prec, img_prec))

        # save and test
        if (epochId + 1) % 5 == 0:

            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            torch.save(model_to_save.state_dict(), '%s/net_epoch_%d.bin' % (total_output_dir, epochId + 1))


    database_dataloader = torch.utils.data.DataLoader(database_dataset, batch_size=args.train_batch_size,
                                                      shuffle=False, num_workers=8)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train_batch_size,
                                                  shuffle=False, num_workers=8)

    for epochId in range(5, args.num_train_epochs+1, 5):
        logger.info("******  Do the evaluation  *********")
        # # Do the evaluation

        model_path = '%s/net_epoch_%d.bin' % (total_output_dir, epochId)
        model.load_state_dict(torch.load(model_path))

        torch.set_grad_enabled(False)

        model.eval()

        img_prec = 0.0
        text_prec = 0.0

        rB_t = []
        rB_v = []
        retrievalL = []
        output_hash_list_t = []
        output_hash_list_v = []
        lab_list = []
        for step, batch in enumerate(database_dataloader):
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

            _, _, output_t, output_v, code_t, code_v, _, _ = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
            )

            code_v_1 = (code_v <= 0.0).float() * -1
            code_v_2 = (code_v > 0.0).float() * 1
            code_v = code_v_1 + code_v_2

            code_t_1 = (code_t <= 0.0).float() * -1
            code_t_2 = (code_t > 0.0).float() * 1
            code_t = code_t_1 + code_t_2

            code_v = code_v.cpu().data.numpy()
            code_t = code_t.cpu().data.numpy()

            rB_t.extend(code_t)
            rB_v.extend(code_v)
            retrievalL.extend(label.cpu().data.numpy())

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

            _, _, output_t, output_v, code_t, code_v, _, _ = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
            )

            code_v_1 = (code_v <= 0.0).float() * -1
            code_v_2 = (code_v > 0.0).float() * 1
            code_v = code_v_1 + code_v_2

            code_t_1 = (code_t <= 0.0).float() * -1
            code_t_2 = (code_t > 0.0).float() * 1
            code_t = code_t_1 + code_t_2

            code_v = code_v.cpu().data.numpy()
            code_t = code_t.cpu().data.numpy()

            output_hash_list_t.extend(code_t)
            output_hash_list_v.extend(code_v)
            lab_list.extend(label.cpu().data.numpy())

            img_prec = img_prec + precision(output_v, label)
            text_prec = text_prec + precision(output_t, label)

        img_prec = img_prec / len(test_dataset)
        text_prec = text_prec / len(test_dataset)

        rB_t = np.array(rB_t).astype(int)
        rB_v = np.array(rB_v).astype(int)
        retrievalL = np.array(retrievalL).astype(int)

        output_hash_arr_t = np.array(output_hash_list_t).astype(int)
        output_hash_arr_v = np.array(output_hash_list_v).astype(int)
        lab_arr = np.array(lab_list).astype(int)

        map_v, p_5000, r_2 = calc_map(qB=output_hash_arr_t, rB=rB_v, queryL=lab_arr, retrievalL=retrievalL, knn=5000)
        logger.info('t->v  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_v, p_5000, r_2))

        map_v, p_5000, r_2 = calc_map(qB=output_hash_arr_v, rB=rB_t, queryL=lab_arr, retrievalL=retrievalL, knn=5000)
        logger.info('v->t  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_v, p_5000, r_2))

        logger.info('*** For Test *** epoch: %d text_prec: %.4f img_prec: %.4f ' % (epochId, text_prec, img_prec))

        torch.set_grad_enabled(True)


if __name__ == "__main__":

    main()
