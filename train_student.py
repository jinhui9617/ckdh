import os
import argparse
import numpy as np
import scipy.io as io

import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import random
from torch.optim.lr_scheduler import (
    LambdaLR,
)

from vilbert.calc_hr import calc_map
from vilbert.net import CKDH, ImgBackbone, TextBackbone
import logging


from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.vilbert import BertConfig
from vilbert.datasets.hash_dataset import (
    DatasetWithAll,
    StuDataSet,
)

from vilbert.model import (
    VilWithHash,
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

    #vilbert parameters
    parser.add_argument(
        "--from_pretrained",
        # default="bert-base-uncased",
        default="./save/teacher/mir/net_epoch.bin",
        type=str,
        help="teacher netword model path",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, roberta-base",
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
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )


    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='number of data loading workers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='input batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=250,
                        help='number of epochs to train for')
    parser.add_argument("--lr_scheduler_epoch",
                        default=[50, 100, 200],
                        help="lr change epoch")
    parser.add_argument("--lr_decay_value",
                        default=0.2,
                        type=float,
                        help="lr decay value")
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--lrMul', type=float, default=100.)
    parser.add_argument('--beta',
                        type=float,
                        default=0.01,
                        help='hyper-param for quantification loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='manual seed')
    parser.add_argument('--y_dim', type=int, default=1386,
                        help='binary text length')
    parser.add_argument('--image_dir',
                        default="./data/mir/image",
                        help='path to dataset')
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
    parser.add_argument('--tag_train',
                        default='./data/mir_train_tag.txt',
                        help='tag file of training set')
    parser.add_argument('--tag_test',
                        default='./data/mir_test_tag.txt',
                        help='tag file of testing set')
    parser.add_argument('--tag_database',
                        default='./data/mir_database_tag.txt',
                        help='tag file of database set')
    parser.add_argument('--num_label',
                        type=int,
                        default=38,
                        help='cls numbers')
    parser.add_argument('--code_length',
                        type=int,
                        default=16,
                        help='code numbers')
    parser.add_argument("--output_dir",
                        default="./save/student/mir",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_mode",
                        default=3,
                        type=int,
                        help="1. train 2.train with val 3. train then val  4. val")



    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    total_output_dir = args.output_dir + '/' + str(args.code_length)
    if not os.path.exists(total_output_dir):
        os.makedirs(total_output_dir)

    config = BertConfig.from_json_file(args.config_file)
    config.code_length = args.code_length
    config.batch_size = args.batch_size
    config.max_region_num = args.max_region_num
    config.max_seq_length = args.max_seq_length
    config.num_label = args.num_label

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    # Load teacher network parameters
    default_gpu = True
    if args.from_pretrained:
        model = VilWithHash.from_pretrained(
            args.from_pretrained, config=config, default_gpu=default_gpu
        )
    else:
        model = VilWithHash(config)

    for p in model.parameters():
        p.requires_grad = False

    model.to(device)

    # Create student network
    dcmh = CKDH(args.code_length, args.y_dim, args.num_label)
    dcmh.to(device)

    dcmh.to(device)

    padding_index = 0
    train_dataset = DatasetWithAll(
        args.imgfeature_root,
        args.image_dir,
        args.cap_train,
        args.train_list,
        args.tag_train,
        args.max_seq_length,
        args.max_region_num,
        tokenizer,
        padding_index,
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

    re_dataset = StuDataSet(
        args.image_dir,
        args.database_list,
        args.tag_database,
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

    base_params_id_v = list(map(id, dcmh.imgnet.alexnet.parameters()))
    # base_params_id_t = list(map(id, dcmh.textnet.transition.parameters()))
    base_params_id = base_params_id_v #+ base_params_id_t
    tune_params = filter(lambda p: id(p) not in base_params_id, dcmh.parameters())
    base_params = filter(lambda p: id(p) in base_params_id, dcmh.parameters())

    optimizer = optim.SGD([{'params': base_params},
                           {'params': tune_params, 'lr': args.lr * args.lrMul}],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)


    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    lr_reduce_list = np.array(args.lr_scheduler_epoch)
    def v_lr_lambda_fun(epoch):
        return pow(args.lr_decay_value, np.sum(lr_reduce_list <= epoch))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=v_lr_lambda_fun)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    logger.info("***** Running training *****")
    logger.info("  code_length = %d output_dir = %s" % (args.code_length, total_output_dir))
    logger.info("  Batch size = %d num_train_epochs = %d" % (args.batch_size, args.epochs))
    logger.info("  lr = %f  lrMul = %f" % (args.lr, args.lrMul))

    if args.train_mode != 4:
        for epochId in range(0, args.epochs):
            model.eval()
            dcmh.train()

            img_loss = 0.0
            text_loss = 0.0
            x_loss = 0.0

            img_clsloss = 0.0
            text_clsloss = 0.0
            img_codeloss = 0.0
            text_codeloss = 0.0
            loss_ban = 0.0
            loss_quan = 0.0

            all_loss = 0.0

            img_prec = 0.0
            text_prec = 0.0

            for step, batch in enumerate(train_dataloader):
                image, image_feat, image_loc, image_mask, input_ids, input_mask, segment_ids, image_id, label, tag = (
                    batch
                )

                image = image.to(device)
                image_feat = image_feat.to(device)
                image_loc = image_loc.to(device)
                image_mask = image_mask.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label = label.to(device)
                tag = tag.to(device)

                pooled_output_t, pooled_output_v, output_t, output_v, code_text, code_img, encoded_layers_t, encoded_layers_v = model(
                    input_ids,
                    image_feat,
                    image_loc,
                    segment_ids,
                    input_mask,
                    image_mask,
                )

                sim = label.matmul(label.t())
                sim[sim > 1] = 1

                fea_v, fea_t, code_v, code_t, label_v, label_t = dcmh(image, tag)

                # cal loss
                theta_t = 1.0 / 2 * torch.matmul(code_t, code_t.t())
                logloss_t = mse_loss(sim * theta_t, torch.log(1.0 + torch.exp(theta_t)))
                theta_v = 1.0 / 2 * torch.matmul(code_v, code_v.t())
                logloss_v = mse_loss(sim * theta_v, torch.log(1.0 + torch.exp(theta_v)))
                theta_x = 1.0 / 2 * torch.matmul(code_t, code_v.t())
                logloss_x = mse_loss(sim * theta_x, torch.log(1.0 + torch.exp(theta_x)))

                bceloss_t = bce_loss(label_t, label)
                bceloss_v = bce_loss(label_v, label)

                mse_code_t = mse_loss(code_t, code_text)
                mse_code_v = mse_loss(code_v, code_img)

                one_mat = torch.ones(label.size(0), args.code_length).cuda()
                err_quan = (mse_loss(code_t.abs(), one_mat) + mse_loss(code_v.abs(), one_mat)) * 0.01

                zero_vet = torch.zeros(args.code_length).cuda()
                err_ban = (mse_loss(code_t.sum(dim=0).div(label.size(0)), zero_vet) + mse_loss(
                    code_v.sum(dim=0).div(label.size(0)), zero_vet))

                err = mse_code_t + mse_code_v + logloss_t + logloss_v + logloss_x + err_ban + err_quan + bceloss_t + bceloss_v

                optimizer.zero_grad()
                err.backward()
                optimizer.step()

                text_loss = text_loss + logloss_t.item()
                img_loss = img_loss + logloss_v.item()
                x_loss = x_loss + logloss_x.item()
                text_clsloss = text_clsloss + bceloss_t.item()
                img_clsloss = img_clsloss + bceloss_v.item()
                text_codeloss = text_codeloss + mse_code_t.item()
                img_codeloss = img_codeloss + mse_code_v.item()
                loss_quan = loss_quan + err_quan.item()
                loss_ban = loss_ban + err_ban.item()

                all_loss = all_loss + err.item()

                img_prec = img_prec + precision(label_v, label)
                text_prec = text_prec + precision(label_t, label)

            lr_scheduler.step()

            text_loss = text_loss / len(train_dataloader)
            img_loss = img_loss / len(train_dataloader)
            x_loss = x_loss / len(train_dataloader)
            text_clsloss = text_clsloss / len(train_dataloader)
            img_clsloss = img_clsloss / len(train_dataloader)
            text_codeloss = text_codeloss / len(train_dataloader)
            img_codeloss = img_codeloss / len(train_dataloader)
            loss_quan = loss_quan / len(train_dataloader)
            loss_ban = loss_ban / len(train_dataloader)

            all_loss = all_loss / len(train_dataloader)

            img_prec = img_prec / len(train_dataset)
            text_prec = text_prec / len(train_dataset)

            logger.info('epoch %d total_loss: %.4f text_loss: %.4f img_loss: %.4f x_loss: %.4f text_clsloss: %.4f '
                        'img_clsloss: %.4f text_codeloss: %.4f img_codeloss: %.4f '
                        'loss_ban:: %.4f loss_quan:: %.4f text_prec: %.4f img_prec: %.4f '
                        % (epochId, all_loss, text_loss, img_loss, x_loss, text_clsloss, img_clsloss, text_codeloss,
                           img_codeloss, loss_ban, loss_quan,
                           text_prec, img_prec))

            # save and test
            if ((epochId + 1) % 5 == 0):
                model_to_save = (
                    dcmh.module if hasattr(dcmh, "module") else dcmh
                )  # Only save the model it-self
                torch.save(model_to_save.state_dict(), '%s/net_epoch_%d.pth' % (total_output_dir, epochId + 1))

                if args.train_mode == 2:
                    logger.info("******  Do the evaluation  *********")
                    # Do the evaluation

                    re_dataloader = torch.utils.data.DataLoader(re_dataset, batch_size=args.batch_size,
                                                                    shuffle=False, num_workers=args.workers)

                    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                                  shuffle=False, num_workers=args.workers)

                    torch.set_grad_enabled(False)

                    dcmh.eval()

                    img_prec = 0.0
                    text_prec = 0.0

                    rB_t = []
                    rB_v = []
                    retrievalL = []
                    output_hash_list_t = []
                    output_hash_list_v = []
                    lab_list = []
                    for step, batch in enumerate(re_dataloader):
                        image, tag, label, filename = (
                            batch
                        )

                        image = image.to(device)
                        tag = tag.to(device)
                        label = label.to(device)

                        fea_v, fea_t, code_v, code_t, label_v, label_t = dcmh(image, tag)

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
                        image, tag, label, filename = (
                            batch
                        )

                        image = image.to(device)
                        tag = tag.to(device)
                        label = label.to(device)

                        fea_v, fea_t, code_v, code_t, label_v, label_t = dcmh(image, tag)

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

                        img_prec = img_prec + precision(label_v, label)
                        text_prec = text_prec + precision(label_t, label)

                    img_prec = img_prec / len(test_dataset)
                    text_prec = text_prec / len(test_dataset)

                    rB_t = np.array(rB_t).astype(int)
                    rB_v = np.array(rB_v).astype(int)
                    retrievalL = np.array(retrievalL).astype(int)

                    output_hash_arr_t = np.array(output_hash_list_t).astype(int)
                    output_hash_arr_v = np.array(output_hash_list_v).astype(int)
                    lab_arr = np.array(lab_list).astype(int)

                    map_tv, p_5000_1, r_2_1 = calc_map(qB=output_hash_arr_t, rB=rB_v, queryL=lab_arr, retrievalL=retrievalL,
                                                  knn=5000)
                    logger.info('t->v  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_tv, p_5000_1, r_2_1))

                    map_vt, p_5000_2, r_2_2 = calc_map(qB=output_hash_arr_v, rB=rB_t, queryL=lab_arr, retrievalL=retrievalL,
                                                  knn=5000)
                    logger.info('v->t  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_vt, p_5000_2, r_2_2))

                    logger.info('*** For Test *** epoch: %d text_prec: %.4f img_prec: %.4f ' % (epochId, text_prec, img_prec))

                    torch.set_grad_enabled(True)

    if (args.train_mode == 3) | (args.train_mode == 4):
        re_dataloader = torch.utils.data.DataLoader(re_dataset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.workers)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=args.workers)

        for epochId in range(5, args.epochs+1, 5):
            logger.info("******  Do the evaluation  *********")
            # # Do the evaluation

            model_path = '%s/net_epoch_%d.pth' % (total_output_dir, epochId)
            dcmh.load_state_dict(torch.load(model_path))

            torch.set_grad_enabled(False)

            dcmh.eval()

            img_prec = 0.0
            text_prec = 0.0

            rB_t1 = []
            rB_v1 = []
            output_hash_list_t1 = []
            output_hash_list_v1 = []

            rB_t = []
            rB_v = []
            retrievalL = []
            output_hash_list_t = []
            output_hash_list_v = []
            lab_list = []
            for step, batch in enumerate(re_dataloader):
                image, tag, label, filename = (
                    batch
                )

                image = image.to(device)
                tag = tag.to(device)
                label = label.to(device)

                fea_v, fea_t, code_v, code_t, label_v, label_t = dcmh(image, tag)

                rB_t1.extend(code_t.cpu().data.numpy())
                rB_v1.extend(code_v.cpu().data.numpy())

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
                image, tag, label, filename = (
                    batch
                )

                image = image.to(device)
                tag = tag.to(device)
                label = label.to(device)

                fea_v, fea_t, code_v, code_t, label_v, label_t = dcmh(image, tag)

                output_hash_list_t1.extend(code_t.cpu().data.numpy())
                output_hash_list_v1.extend(code_v.cpu().data.numpy())

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

                img_prec = img_prec + precision(label_v, label)
                text_prec = text_prec + precision(label_t, label)

            img_prec = img_prec / len(test_dataset)
            text_prec = text_prec / len(test_dataset)

            rB_t = np.array(rB_t).astype(int)
            rB_v = np.array(rB_v).astype(int)
            retrievalL = np.array(retrievalL).astype(int)

            output_hash_arr_t = np.array(output_hash_list_t).astype(int)
            output_hash_arr_v = np.array(output_hash_list_v).astype(int)
            lab_arr = np.array(lab_list).astype(int)


            rB_t1 = np.array(rB_t1)
            rB_v1 = np.array(rB_v1)
            output_hash_arr_t1 = np.array(output_hash_list_t1)
            output_hash_arr_v1 = np.array(output_hash_list_v1)

            mat_path = '/home/10201008/data/hash_code/dcbh_coco_16.mat'

            io.savemat(mat_path, {'rB_t': rB_t1, 'rB_v': rB_v1, 'reL': retrievalL,
                                  'qB_t': output_hash_arr_t1, 'qB_v': output_hash_arr_v1, 'quL': lab_arr})

            map_tv, p_5000_1, r_2_1 = calc_map(qB=output_hash_arr_t, rB=rB_v, queryL=lab_arr, retrievalL=retrievalL,
                                               knn=5000)
            logger.info('t->v  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_tv, p_5000_1, r_2_1))

            map_vt, p_5000_2, r_2_2 = calc_map(qB=output_hash_arr_v, rB=rB_t, queryL=lab_arr, retrievalL=retrievalL,
                                               knn=5000)
            logger.info('v->t  epoch: %d | map: %f | p_5000: %f | r_2: %f' % (epochId, map_vt, p_5000_2, r_2_2))

            logger.info('*** For Test *** epoch: %d text_prec: %.4f img_prec: %.4f ' % (epochId, text_prec, img_prec))

            torch.set_grad_enabled(True)

if __name__ == '__main__':
    main()


