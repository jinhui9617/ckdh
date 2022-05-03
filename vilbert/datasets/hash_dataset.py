import torch
from torch.utils.data import Dataset
import numpy as np
import copy
from pytorch_transformers.tokenization_bert import BertTokenizer
from PIL import Image
import os
import scipy.io as io


# load tag info
def _load_caption(caption_path, list_path):
    entries = []
    with open(caption_path, "r") as f:
        for line in f:
            line = line[0:-1]
            image_id = line.split(' ')[0]
            words = line.split(' ')[1:]
            str = ''
            sentence = str.join(words)
            entries.append({"caption": sentence, "image_id": image_id})

    return entries


# load image info
def _load_img_info(img_root, img_name):
    index = img_name.split('.')[0]
    img_path = img_root + index + '.npy'

    reader = np.load(img_path, allow_pickle=True)
    item = {}
    item["image_id"] = reader.item().get("image_id")
    item["image_h"] = reader.item().get("image_height")
    item["image_w"] = reader.item().get("image_width")
    item["num_boxes"] = reader.item().get("num_boxes")
    item["boxes"] = reader.item().get("bbox")
    item["features"] = reader.item().get("features")

    image_id = item["image_id"]
    image_h = int(item["image_h"])
    image_w = int(item["image_w"])
    # num_boxes = int(item['num_boxes'])

    features = item["features"].reshape(-1, 2048)
    boxes = item["boxes"].reshape(-1, 4)

    num_boxes = features.shape[0]
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1
    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location_ori = copy.deepcopy(image_location)
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
    image_location_ori = np.concatenate(
        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
    )

    return features, num_boxes, image_location, image_location_ori


# load images' label
def get_names_labels(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data_list = []
    data_label_dict = dict()
    for line in lines:
        file_name = line.split(' ')[0]
        data_list.append(file_name)
        labels_str = line.strip().split(' ')[1:]
        labels_arr = np.array([int(lab) for lab in labels_str])
        data_label_dict[file_name] = labels_arr

    return data_list, data_label_dict


# load 0/1 text label
def get_names_tags(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data_list = []
    data_tag_dict = dict()
    for line in lines:
        file_name = line.split(' ')[0]
        data_list.append(file_name)
        tags_str = line.strip().split(' ')[1:]
        tags_arr = np.array([int(lab) for lab in tags_str])
        data_tag_dict[file_name] = tags_arr

    return data_list, data_tag_dict


class StuDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 label_path,
                 tag_path,
                 transform_pre=None,
                 transform_totensor=None):

        self.image_dir = image_dir

        self.file_list, self.file_label_dict = get_names_labels(label_path)
        _, self.file_tag_dict = get_names_labels(tag_path)

        self.transform_pre = transform_pre
        self.transform_totensor = transform_totensor

    def __getitem__(self, index):

        file_name = self.file_list[index]
        img_raw = Image.open(os.path.join(self.image_dir, file_name)).convert('RGB')
        img_pre = self.transform_pre(img_raw)
        img_tensor = self.transform_totensor(img_pre)

        label = torch.FloatTensor(self.file_label_dict[file_name])
        tag = torch.FloatTensor(self.file_tag_dict[file_name])

        return img_tensor, tag, label, file_name

    def __len__(self):
        return len(self.file_list)


class TeaDataset(Dataset):
    def __init__(
        self,
        imgfeature_root,
        caption_path,
        label_path,
        max_seq_length,
        max_region_num,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
    ):
        self.imgfeature_root = imgfeature_root
        self.caption_path = caption_path
        self.label_path = label_path

        self.file_list, self.file_label_dict = get_names_labels(label_path)

        self._entries = _load_caption(self.caption_path, self.file_list)

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length

        self.tokenize()
        self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:

            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = [101] + tokens + [102]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        entry = self._entries[index]

        image_id = entry["image_id"]

        label = torch.FloatTensor(self.file_label_dict[image_id])

        features, num_boxes, image_location, _ = _load_img_info(self.imgfeature_root, image_id)

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = image_location[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        caption_ori = entry["caption"]
        caption = entry["token"].long()
        input_mask = entry["input_mask"].long()
        segment_ids = entry["segment_ids"].long()

        return (
            features,
            spatials,
            image_mask,
            caption,
            input_mask,
            segment_ids,
            image_id,
            label,
            caption_ori,
        )

    def __len__(self):
        return len(self._entries)


class DatasetWithAll(Dataset):
    def __init__(
        self,
        imgfeature_root,
        img_root,
        cap_path,
        label_path,
        tag_path,
        max_seq_length,
        max_region_num,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        transform_pre=None,
        transform_totensor=None,
    ):

        self.imgfeature_root = imgfeature_root
        self.img_root = img_root
        self.cap_path = cap_path
        self.label_path = label_path
        self.tag_path = tag_path

        self.file_list, self.file_label_dict = get_names_labels(label_path)
        _, self.file_tag_dict = get_names_labels(tag_path)

        self._entries = _load_caption(self.cap_path, self.file_list)

        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length

        self.transform_pre = transform_pre
        self.transform_totensor = transform_totensor

        self.tokenize()
        self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:

            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = [101] + tokens + [102]
            # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
            # tokens = tokens[: self._max_seq_length - 1]
            # g_token = 0
            # for token in tokens:
            #     g_token += token
            # if len(tokens) > 0:
            #     g_token = g_token // len(tokens)
            # tokens = [g_token] + tokens

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            entry["token_ori"] = tokens

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._entries:
            token_ori = torch.from_numpy(np.array(entry["token_ori"]))
            entry["token_ori"] = token_ori

            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        entry = self._entries[index]

        image_id = entry["image_id"]

        label = torch.FloatTensor(self.file_label_dict[image_id])
        tag = torch.FloatTensor(self.file_tag_dict[image_id])

        file_path = self.file_list[index]
        img_raw = Image.open(os.path.join(self.img_root, file_path)).convert('RGB')
        img_pre = self.transform_pre(img_raw)
        img_tensor = self.transform_totensor(img_pre)

        features, num_boxes, image_location, _ = _load_img_info(self.imgfeature_root, image_id)

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = image_location[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        caption_ori = entry["token_ori"].long()
        caption = entry["token"].long()
        input_mask = entry["input_mask"].long()
        segment_ids = entry["segment_ids"].long()

        return (
            img_tensor,
            features,
            spatials,
            image_mask,
            caption,
            input_mask,
            segment_ids,
            image_id,
            label,
            tag
        )

    def __len__(self):
        return len(self._entries)

