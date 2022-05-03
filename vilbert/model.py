from vilbert.vilbert import (
    BertModel,
    BertPreTrainedModel,

)
import torch.nn as nn
import torchvision
import torch
import numpy as np
import copy
import math


# for teacher network classification pre-train
class VilForCLS(BertPreTrainedModel):
    def __init__(self, config):
        super(VilForCLS, self).__init__(config)
        self.num_label = config.num_label
        self.bert = BertModel(config)  # ViLBert模型
        self.classification = nn.Sequential(
            nn.Linear(config.bi_hidden_size, self.num_label),
            nn.Sigmoid()
        )
        self.v_classification = nn.Sequential(
            nn.Linear(config.bi_hidden_size, self.num_label),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):

        _, _, pooled_output_t, pooled_output_v, _ = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
        )
        output_t = self.classification(pooled_output_t)
        output_v = self.v_classification(pooled_output_v)

        return pooled_output_t, pooled_output_v, output_t, output_v


# teacher hashing network
class VilWithHash(BertPreTrainedModel):
    def __init__(self, config):
        super(VilWithHash, self).__init__(config)

        self.num_label = config.num_label
        self.code_length = config.code_length
        self.bert = BertModel(config)  # ViLBert model
        self.hash = nn.Sequential(
            nn.Linear(config.bi_hidden_size, self.code_length),
            nn.Tanh()
        )
        self.v_hash = nn.Sequential(
            nn.Linear(config.bi_hidden_size, self.code_length),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.code_length, self.num_label),
            nn.Sigmoid()
        )
        self.v_classifier = nn.Sequential(
            nn.Linear(self.code_length, self.num_label),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
    ):
        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, _ = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
        )
        code_t = self.hash(pooled_output_t)
        code_v = self.v_hash(pooled_output_v)
        output_t = self.classifier(code_t)
        output_v = self.v_classifier(code_v)

        return pooled_output_t, pooled_output_v, output_t, output_v, code_t, code_v, encoded_layers_t, encoded_layers_v

