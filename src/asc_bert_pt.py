# Copyright 2020 Daniel Williams.
# Contains code contributions by the Google AI Language Team, HuggingFace Inc.,
# NVIDIA CORPORATION, authors from the University of Illinois at Chicago, and 
# authors from the University of Parma and Adidas AG.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from pytorch_pretrained_albert.modeling import AlbertPreTrainedModel, AlbertModel
from transformers.modeling_albert import AlbertPreTrainedModel, AlbertModel
import torch
from torch.autograd import Variable, grad

# class AlbertForABSA(AlbertModel):
class AlbertForABSA(AlbertModel):
    def __init__(self, config, num_labels=3, dropout=None):
        # super(AlbertForABSA, self).__init__(config)
        super(AlbertForABSA, self).__init__(config)
        self.num_labels = num_labels
        # self.albert = AlbertModel(config)
        self.albert = AlbertModel(config)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_albert_weights)
        self.init_weights

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.albert(input_ids, token_type_ids, attention_mask)#, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            _loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return _loss
        else:
            return logits

