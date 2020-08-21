# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# authors from University of Parma
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers.modeling_albert import AlbertPreTrainedModel, AlbertModel
import torch
from torch.autograd import grad
import math

class AlbertForABSA(AlbertModel):
    def __init__(self, config, num_labels=3, dropout=None, epsilon=None, od=1,use_relu=0):
        super(AlbertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon
        # self.dropout = torch.nn.Dropout(dropout)
        # self.relu = torch.nn.ReLU()
        # self.od1 = math.floor(2.5*config.hidden_size)
        # self.od2 = math.floor(2*config.hidden_size)
        # self.use_relu = use_relu
        # self.preclass = torch.nn.Linear(config.hidden_size, self.od1)
        # self.preclass2 = torch.nn.Linear(self.od1, self.od2)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, albert_emb = self.albert_forward(input_ids, 
                                                token_type_ids, 
                                                attention_mask, 
                                                output_hidden_states=False)
        # sequence_output = self.dropout(sequence_output)

        # sequence_output = self.preclass(sequence_output) ########
        # if self.use_relu:
        #     sequence_output = self.relu(sequence_output)
        # sequence_output = self.dropout(sequence_output)

        # sequence_output = self.preclass2(sequence_output) ########
        # if self.use_relu:
        #     sequence_output = self.relu(sequence_output)
        # sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        if labels is not None:
            _loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if sequence_output.requires_grad: #if training mode
                perturbed_sentence = self.adv_attack(albert_emb, _loss, self.epsilon)
                # perturbed_sentence = self.replace_cls_token(albert_emb, perturbed_sentence) #                
                adv_loss = self.adversarial_loss(perturbed_sentence, attention_mask, labels)
                return _loss, adv_loss
            return _loss
        else:
            return logits

    def adv_attack(self, emb, loss, epsilon):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_sentence

    def adversarial_loss(self, perturbed, attention_mask, labels):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(perturbed, extended_attention_mask)
        encoded_layers_last = encoded_layers[-1]
        # encoded_layers_last = self.dropout(encoded_layers_last)

        # encoded_layers_last = self.preclass(encoded_layers_last) ########
        # if self.use_relu:
        #     encoded_layers_last = self.relu(encoded_layers_last)
        # encoded_layers_last = self.dropout(encoded_layers_last)

        # encoded_layers_last = self.preclass2(encoded_layers_last) ########
        # if self.use_relu:
        #     encoded_layers_last = self.relu(encoded_layers_last)
        # encoded_layers_last = self.dropout(encoded_layers_last)
        
        logits = self.classifier(encoded_layers_last)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss

    def albert_forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(hidden_states=embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoded_layers[-1]
        
        return sequence_output, embedding_output
