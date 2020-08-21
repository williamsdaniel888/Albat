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

import os
import logging
import argparse
import random
import json
from math import ceil
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from transformers.tokenization_albert import AlbertTokenizer
from transformers.modeling_albert import AlbertModel, AlbertPreTrainedModel

import absa_data_utils as data_utils
from absa_data_utils import ABSATokenizer
import modelconfig
from albat_e2e import AlbertForABSA

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args):
    start = time.time()
    torch.cuda.empty_cache()
    # e2e best values
    epsilon = 2
    wdec = 1e-1
    
    processor = data_utils.E2EProcessor()
    label_list = processor.get_labels()
    # tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.albert_model])
    tokenizer = ABSATokenizer.from_pretrained("albert-base-v2")

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, "e2e")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    #>>>>> validation
    if args.do_valid:
        valid_examples = processor.get_dev_examples(args.data_dir)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "e2e")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)    

        best_valid_loss=float('inf')
        valid_losses=[]
    #<<<<< end of validation declaration

    # model = AlbertForABSA.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.albert_model], num_labels = len(label_list), epsilon=epsilon)
    model = AlbertForABSA.from_pretrained("albert-base-v2", num_labels = len(label_list), epsilon=epsilon)
    
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Model Properties *****")
    logger.info("  Parameters (Total): {:.2e}".format(params_total))
    logger.info("  Parameters (Trainable): {:.2e}".format(params_trainable))
    
    model.to(device)
    
    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': wdec},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)#
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*t_total), num_training_steps=t_total)

    global_step = 0
    model.train()
    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch
            
            _loss, adv_loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss = _loss + adv_loss
            loss.backward()
            
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            #>>>> perform validation at the end of each epoch .
        if args.do_valid:
            model.eval()
            with torch.no_grad():
                losses=[]
                valid_size=0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                    input_ids, segment_ids, input_mask, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    losses.append(loss.data.item()*input_ids.size(0) )
                    valid_size+=input_ids.size(0)
                valid_loss=sum(losses)/valid_size
                logger.info("validation loss: %f", valid_loss)
                valid_losses.append(valid_loss)
            if valid_loss<best_valid_loss:
                torch.save(model, os.path.join(args.output_dir, "model.pt") )
                best_valid_loss=valid_loss
            model.train()
    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        torch.save(model, os.path.join(args.output_dir, "model.pt") )
    mstats = torch.cuda.memory_stats()
    duration = time.time()-start
    logger.info("Training completed in {} minutes, {} seconds".format(duration//60,ceil(duration%60)))
    logger.info("***** GPU Memory Statistics *****")
    logger.info("  Allocated bytes (Peak):      {} MiB".format(mstats['allocated_bytes.all.peak']/1048576))
    logger.info("  Allocated bytes (Allocated): {} MiB".format(mstats['allocated_bytes.all.allocated']/1048576))


def test(args):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)  
    start = time.time()
    torch.cuda.empty_cache()
    processor = data_utils.E2EProcessor()
    label_list = processor.get_labels()
    # tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.albert_model])
    tokenizer = ABSATokenizer.from_pretrained("albert-base-v2")

    eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list,
     args.max_seq_length, tokenizer, "e2e")

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = torch.load(os.path.join(args.output_dir, "model.pt") )
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Model Properties *****")
    logger.info("  Parameters (Total): {:.2e}".format(params_total))
    logger.info("  Parameters (Trainable): {:.2e}".format(params_trainable))
    model.to(device)
    model.eval()
    
    full_logits=[]
    full_label_ids=[]
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.cpu().numpy()

        full_logits.extend(logits.tolist() )
        full_label_ids.extend(label_ids.tolist() )

    output_eval_json = os.path.join(args.output_dir, "predictions.json") 
    with open(output_eval_json, "w") as fw:
        json.dump({"logits": full_logits, "label_ids": full_label_ids}, fw,indent=4)
        # assert len(full_logits)==len(eval_examples)
        # #sort by original order for evaluation
        # recs={}
        # for qx, ex in enumerate(eval_examples):
        #     recs[int(ex.guid.split("-")[1]) ]={"sentence": ex.text_a, "idx_map": ex.idx_map, "logit": full_logits[qx][1:]} #skip the [CLS] tag.
        # full_logits=[recs[qx]["logit"] for qx in range(len(full_logits))]
        # raw_X=[recs[qx]["sentence"] for qx in range(len(eval_examples) ) ]
        # idx_map=[recs[qx]["idx_map"] for qx in range(len(eval_examples)) ]
        # json.dump({"logits": full_logits, "raw_X": raw_X, "idx_map": idx_map}, fw, indent=4)
    mstats = torch.cuda.memory_stats()
    duration = time.time()-start
    logger.info("Testing completed in {} minutes, {} seconds".format(duration//60,ceil(duration%60)))
    logger.info("***** GPU Memory Statistics *****")
    logger.info("  Allocated bytes (Peak):      {} MiB".format(mstats['allocated_bytes.all.peak']/1048576))
    logger.info("  Allocated bytes (Allocated): {} MiB".format(mstats['allocated_bytes.all.allocated']/1048576))

def main():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--albert_model", default='albert-base', type=str)

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir containing json files.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--num_train_epochs",
                        default=6,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)
            
if __name__=="__main__":
    main()
