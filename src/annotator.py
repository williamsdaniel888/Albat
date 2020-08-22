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

from functools import reduce
from math import floor
from termcolor import colored
import json
import numpy as np
import matplotlib.pyplot as plt
import absa_data_utils as adu
import argparse

def colorizer(tokens,labels,names):
    print('')
    c = 0
    for i in range(len(tokens)):
        l = labels[i]
        is_bold = []
        c+=len(tokens[i])
        if l=='O':
            print(tokens[i],end='')
        elif l=='B-positive' or l=='I-positive':
            if l[0]=='B':
                is_bold=['bold']
            print(colored(tokens[i],color="white",on_color="on_green",attrs=is_bold),end='')
        elif l=='B-negative' or l=='I-negative':
            if l[0]=='B':
                is_bold=['bold']
            print(colored(tokens[i],color="white",on_color="on_red",attrs=is_bold),end='')
        elif l=='B-neutral' or l=='I-neutral':
            if l[0]=='B':
                is_bold=['bold']
            print(colored(tokens[i],color="white",on_color="on_cyan",attrs=is_bold),end='')
        elif l=='B-conflict' or l=='I-conflict':
            if l[0]=='B':
                is_bold=['bold']
            print(colored(tokens[i],color="white",on_color="on_yellow",attrs=is_bold),end='')
        if c >= 60:
            print('')
            c=0
    print('\n')
    return None

def pad(s):
    return " "*(15-len(s))

def extractor(n,prediction_path,truth_path,albert_model):

    with open(prediction_path) as f:
        preds1 = json.load(f)
    with open(truth_path+"test.json") as f:
        truth = json.load(f)
    if albert_model == 'voidful/albert_chinese_base':
        tokenizer = adu.ABSATokenizerB.from_pretrained(albert_model)
    else:
        tokenizer = adu.ABSATokenizer.from_pretrained(albert_model)

    processor = adu.E2EProcessor()
    names = processor.get_labels()
    eval_examples = processor.get_test_examples(truth_path)
    eval_features = adu.cetf(eval_examples, names, 100, tokenizer, "e2e")

    logits = np.argmax(preds1['logits'][n][1:np.where(np.array(preds1['label_ids'][n])==-1)[0][1]],axis=1)
    base = np.array(preds1['label_ids'][n][1:np.where(np.array(preds1['label_ids'][n])==-1)[0][1]])
    cfm1 = np.zeros((9,9))
    for a in range(len(logits)):
        i = logits[a]
        j = base[a]
        if i!=j:
            cfm1[i,j]+=1

    labels = []
    gold = []
    for i in range(len(logits)):
        labels.append(names[logits[i]])
        gold.append(names[eval_features[n][1].label_id[i+1]])

    # print example number
    h = "EXAMPLE #{}\n\nColor key:".format(n)
    print(colored(h,attrs=['bold']))
    print("Non-aspect word: [black on white]")
    print("Positive: ", colored("beginning,",'white','on_green',attrs=['bold']), colored("inside.",'white','on_green'))
    print("Negative: ", colored("beginning,",'white','on_red',attrs=['bold']), colored("inside.",'white','on_red'))
    print("Neutral:  ", colored("beginning,",'white','on_cyan',attrs=['bold']), colored("inside.",'white','on_cyan'))
    print("Conflict: ", colored("beginning,",'white','on_yellow',attrs=['bold']), colored("inside.",'white','on_yellow'))

    # print labeled sentences
    print(colored("\nTrue labeling scheme:",attrs=['bold']))
    colorizer(eval_features[n][0],gold,names)
    print(colored("Labels predicted by Albat:",attrs=['bold']))
    colorizer(eval_features[n][0],labels,names)
    
    # print confusion matrices
    print(colored("CONFUSION MATRIX:",attrs=['bold']))
    plt.figure()
    a = plt.imshow(cfm1)
    b = plt.colorbar(a)
    b.set_label("Error Frequency")
    plt.title("Confusion Matrix for Albat")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.xticks(np.arange(9),labels=names)
    plt.yticks(np.arange(9),labels=names)
    plt.xticks(rotation=90)
    return None
