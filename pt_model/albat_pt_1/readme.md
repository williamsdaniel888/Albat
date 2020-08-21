# Further Pre-training with Albert

It is possible to perform further pre-training with Albert using code from Xu et al.'s [BERT for Review Reading Comprehension](https://github.com/howardhsu/BERT-for-RRC-ABSA) repository. We assume that the main Albat repository has been cloned to the directory `/content/`.

**Hardware pre-requisites:** It is strongly recommended to have at least 100 GB of hard disk space, 12 GB of RAM, and a GPU with at least 12 GB of memory (Colab uses either the NVIDIA Tesla K80 or Tesla P100).

**Software pre-requisites:**
"*The code is tested on Ubuntu 18.04 with Python 3.6.9 (Anaconda), PyTorch 1.3 (apex 0.1) and Transformers 2.4.1.*"

# Instructions

If you already have a further pre-trained model (with the files `config.json`,  `pytorch_model.bin`, `spiece.model` and `tokenizer_config.json`), go to step 6. 

## Further Pre-training Albert
1. Install pre-requisites
```
mkdir /content/
cd /content/
pip install  --force-reinstall torch==1.3
pip install  --force-reinstall transformers==2.4.1
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
2. Download training scripts.
```
cd /content/
git clone https://github.com/howardhsu/BERT-for-RRC-ABSA.git
```

3. Download training data to the directory `transformers/data/pt` as per Xu et al.'s [instructions](https://github.com/howardhsu/BERT-for-RRC-ABSA/blob/master/transformers/amazon_yelp.md) and run `prep_domain_tag_corpus.py`.

4. Install patches for the pre-training scripts

```
cd /content/BERT-for-RRC-ABSA/transformers/src
cp /content/Albat/pt_model/albat_pt_1/pt.py /content/BERT-for-RRC-ABSA/transformers/src/pt.py
cd ../script/
cp /content/Albat/pt_model/albat_pt_1/pt.sh /content/BERT-for-RRC-ABSA/transformers/src/pt.sh
```
5. Run further pre-training with `albert-base-v2`.
```
cd /content/BERT-for-RRC-ABSA/transformers/
bash script/pt.sh albert SkipDomBert 0 # preprocessing data takes ~1 hour for 5% subset
cd /content/BERT-for-RRC-ABSA/transformers/pt_runs/pt_albert-SkipDomBert 
zip albat_pt_model config.json pytorch_model.bin spiece.model tokenizer_config.json
mv /content/BERT-for-RRC-ABSA/transformers/pt_runs/pt_albert-SkipDomBert/albat_pt_model.zip /content/albat_pt_model.zip
```
We have found that when using the Tesla P100 training with 1\% of the review corpus takes **~2 hours/epoch**, with 5\% of the review corpus takes **~11 hours/epoch**, and with 10\% of the review corpus takes **~22 hours/epoch**.

## Preparing for Fine-tuning
6. Copy the model files `config.json`,  `pytorch_model.bin`, `spiece.model` and `tokenizer_config.json` to the folder `/content/Albat/pt_model/albat_pt_1/`.
7. In the three scripts `run_ae.py`, `run_asc.py` and `run_e2e.py` you will need to modify the functions **train()** and **test()**.

7a. In **train()**: 
 - Uncomment `tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])`
 - Comment out `tokenizer = ABSATokenizer.from_pretrained("albert-base-v2")`
 - Uncomment `model = AlbertForABSA.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.albert_model], num_labels = len(label_list), epsilon=epsilon)`
 - Comment out `model = AlbertForABSA.from_pretrained("albert-base-v2", num_labels = len(label_list), epsilon=epsilon)`

7b. In **test()**:
 - Uncomment `model = AlbertForABSA.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.albert_model], num_labels = len(label_list), epsilon=epsilon)`
 - Comment out `model = AlbertForABSA.from_pretrained("albert-base-v2", num_labels = len(label_list), epsilon=epsilon)`

8. The pre-trained models can now be loaded for fine-tuning.