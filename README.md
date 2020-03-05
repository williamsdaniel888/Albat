# Adversarial Training for Aspect-Based Sentiment Analysis with BERT
Code for "[Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/pdf/2001.11316)".

We have used the codebase from the following paper and improved upon their results by applying adversarial training.
"[BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)".

## Problem to Solve
We focus on two major tasks in Aspect-Based Sentiment Analysis (ABSA).

Aspect Extraction (AE): given a review sentence ("The retina display is great."), find aspects("retina display");

Aspect Sentiment Classification (ASC): given an aspect ("retina display") and a review sentence ("The retina display is great."), detect the polarity of that aspect (positive).


### Evaluation
Evaluation wrapper code has been written in ipython notebook ```eval/eval.ipynb```. 
AE ```eval/evaluate_ae.py``` additionally needs Java JRE/JDK to be installed.

## Fine-tuning Setup

step1: make 2 folders for post-training and fine-tuning.
```
mkdir -p pt_model ; mkdir -p run
```
step2: place post-trained BERTs into ```pt_model/```. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

step3: make 3 folders for 3 tasks: 
place fine-tuning data to each respective folder: ```rrc/, ae/, asc/```. A pre-processed data in json format is in `data/json_data.tar.gz`, or can be downloaded [here](https://drive.google.com/file/d/1NGH5bqzEx6aDlYJ7O3hepZF4i_p4iMR8/view?usp=sharing).

step4: fire a fine-tuning from a BERT weight, e.g.
```
cd script
bash run_rrc.sh rrc laptop_pt laptop pt_rrc 10 0
```
In order to run model for AE task, execute the following command:
```bash run_absa.sh ae laptop_pt laptop pt_ae 9 0```
Here, laptop_pt is the post-trained weights for laptop, laptop is the domain, pt_ae is the fine-tuned folder in ```run/```, 9 means run 9 times and 0 means use gpu-0.

similarly,
```
bash run_absa.sh ae rest_pt rest pt_ae 9 0
bash run_absa.sh asc laptop_pt laptop pt_asc 9 0
bash run_absa.sh asc rest_pt rest pt_asc 9 0
```
step5: evaluation

AE: place official evaluation .jar files as ```eval/A.jar``` and ```eval/eval.jar```.
Place testing xml files as (the step 4 of [this](https://github.com/howardhsu/DE-CNN) has a similar setup)
```
ae/official_data/Laptops_Test_Gold.xml
ae/official_data/Laptops_Test_Data_PhaseA.xml
ae/official_data/EN_REST_SB1_TEST.xml.gold
ae/official_data/EN_REST_SB1_TEST.xml.A
```
ASC: built-in as part of ```eval/eval.ipynb```

Open ```result.ipynb``` and check the results.


## Citation

```
@misc{karimi2020adversarial,
    title={Adversarial Training for Aspect-Based Sentiment Analysis with BERT},
    author={Akbar Karimi and Leonardo Rossi and Andrea Prati and Katharina Full},
    year={2020},
    eprint={2001.11316},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
