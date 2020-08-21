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

task=$1
bert=$2
domain=$3
run_dir=$4
runs=$5
epochs=$6

#. ~/anaconda2/etc/profile.d/conda.sh

#conda activate p3-torch10


if ! [ -z $6 ] ; then
    export CUDA_VISIBLE_DEVICES=$7
    echo "using cuda"$CUDA_VISIBLE_DEVICES
fi


DATA_DIR="../"$task/$domain   ################ CHANGE THIS?
python -c "import torch; print('using pytorch', torch.__version__)"
for run in `seq 1 1 $runs`
do
    OUTPUT_DIR="../run/"$run_dir/$domain/$run ########## CHANGE THIS?

    mkdir -p $OUTPUT_DIR
    echo "Run $run"
    if ! [ -e $OUTPUT_DIR/"valid.json" ] ; then
       python ../src/run_$task.py \
           --albert_model $bert --do_train --do_valid \
           --max_seq_length 100 --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs $epochs \
           --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/train_log.txt 2>&1
    fi

    if ! [ -e $OUTPUT_DIR/"predictions.json" ] ; then 
        python ../src/run_$task.py \
            --albert_model $bert --do_eval --max_seq_length 100 \
            --output_dir $OUTPUT_DIR --data_dir $DATA_DIR --seed $run > $OUTPUT_DIR/test_log.txt 2>&1
    fi
    #if [ -e $OUTPUT_DIR/"predictions.json" ] && [ -e $OUTPUT_DIR/model.pt ] ; then
    #    rm $OUTPUT_DIR/model.pt
    #fi
done
