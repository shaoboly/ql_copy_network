#/bin/bash
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/yegong/cuda9.0/lib64"
#export CUDA_HOME=/home/yegong/cuda9.0/
#export CUDA_VISIBLE_DEVICES=""

export BERT_BASE_DIR="data/uncased_base"
export DATA_DIR="data"
export MODEL="bert_summarizer_dec_v4"
export DATASET="cnn_dm"
export EXP_NAME="02-21-1"
export DP_RATE="0.14"
export DECODER_PARAMS"=--num_decoder_layers=12 --num_heads=12 --filter_size=3072"
export GPU_LIST="0"
export TRAIN=1
export SELECT_MODEL=1
export TEST=1
export EVAL_ONLY=False
export LOG_FILE="--log_file=${DATA_DIR}/log/${MODEL}-${DATASET}-${EXP_NAME}.log"
#export LOG_FILE=""

if [[ ${TRAIN} == 1 ]]
then

python run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --model_name=${MODEL} \
  --task_name=${DATASET} \
  --mode=train \
  --data_dir=${DATA_DIR}/${DATASET} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --output_dir=${DATA_DIR}/${MODEL}-${DATASET}-${EXP_NAME} \
  --attention_dropout=${DP_RATE} \
  --residual_dropout=${DP_RATE} \
  --relu_dropout=${DP_RATE} \
  --gpu=${GPU_LIST} \
  --num_train_epochs=5.0 \
  --learning_rate=3e-4 \
  --max_seq_length=512 \
  --evaluate_every_n_step=300 \
  --train_batch_size=3 \
  --accumulate_step=12 \
  --rl_lambda=0.99 \
  --start_portion_to_feed_draft=0.1 \
  --draft_feed_freq=5 \
  --mask_percentage=0.1 \
  --total_percentage=0.2

fi

if [[ ${SELECT_MODEL} == 1 ]]
then

python run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --model_name=${MODEL} \
  --task_name=${DATASET} \
  --mode=eval \
  --data_dir=${DATA_DIR}/${DATASET} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --output_dir=${DATA_DIR}/${MODEL}-${DATASET}-${EXP_NAME} \
  --gpu=${GPU_LIST} \
  --max_seq_length=512 \
  --eval_batch_size=25

fi

if [[ ${TEST} == 1 ]]
then

python run.py \
  ${DECODER_PARAMS} \
  ${LOG_FILE} \
  --model_name=${MODEL} \
  --task_name=${DATASET} \
  --mode=test \
  --eval_only=${EVAL_ONLY} \
  --data_dir=${DATA_DIR}/${DATASET} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --output_dir=${DATA_DIR}/${MODEL}-${DATASET}-${EXP_NAME} \
  --gpu=${GPU_LIST} \
  --max_seq_length=512 \
  --eval_batch_size=32 \
  --beam_size=4 \
  --decode_alpha=1.0

fi