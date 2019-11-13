set BERT_BASE_DIR="data/uncased_base"
set DATA_DIR="data"
set MODEL="bert_sentence_summarizer_copy_rl"
set DATASET="nyt"
set EXP_NAME="02-12-1"
set DP_RATE="0.14"
set "DECODER_PARAMS=--num_decoder_layers=12 --num_heads=8 --filter_size=3072"
set GPU_LIST="0"
set TRAIN=1
set SELECT_MODEL=1
set TEST=1
set EVAL_ONLY=False
::set "LOG_FILE=--log_file=%DATA_DIR%/log/%MODEL%-%DATASET%-%EXP_NAME%.log"
set LOG_FILE=""

IF not %TRAIN%==1 GOTO train_end

python run.py ^
  %DECODER_PARAMS% ^
  %LOG_FILE% ^
  --model_name=%MODEL% ^
  --task_name=%DATASET% ^
  --mode=train ^
  --data_dir=%DATA_DIR%/%DATASET% ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --output_dir=%DATA_DIR%/%MODEL%-%DATASET%-%EXP_NAME% ^
  --attention_dropout=%DP_RATE% ^
  --residual_dropout=%DP_RATE% ^
  --relu_dropout=%DP_RATE% ^
  --gpu=%GPU_LIST% ^
  --num_train_epochs=4.0 ^
  --learning_rate=3e-4 ^
  --max_seq_length=512 ^
  --evaluate_every_n_step=300 ^
  --train_batch_size=3 ^
  --accumulate_step=12 ^
  --rl_lambda=0.99

:train_end

IF not %SELECT_MODEL%==1 GOTO eval_end

python run.py ^
  %DECODER_PARAMS% ^
  %LOG_FILE% ^
  --model_name=%MODEL% ^
  --task_name=%DATASET% ^
  --mode=eval ^
  --data_dir=%DATA_DIR%/%DATASET% ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --output_dir=%DATA_DIR%/%MODEL%-%DATASET%-%EXP_NAME% ^
  --gpu=%GPU_LIST% ^
  --max_seq_length=512 ^
  --eval_batch_size=25

:eval_end

IF not %TEST%==1 GOTO test_end

python run.py ^
  %DECODER_PARAMS% ^
  %LOG_FILE% ^
  --model_name=%MODEL% ^
  --task_name=%DATASET% ^
  --mode=test ^
  --eval_only=%EVAL_ONLY% ^
  --data_dir=%DATA_DIR%/%DATASET% ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --output_dir=%DATA_DIR%/%MODEL%-%DATASET%-%EXP_NAME% ^
  --gpu=%GPU_LIST% ^
  --max_seq_length=512 ^
  --eval_batch_size=32 ^
  --beam_size=4 ^
  --decode_alpha=1.0

:test_end