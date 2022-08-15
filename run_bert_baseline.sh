export CUDA_VISIBLE_DEVICES=3

# 根路径
ROOT_DIR=$(pwd)
TASK_DIR=${ROOT_DIR}/data/

# 预训练模型类型
MODEL_TYPE="bert-base-chinese"

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=${ROOT_DIR}/pretrain_model/${MODEL_TYPE}/
# 微调模型存储路径
FINETUNE_MODEL_DIR=${TASK_DIR}/model
FINETUNE_MODEL_PATH=${FINETUNE_MODEL_DIR}/bert_baseline/bert_base_hotevents.pk

LOG_DIR=log
# 创建相关目录
mkdir -p ${LOG_DIR}

# 用户提供的模型训练、验证、测试文件数据、日志
FORMAT_DATA_DIR=${TASK_DIR}/label_datasets_25w
TRAIN_DATA_PATH=${FORMAT_DATA_DIR}/train_data.csv
DEV_DATA_PATH=${FORMAT_DATA_DIR}/train_data.csv
TEST_DATA_PATH=${FORMAT_DATA_DIR}/test_data.csv
LOG_FILE=${LOG_DIR}/bert_baseline_test.txt

nohup python -u run_model/run_bert_baseline.py \
  --do_train=False \
  --pretrain_model_path=${PRE_TRAINED_MODEL_DIR} \
  --train_data_path=${TRAIN_DATA_PATH} \
  --dev_data_path=${DEV_DATA_PATH} \
  --test_data_path=${TEST_DATA_PATH} \
  --model_save_path=${FINETUNE_MODEL_PATH} \
  > $LOG_FILE 2>&1 &
