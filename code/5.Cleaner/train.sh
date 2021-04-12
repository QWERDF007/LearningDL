#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
source ~/anaconda3/bin/activate torch

DATA=data  # 数据，包含train [val]目录，val可选，但精度可能下降
NUM_CLASSES=2  # 分类数
BATCH_SIZE=20  # 总的batch size，均分到每块卡
EPOCHS=10
CVN=5  # k折交叉验证的k
WORKERS=8  # 读取进程数总数，均分到每个训练进程
LOG_DIR=logs  # 日志目录
LR=0.001  # 学习率
URL=tcp://localhost:8001  # 同步的网络端口
TODAY=$(date "+%Y%m%d%H%M%S")
OUTPUT=output  # 输出目录
DB=test.db  # 输出数据库，可选
TABLE=test  # 输出表名，无此表，则自动创建，字段：id,given,p_given,pred,p_pred
FNAME=error_labels.txt  # 输出文件名

ARCH=efficientnet-b6  # 模型

# 没有目录则创建
if [ ! -d ${OUTPUT} ];then
mkdir -p ${OUTPUT}
fi

# 没有目录则创建
if [ ! -d ${LOG_DIR} ];then
mkdir -p ${LOG_DIR}
fi


# k折交叉验证，获取样本外的预测概率
for ((i=0; i<${CVN}; i++))
do
NOW=$(date "+%Y-%m-%d %H:%M:%S")
echo ${NOW}
echo "Training crossval ${i}"
python train_crossval.py -a ${ARCH} -o ${OUTPUT} -j ${WORKERS} -b ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --pretrained --world-size 1 --rank 0 --dist-url ${URL} --multiprocessing-distributed --cvn ${CVN} --cv ${i} --num-classes ${NUM_CLASSES} ${DATA} &> ${LOG_DIR}/train.${TODAY}.${i}.log
NOW=$(date "+%Y-%m-%d %H:%M:%S")
echo ${NOW}
echo "Finish training crossval ${i}"
sleep 10
done

# 合并交叉验证的结果，生成psx
python train_crossval.py -a ${ARCH} --cvn ${CVN} -o ${OUTPUT} --num-classes ${NUM_CLASSES} --combine-folds ${DATA} &> ${LOG_DIR}/combine.${TODAY}.log

# 获取错误的标签
#python /home/7t/classtrain/Cleaner/get_error_labels.py --psx ${OUTPUT}/train__model_${ARCH}__pyx.npy ${DATA}/train &> ${LOG_DIR}/test.${TODAY}.log

# 输出到文件
#python /home/7t/classtrain/Cleaner/get_error_labels.py --psx ${OUTPUT}/train__model_${ARCH}__pyx.npy -o ${OUTPUT}/${FNAME} ${DATA}/train &> ${LOG_DIR}/test.${TODAY}.log

# 输出到数据库
python get_error_labels.py --psx ${OUTPUT}/train__model_${ARCH}__pyx.npy -db ${OUTPUT}/${DB} -t ${TABLE} ${DATA}/train  &> ${LOG_DIR}/test.${TODAY}.log
