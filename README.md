# HotEvents

## 1.登陆服务器获取github项目
git clone git@github.com:xiaohong404/HotEvents.git

## 2.进入项目文件夹
cd HotEvents

## 3.从服务器公共文件夹复制所需文件
cp -rf /data/dl4nlp/pretrain_model ./

cp /data/dl4nlp/HotEvents/bert_base_hotevents.pk ./data/model/bert_baseline/

## 4.运行文件
sh run_bert_baseline.sh

## 5.查看运行结果
tail -f log/bert_baseline_test.txt

