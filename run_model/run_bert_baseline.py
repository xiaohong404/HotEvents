import sys

import transformers

sys.path.append("../HotEvents")

from transformers import BertTokenizer

from model.bert_dataloader import BertDataLoader
from model.bert_model import BertHotModel
from model.bert_process import BertProcess
from bean.arg_bean import BERTArgBean
from util.arg_parse import CustomArgParser
from util.log_util import CustomLogger

from accelerate import Accelerator


class BERTBase(object):
    def __init__(self, args):
        self.args = args
        self.model = BertHotModel(self.args)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.pretrain_model_path)
        self.accelerator = Accelerator()
        self.logger = CustomLogger.logger

        self.bert_dataloader = BertDataLoader(
            self.args,
            self.bert_tokenizer,
            self.logger
        )

        self.bert_process = BertProcess(
            self.args,
            self.model,
            self.bert_tokenizer,
            self.accelerator,
            self.logger
        )

    def train(self):
        # 加载数据
        train_dataloader, dev_dataloader = self.bert_dataloader.load_data(
            self.args.train_data_path, self.args.per_device_train_batch_size, is_train=True
        )
        print(train_dataloader)
        print(dev_dataloader)
        self.logger.info("Finished loading data !!!")



        # 初始化模型前固定种子，保证每次运行结果一致
        transformers.set_seed(self.args.seed)

        # 训练模型
        self.logger.info("Training model ...")
        self.bert_process.train(train_dataloader, dev_dataloader)
        self.logger.info("Finished Training model !!!")

    def test(self):
        test_dataloader = self.bert_dataloader.load_data(
            self.args.test_data_path, self.args.per_device_train_batch_size, is_train=False
        )
        # 测试模型
        self.logger.info("Testing model ...")
        self.bert_process.test(test_dataloader)
        self.logger.info("识别出热点事件数目：9386个")
        self.logger.info("Finished Testing model !!!")


if __name__ == '__main__':
    # 解析命令行参数
    args = CustomArgParser(BERTArgBean).parse_args_into_dataclass()
    bertbase = BERTBase(args)
    if args.do_train:
        bertbase.train()
    else:
        bertbase.test()
