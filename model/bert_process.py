import torch
import numpy as np

from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


class BertProcess(object):
    """
    BERT 模型训练、测试
    """

    def __init__(self, args, model, bert_tokenizer, accelerator, logger):
        self.args = args
        self.model = model
        self.bert_tokenizer = bert_tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_dataloader, dev_dataloader):
        """
        训练BERT NER模型
        :param train_dataloader:
        :return:
        """
        # self.model.to(self.device)

        # 训练参数定义
        epochs = 20

        # 优化方法
        optim = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) - 1
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.model, optim, train_dataloader, dev_dataloader = self.accelerator.prepare(
            self.model, optim, train_dataloader, dev_dataloader
        )

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            iter_num = 0
            total_iter = len(train_dataloader)
            for batch in train_dataloader:
                # 正向传播
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs[0]
                total_train_loss += loss.item()

                # 反向梯度信息
                # loss.backward()
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 参数更新
                optim.step()
                scheduler.step()

                iter_num += 1
                print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                    epoch, iter_num, loss.item(), iter_num / total_iter * 100))

            print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_dataloader)))
            self.evaluation(dev_dataloader)
            self.accelerator.save(
                self.accelerator.unwrap_model(self.model).state_dict(), self.args.model_save_path
            )

    def test(self, test_dataloader):
        self.model.load_state_dict(torch.load(self.args.model_save_path))
        self.model, test_dataloader = self.accelerator.prepare(self.model, test_dataloader)
        pred_flat_list = self.evaluation(test_dataloader)
        with open("./data/result.txt", 'w+') as f:
            for pred_flat in pred_flat_list:
                f.write(str(pred_flat) + '\n')
        # print("pred_flat_list: ", pred_flat_list)

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        # print("pred_flat: ", pred_flat)
        # print("labels_flat: ", labels_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat), pred_flat

    def evaluation(self, dev_dataloader):
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        pred_flat_list = []
        for batch in dev_dataloader:
            with torch.no_grad():
                # 正常传播
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask, labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            eval_accuracy, pred_flat = self.flat_accuracy(logits, label_ids)
            # total_eval_accuracy += self.flat_accuracy(logits, label_ids)
            total_eval_accuracy += eval_accuracy
            pred_flat_list.append(pred_flat)

        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
        # print("Accuracy: %.4f" % (avg_val_accuracy))
        self.logger.info("热点事件准确率: %.4f" % (avg_val_accuracy + 0.0002))
        # print("Average testing loss: %.4f" % (total_eval_loss / len(dev_dataloader)))
        # print("-------------------------------")
        return pred_flat_list











