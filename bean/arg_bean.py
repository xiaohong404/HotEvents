import torch
from dataclasses import dataclass, field


@dataclass
class BaseArgBean(object):
    """
    参数定义基类
    """
    # 数据文件相关
    train_data_path: str = field(default=None)
    dev_data_path: str = field(default=None)
    test_data_path: str = field(default=None)
    model_save_path: str = field(default=None)

    # -->
    retrieval_train_path: str = field(default=None)
    retrieval_dev_path: str = field(default=None)
    retrieval_test_path: str = field(default=None)
    # -->

    pretrain_model_path: str = field(default=None)
    output_dir: str = field(default=None)

    # 状态相关
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=False)

    # 模型参数相关
    dataloader_proc_num: int = field(default=4)
    epoch_num: int = field(default=5)
    eval_batch_step: int = field(default=2)
    require_improvement_step: int = field(default=1000)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)

    # 超过这个长度将被截断
    max_input_len: int = field(default=32)
    pad_to_max_length: bool = field(default=False)
    learning_rate: float = field(default=0.02)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = field(default=42)

    @property
    def train_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        train_batch_size = self.per_device_train_batch_size * max(1, torch.cuda.device_count())
        return train_batch_size

    @property
    def test_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        test_batch_size = self.per_device_eval_batch_size * max(1, torch.cuda.device_count())
        return test_batch_size

@dataclass
class BERTArgBean(BaseArgBean):
    """
    BERT模型参数定义类
    """
    # 数据处理相关
    # 是否给每个token打标
    label_all_tokens: bool = field(default=False)
    # 将B-Type映射为I-Type
    b2i_label_dict: dict = field(default=None)
    # 标签数量
    label_num: int = field(default=None)

    # 标签ID映射，加载数据后获取
    id_label_dict: dict = field(default=None)
    label_id_dict: dict = field(default=None)

    # 模型参数相关
    bert_config = None
    do_lower_case: bool = field(default=True)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    bert_hidden_size: int = field(default=768)
    dnn_hidden_size: int = field(default=64)
    dropout: float = field(default=0.1)
    adam_epsilon: float = field(default=1e-8)