# ccNERx

- ICCSupervised
- CC
    - loaders
        - utils
            - embedding.py
            - file_util.py
            - lexicon_factory.py
            - lexicon_tree.py
            - vocab.py
        - cn_loader.py
        - le_loader.py
    - analysis.py
    - birnncrf.py
    - crf.py
    - dataloader.py
    - LEBert.py
    - model.py
    - predicter.py
    - trainer.py
    - utils.py

## Requirements

**environment**

```bash
transformers==4.5.1
torch
```

**LLoader requirements**

1. Word vocab

[Word vocab](https://drive.google.com/file/d/1UmtbCSPVrXBX_y4KcovCknJFu9bXXp12/view?usp=sharing)

2. Word embedding

[Chinese word embedding](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz)

## Demo


数据集：
ccks，cdd，FN（福能提供的数据集，医院的体检文本，分成福建和四川的，其中福建的标注数据量较多，四川标注的数据量只有500条，我们实验室人工标注的）
weibo，ontonotes，msra

**pretrain**
工作点1：
预训练模型，在训练的过程中加入提示信息，如：原句：“小明是一个中国人”，提示信息会转变成“小是一个实体，明是一个实体，是是一个非实体，依此类推”，预训练过程中会将原句和提示信息一同输入到模型中

# %% 
```python
# Bert预训练
import os
args = {
    'num_epochs': 50,
    'num_gpus': [0],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    # 'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    'pretrained_file_name': './save_pretrained/weibo_256_pre_1/Bert_8450/pytorch_model.bin',
    'max_seq_length': 256,
    'max_scan_num': 1000000,
    'train_file': './data/weibo/train.json',
    'eval_file': './data/weibo/dev.json',
    'test_file': './data/weibo/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/weibo/labels.txt',
    'loader_name': 'ptloader_v2',
    "word_embedding_file": "./data/tencent/word_embedding.txt",
    "word_vocab_file": "./data/tencent/tencent_vocab.txt",
    "word_vocab_file_with_tag": "./data/tencent/tencent_vocab_with_tag.json",
    "default_tag": "O",
    'batch_size': 8,
    'eval_batch_size': 32,
    'pass_none_rule': True,
    'skip_single_matched_word': True,
    'do_shuffle': True,
    'task_name': 'weibo_256_pre_2',
    "use_gpu": True,
    "debug": True,
    "tag_rules": {
        "O": "非实体",
        "PER.NOM": "指代人名",
        "LOC.NAM": "地名",
        "PER.NAM": "人名",
        "GPE.NAM": "政体",
        "ORG.NAM": "机构",
        "ORG.NOM": "指代机构",
        "LOC.NOM": "指代地名",
        "GPE.NOM": "指代政体",
    }
}

from CC.pre_trained import NERPreTrainer
pre_trainer = NERPreTrainer(**args)

for i in pre_trainer(lr=1e-4):
    a = i
```
工作点2：
命名实体模型，在LEBERT模型的基础上，融入了外部知识词典，在BERT的第一层transformer中，外部知识特征通过简单的线性注意力融合
预训练过程中要将损失降低到1e-3
验证的时候需要将预训练模型替换成你自己的。

需要用到一个外部词典：THUOCL_FN_medical（一个是THUOCL的词库，还有一个是福能体检那边提供关键词）
disease.dic是LSTM-CRF-medical-master 用命名实体识别模型抽取的实体和用匹配词等七七八八的规则抽取
**Trainer LELoader**

```python
from CC.predicter import NERPredict
from CC.trainer import NERTrainer

# %%
args = {
    'num_epochs': 30,
    'num_gpus': [0],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/cdd_pre_3/Bert_10470/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 128,
    'max_scan_num': 1000000,
    'inter_max_scan_num': 39000,
    'train_file': './data/CDD/train_1000.json',
    'eval_file': './data/CDD/dev.json',
    'test_file': './data/CDD/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/CDD/labels.txt',
    'loader_name': 'le_loader_zl',
    'output_eval':True,
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "inter_knowledge_file":"./data/tencent/disease.dic",
    "default_tag":"O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'do_shuffle': True,
    "use_gpu": True,
    "debug": True,
    'model_name': 'ZLEBert',
    'classify':'lstm_crf',
    'task_name': 'cdd_disease_1000_1'
}
trainer = NERTrainer(**args)

for i in trainer():
    a = i
```

**Trainer CNLoader**

```python
from CC.trainer import NERTrainer
from CC.predicter import NERPredict

args = {
    'num_epochs': 30,
    'num_gpus': [0],
    'bert_config_file_name': './model/chinese_wwm_ext/bert_config.json',
    'pretrained_file_name': './model/chinese_wwm_ext/pytorch_model.bin',
    # 'pretrained_file_name': './save_pretrained/cdd_pre_3/Bert_10470/pytorch_model.bin',
    'hidden_dim': 300,
    'max_seq_length': 128,
    'max_scan_num': 1000000,
    'inter_max_scan_num': 39000,
    'train_file': './data/CDD/train_2000.json',
    'eval_file': './data/CDD/dev.json',
    'test_file': './data/CDD/test.json',
    'bert_vocab_file': './model/chinese_wwm_ext/vocab.txt',
    'tag_file': './data/CDD/labels.txt',
    'loader_name': 'le_loader_zl',
    # 'loader_name': 'le_loader',
    'output_eval':True,
    "word_embedding_file":"./data/tencent/word_embedding.txt",
    "word_vocab_file":"./data/tencent/tencent_vocab.txt",
    "inter_knowledge_file":"./data/tencent/disease.dic",
    "default_tag":"O",
    'batch_size': 8,
    'eval_batch_size': 64,
    'do_shuffle': True,
    "use_gpu": True,
    "debug": True,
    'model_name': 'ZLEBert',
    'classify':'lstm_crf',
    'task_name': 'cdd_disease_1'
}
trainer = NERTrainer(**args)

for i in trainer(lr2=1e-2):
    a = i
```

**Predictor**

```python
# %%
from predicter import NERPredict

# %%
predict = NERPredict(True,
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/news_tags_list.txt',
                     bert_model_path='./save_model/bert/cbef37de_bert.pth',
                     lstm_crf_model_path='./save_model/lstm_crf/cbef37de_lstm_crf.pth',
                     hidden_dim=150)

# %%
print(predict(["坐落于福州的福州大学ACM研究生团队, 在帅气幽默的傅仰耿老师带领下, 正在紧张刺激的开发一套全新的神秘系统。","在福州大学的后山, 驻扎着福大后山协会, 会长是陈学勤同志。"])[2:])


# %%
labels, text = predict(["福建省能源集团有限责任公司（以下简称集团）成立于2009年12月,是由福建省煤炭工业（集团）有限责任公司和福建省建材（控股）有限责任公司整合重组而成，系福建省属国有企业，2015年7月起并表福建石油化工集团有限责任公司。集团拥有全资及控股并表企业176家，在职员工2万余人，其中福能股份、福建水泥在主板上市，福能租赁、福能期货在新三板挂牌。集团注册资本金100亿元，资信等级连续多年保持AAA级别。集团连年列入中国企业500强。"])[2:]

# %%
labels, text = predict(["欢迎福能集团黄守清院长莅临福州大学ACM团队指导工作！"])[2:]

# %%
for idx, label in enumerate(labels):
    t = text[idx]
    for j, item in enumerate(label):
        print('{}\t{}'.format(t[j], item))

# %%
```

