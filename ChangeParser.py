import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F
import random
import numpy as np
import json
import os
from tqdm import tqdm  # 直接导入 tqdm

# === 自定义导入 ===
from vlm_mner import (
    BertLabelAttentionCRFNER,
    AuxiliaryEncoder,
    GraphFusion,
    LabelEmbbeding,
    predict,
)
from dataset import MMNerDataset, collate_fn
from config import tag2idx, max_len
import logging
logger = logging.getLogger("optuna")
logger.setLevel(logging.INFO)
# 例如添加一个控制台 handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# === 基础设置 ===
bert_model_path = "./bert-base-cased"
data_path = "./my_data/twitter2015"
image_path = "./data/data_I2T_2015"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def objective(trial):
    # === 超参数搜索空间 ===
    seed = trial.suggest_int('seed', 42, 1000)
    set_seed(seed)

    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    aux_num_layers = trial.suggest_int('aux_num_layers', 2, 6)
    aux_dropout = trial.suggest_float('aux_dropout', 0.3, 0.7)
    top_k = trial.suggest_int('top_k', 2, 5)
    fusion_num_layers = trial.suggest_int('fusion_num_layers', 1, 3)
    fusion_heads = trial.suggest_categorical('fusion_heads', [2, 4, 8])
    aux_heads = trial.suggest_categorical('aux_heads', [2, 4, 8])

    # === Tokenizer & Dataset ===
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
    train_dataset = MMNerDataset(data_path, image_path, split="train", max_length=max_len)
    valid_dataset = MMNerDataset(data_path, image_path, split="valid", max_length=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # === 模型构建 ===
    aux_encoder = AuxiliaryEncoder(
        hidden_size=768,
        heads=aux_heads,
        dropout=aux_dropout,
        num_layers=aux_num_layers,
        use_label_type_embed=True
    )

    model = BertLabelAttentionCRFNER(bert_model_path, tag2idx, aux_encoder=aux_encoder)
    model.graph_fusion = GraphFusion(
        hidden_size=768,
        num_layers=fusion_num_layers,
        dropout=dropout_rate,
        top_k=top_k,
        heads=fusion_heads
    )
    model.dropout = nn.Dropout(dropout_rate)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    # === 训练 ===
    best_f1 = 0.0
    epochs = 10
    for epoch in range(epochs):
        model.train()
        # 使用 tqdm 显示训练 batch 的进度
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label_ids"].to(device)
            text_segments = batch["text_segments"]

            label_embeddings = LabelEmbbeding(text_segments, tokenizer=tokenizer, model=model, device=device)
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_embeddings=label_embeddings,
                labels=label_ids
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        scheduler.step()

        val_f1 = predict(model, valid_loader, data_path, mode="val", logger=logger)
        print(f"[Epoch {epoch}] Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # 保存当前最优模型参数
            trial.set_user_attr("best_model_state_dict", model.state_dict())

    return best_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    # 使用 tqdm 显示调参过程的进度条
    n_trials = 20
    with tqdm(total=n_trials, desc="Optimizing", position=0) as pbar:
        def update_progress(study, trial):
            pbar.update(1)  # 每进行一个 trial，进度条更新
        
        study.optimize(objective, n_trials=n_trials, callbacks=[update_progress])

    best_trial = study.best_trial
    print("\n===== Best Trial Result =====")
    print(f"Best Val F1: {best_trial.value:.4f}")
    print("Best Hyperparameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val}")

    # === 保存模型和参数 ===
    os.makedirs("optuna_result", exist_ok=True)
    torch.save(best_trial.user_attrs["best_model_state_dict"], "optuna_result/best_model.pt")
    with open("optuna_result/best_config.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)

    print("\nBest model and config saved to `optuna_result/`.")
