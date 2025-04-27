import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from torchcrf import CRF
import time
import random
import argparse
from tqdm import tqdm
import warnings
import os
import glob
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.nn import RGCNConv
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.nn.functional as F
import torchvision.models as models
from metric import evaluate, evaluate_each_class 
from config import tag2idx, idx2tag, max_len, log_fre
from dataset import MMNerDataset
from dataset import collate_fn
from torch.utils.data import DataLoader
import logging
import datetime
import os

def init_logger(mode):
    """
    初始化日志记录器。
    mode: 'train' 或 'test'
    日志将保存在当前目录下的 log/train 或 log/test 文件夹中，
    文件名为当前时间，精确到秒，例如 "20230415_153045.txt"
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.getcwd(), "log", mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{now}.txt")
    
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)
    # 清空旧的 handler
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件 handler
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台 handler（可选）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

torch.cuda.empty_cache()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义模型
###################################################################################################################################
#-----------------辅助文本编码模块------------------------#
def LabelEmbbeding(text_segments_list, tokenizer, model, device):
    """
    处理一批 text_segments 数据，生成标签向量。
    输入：
        text_segments_list: list，每个元素是一个样本的字符串，
            其中包含4行文本，每行对应一个类别，行内使用冒号分割，
    输出：
        标签向量，形状为 [batch_size, 4, hidden_size]
    """
    batch_embeddings = []
    ordered_labels = ["person", "location", "miscellaneous", "organization"]
    
    # 遍历 batch 中每个样本
    for text in text_segments_list:
        # 对单个样本字符串按换行符分割成行（每行对应一个类别描述）
        lines = text.split('\n')
        label_embeddings_list = []
        for seg in lines:
            seg = seg.strip()
            if not seg:
                continue
            try:
                label_word, sentence = seg.split('：', 1)
            except Exception as e:
                print("文本格式错误，请检查冒号是否正确，出错内容：", seg, e)
                continue
            label_word = label_word.strip()
            sentence = sentence.strip()
            if label_word in ordered_labels:
                # 对该段文本进行编码，取 [CLS] 表示作为标签向量
                inputs = tokenizer(seg, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.eval()
                with torch.no_grad():
                    outputs = model.bert(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden_size]
                label_embeddings_list.append(cls_embedding)
            else:
                print(f"未在预设标签中找到：{label_word}")
        if len(label_embeddings_list) != 4:
            raise ValueError(f"标签数量不足4个，实际获得：{len(label_embeddings_list)}，请检查样本文本：{text}")
        # 将四个标签向量堆叠成 [1, 4, hidden_size]
        sample_embeddings = torch.stack(label_embeddings_list, dim=1)
        batch_embeddings.append(sample_embeddings)
    # 将所有样本的结果拼接，得到 shape [batch_size, 4, hidden_size]
    batch_embeddings = torch.cat(batch_embeddings, dim=0).to(device)
    return batch_embeddings

class AuxiliaryEncoder(nn.Module):
    """
    使用多层 GAT + FFN + 残差结构对标签嵌入建模。
    输入: [batch_size, 4, hidden_size]
    输出: [batch_size, 4, hidden_size]
    """
    def __init__(self, hidden_size, heads=4, dropout=0.3, num_layers=3, use_label_type_embed=True):
        super(AuxiliaryEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_label_type_embed = use_label_type_embed

        # 标签类型嵌入
        if self.use_label_type_embed:
            self.label_type_embed = nn.Embedding(4, hidden_size)

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            ))
        # for _ in range(num_layers):
        # # 这里输入和输出均是 hidden_size，且设 num_relations 为 1（或者根据需求设置多个关系类型）
        #     self.gat_layers.append(RGCNConv(hidden_size, hidden_size, num_relations=1))
        #     self.norm_layers.append(nn.LayerNorm(hidden_size))
        #     self.ffn_layers.append(nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size * 2),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(hidden_size * 2, hidden_size),
        #     ))

        # 最后添加一个BiLSTM 处理4个label间顺序信息（类似 Transformer）
        self.use_lstm = False   # 消融实验可移除项
        if self.use_lstm:
            self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, label_embeddings):
        # label_embeddings: [B, 4, H]
        B, N, H = label_embeddings.size()
        output_list = []

        for b in range(B):
            x = label_embeddings[b]  # [4, H]

            if self.use_label_type_embed:
                type_ids = torch.arange(N, device=x.device)  # 0,1,2,3
                type_embed = self.label_type_embed(type_ids)  # [4, H]
                x = x + type_embed

            # 构建完全图（每个节点连接其他所有节点）
            idx = torch.arange(N, device=x.device)
            row = idx.repeat_interleave(N)
            col = idx.repeat(N)
            edge_index = torch.stack([row, col], dim=0)
            # edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

            # 多层 GAT + FFN + 残差结构
            for l in range(self.num_layers):
                x_res = x
                x = self.gat_layers[l](x, edge_index)
                x = self.norm_layers[l](x + x_res)

                ffn_res = x
                x = self.ffn_layers[l](x)
                x = self.norm_layers[l](x + ffn_res)

            output_list.append(x)

        output = torch.stack(output_list, dim=0)  # [B, 4, H]

        if self.use_lstm:
            lstm_out, _ = self.bilstm(output)  # [B, 4, H]
            output = self.proj(lstm_out) + output  # 残差加和

        return output  # [B, 4, H]


#-------------------图像特征提取模块------=-------------------#
class ImageFeatureExtractor(nn.Module):
    """
    图像特征提取模块，使用 ResNet50 提取图像特征。
    """
    def __init__(self, hidden_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # 去除最后一层FC，保留 avgpool 输出
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(2048, hidden_size)
    
    def forward(self, x):
        B, num_images, C, H, W = x.size()
        x = x.view(B * num_images, C, H, W)
        features = self.feature_extractor(x)  # [B*num_images, 2048, 1, 1]
        features = features.view(B * num_images, -1)
        features = self.proj(features)
        features = features.view(B, num_images, -1)
        return features

# 类别映射
entity2idx = {"person": 0, "location": 1, "miscellaneous": 2, "organization": 3}
mrcnn2entity = {1: "person", 2: "location", 3: "miscellaneous", 4: "organization"}

class MaskRCNNGuidedEncoder(nn.Module):
    """
    使用 Mask R-CNN 提取图像区域特征，并与辅助文本特征对齐。
    修改后：输入形状 [B, 4, 3, 244, 244]，输出形状 [B, 4, hidden_size]
    """
    def __init__(self, hidden_size, top_k=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True)
        self.mask_rcnn.eval()
        self.mask_rcnn.to('cpu') 

        # 图像区域特征提取器
        self.feature_extractor = ImageFeatureExtractor(hidden_size)
        self.sim_proj = nn.Linear(hidden_size, hidden_size)
        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, images, aux_text_features):
        """
        :param images: [B, 4, 3, 244, 244] 原始图像，即每个样本有4张图像
        :param aux_text_features: [B, 4, hidden_size] 辅助文本向量，用于指导匹配
        输出: [B, 4, hidden_size] 每个实体类别的图像区域融合特征
        """
        B, N, C, H, W = images.size()  # N=4
        device = images.device
        images_flat = images.view(B * N, C, H, W)  # [B*4, 3, 244, 244]
        with torch.no_grad():
            detection_results = self.mask_rcnn(images_flat)  # List[Dict] 长度为 B*4
        detection_results = [detection_results[i * N:(i + 1) * N] for i in range(B)]
        
        aggregated_features = []
        # 对于每个样本，按类别收集候选区域特征
        for b in range(B):
            cand_dict = {cat: [] for cat in range(4)}
            # 遍历该样本中的4张图像
            for n in range(N):
                result = detection_results[b][n]
                # 对每个候选区域（遍历所有 boxes）
                for i in range(len(result['boxes'])):
                    label_id = result['labels'][i].item()
                    label_str = mrcnn2entity.get(label_id, None)
                    if label_str is None:
                        continue
                    cat = entity2idx.get(label_str, None)
                    if cat is None:
                        continue
                    box = result['boxes'][i]
                    # 裁剪图像区域
                    crop = torchvision.transforms.functional.crop(
                        images[b, n], 
                        int(box[1]), int(box[0]),
                        int(box[3] - box[1]), int(box[2] - box[0])
                    )
                    # 调整大小为 224x224
                    crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                    feat = self.feature_extractor(crop.unsqueeze(1)).detach() # [1, 1, hidden_size]
                    feat = feat.squeeze(0).squeeze(0)  # [hidden_size]
                    cand_dict[cat].append(feat.unsqueeze(0).detach())
            # 对于每个类别，对所有候选区域进行加权聚合
            sample_feats = []
            for cat in range(4):
                if len(cand_dict[cat]) == 0:
                    # 如果没有候选区域，使用零向量代替
                    agg_feat = torch.zeros(1, self.hidden_size, device=device)
                else:
                    feats_cat = torch.cat(cand_dict[cat], dim=0)
                    # 计算辅助文本特征与候选区域特征之间的余弦相似度
                    sim = F.cosine_similarity(self.sim_proj(feats_cat), aux_text_features[b, cat].unsqueeze(0), dim=-1)
                    # 根据相似度构造 soft attention 权重
                    attn = F.softmax(sim, dim=0).unsqueeze(-1)  # [n, 1]
                    agg_feat = torch.sum(attn * feats_cat, dim=0, keepdim=True)  # [1, hidden_size]
                sample_feats.append(agg_feat)
            sample_feats = torch.cat(sample_feats, dim=0)  # [4, hidden_size]
            aggregated_features.append(sample_feats)
        aggregated_features = torch.stack(aggregated_features, dim=0)  # [B, 4, hidden_size]
        return self.out_proj(aggregated_features)

class ImageFeatureEncoder(nn.Module):
    """
    对图像特征进行图建模（每个样本4个图像：person, location, misc, org）。
    输入: [B, 4, H]
    输出: [B, 4, H]
    """
    def __init__(self, hidden_size, heads=4, dropout=0.3, num_layers=3, use_type_embedding=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_type_embedding = use_type_embedding

        if use_type_embedding:
            self.type_embedding = nn.Embedding(4, hidden_size)  # person, location, misc, org

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            ))

    def forward(self, image_features):
        # image_features: [B, 4, H]
        B, N, H = image_features.size()
        outputs = []

        for b in range(B):
            x = image_features[b]  # [4, H]

            if self.use_type_embedding:
                type_ids = torch.arange(N, device=x.device)
                type_embed = self.type_embedding(type_ids)
                x = x + type_embed

            # 构建完全图（每个图像特征相互连接）
            row = torch.arange(N, device=x.device).repeat_interleave(N)
            col = torch.arange(N, device=x.device).repeat(N)
            edge_index = torch.stack([row, col], dim=0)

            # 多层 GAT + FFN + 残差
            for l in range(self.num_layers):
                x_res = x
                x = self.gat_layers[l](x, edge_index)
                x = self.norm_layers[l](x + x_res)

                ffn_res = x
                x = self.ffn_layers[l](x)
                x = self.norm_layers[l](x + ffn_res)

            outputs.append(x)

        return torch.stack(outputs, dim=0)  # [B, 4, H]

#----------------------图融合模块----------------------#

class ResidualLayer(nn.Module):
    """
    图注意力层，带有残差连接和层归一化。
    输入: [B, T+L+I, H]
    输出: [B, T+L+I, H]
    """
    def __init__(self, hidden_size, heads=4, dropout=0.3):
        super().__init__()
        self.gat = GATConv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        residual = x
        out = self.gat(x, edge_index)
        out = self.activation(out)
        out = self.dropout(out)
        return self.norm(out + residual)

class GraphFusion(nn.Module):
    def __init__(self, hidden_size, num_layers=3, dropout=0.3, top_k=3, heads=4, 
                 connect_image_to_label=True, connect_image_to_image=True, connect_label_to_label=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.gat_layers = nn.ModuleList([
            ResidualLayer(hidden_size, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.connect_image_to_label = connect_image_to_label      #消融实验可移除项
        self.connect_image_to_image = connect_image_to_image      #消融实验可移除项
        self.connect_label_to_label = connect_label_to_label      #消融实验可移除项

    def forward(self, text_repr, label_repr, mask=None, image_repr=None):
        """
        text_repr: [B, T, H]
        label_repr: [B, L=4, H]
        image_repr: [B, I=4, H] or None
        return: [B, T, H]
        """
        B, T, H = text_repr.size()
        L = label_repr.size(1)
        I = image_repr.size(1) if image_repr is not None else 0
        outputs = []

        for b in range(B):
            text_feat = text_repr[b]   # [T, H]
            label_feat = label_repr[b] # [4, H]
            if image_repr is not None:
                image_feat = image_repr[b]  # [4, H]
                x = torch.cat([text_feat, label_feat, image_feat], dim=0)  # [T+L+I, H]
            else:
                x = torch.cat([text_feat, label_feat], dim=0)  # [T+L, H]

            N = x.size(0)
            edge_index = []

            # --- Text-to-Text 邻接连接 ---
            for i in range(T - 1):
                edge_index += [[i, i + 1], [i + 1, i]]

            # --- Text-to-Label 相似连接 ---
            sim_label = F.cosine_similarity(
                text_feat.unsqueeze(1), label_feat.unsqueeze(0), dim=-1
            )  # [T, 4]
            topk_vals, topk_idx = torch.topk(sim_label, self.top_k, dim=-1)
            for i in range(T):
                for j in topk_idx[i]:
                    edge_index.append([i, T + j.item()])  # text->label
                    edge_index.append([T + j.item(), i])  # label->text

            if image_repr is not None:
                # --- Text-to-Image 相似连接 ---
                sim_image = F.cosine_similarity(
                    text_feat.unsqueeze(1), image_feat.unsqueeze(0), dim=-1
                )  # [T, 4]
                topk_vals_img, topk_idx_img = torch.topk(sim_image, self.top_k, dim=-1)
                for i in range(T):
                    for j in topk_idx_img[i]:
                        edge_index.append([i, T + L + j.item()])       # text -> image
                        edge_index.append([T + L + j.item(), i])       # image -> text

                # --- Image-to-Label 全连接---
                if self.connect_image_to_label:
                    for i in range(I):
                        for j in range(L):
                            edge_index.append([T + L + i, T + j])
                            edge_index.append([T + j, T + L + i])

                # --- Image-to-Image 全连接 ---            
                if self.connect_image_to_image:
                    for i in range(I):
                        for j in range(I):
                            edge_index.append([T + L + i, T + L + j])

            # --- Label-to-Label 全连接 ---
            if self.connect_label_to_label:
                for i in range(L):
                    for j in range(L):
                        edge_index.append([T + i, T + j])

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device).t().contiguous()

            # GAT 层处理
            h = x
            for layer in self.gat_layers:
                h = layer(h, edge_index)

            outputs.append(h[:T])  # 只返回文本部分的融合特征

        return torch.stack(outputs, dim=0)  # [B, T, H]

#-------------------文本处理模块--------------------#
class TextEncoder(nn.Module):
    """
    使用 GAT 编码 BERT token 表示，使用滑动窗口构图以节省显存。
    输出维度不变：[B, T, hidden_size]
    """
    def __init__(self, hidden_size, num_layers=3, heads=4, dropout=0.3, window_size=2):
        super().__init__()
        self.num_layers = num_layers
        self.window_size = window_size
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_size, hidden_size, heads=heads, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_size))

    def forward(self, x):
        # x: [B, T, H]
        B, T, H = x.size()
        outputs = []

        for b in range(B):
            node_features = x[b]  # [T, H]
            edges = []

            # 使用滑动窗口方式构建稀疏图（只连临近token）
            for i in range(T):
                for j in range(i + 1, min(i + self.window_size + 1, T)):
                    edges.append([i, j])
                    edges.append([j, i])

            # 构建全连接图
            # for i in range(T):
            #     for j in range(T):
            #         if i != j:
            #             edges.append([i, j])

            edge_index = torch.tensor(edges, dtype=torch.long, device=node_features.device).t()

            for l in range(self.num_layers):
                res = node_features
                node_features = self.gat_layers[l](node_features, edge_index)  # [T, H]
                node_features = self.dropout(node_features)
                node_features = self.norms[l](node_features + res)

            outputs.append(node_features)

        return torch.stack(outputs, dim=0)  # [B, T, H]
    
# class TextEncoder(nn.Module):
#     """
#     对 BERT 输出的 token 表示进行图卷积编码，这里采用关系型图卷积网络（R-GCN）对每个样本构造的全连通图进行处理，
#     输出形状不变：[B, T, hidden_size]。其中这里简化起见，所有边都赋予相同关系类型（0）。
#     """
#     def __init__(self, hidden_size, num_layers=2, dropout=0.3, num_relations=1):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             RGCNConv(hidden_size, hidden_size, num_relations=num_relations)
#             for _ in range(num_layers)
#         ])
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         # x: [B, T, hidden_size]
#         B, T, H = x.size()
#         outs = []
#         for b in range(B):
#             node_features = x[b]  # [T, hidden_size]
#             # 构造全连通图：每个节点之间都有边（不包括自环），然后扩充为双向边
#             edge_index = torch.combinations(torch.arange(T, device=node_features.device), r=2)
#             edge_index = edge_index.t()  # shape [2, num_edges] with one direction
#             # 构造双向边
#             edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
#             # 所有边赋予同一关系类别 0
#             edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=node_features.device)
            
#             # RGCN 逐层处理
#             for layer in self.layers:
#                 node_features = layer(node_features, edge_index, edge_type)
#                 node_features = self.dropout(node_features)
#             outs.append(node_features)
#         return torch.stack(outs, dim=0)  # [B, T, hidden_size]


#----------------------主模块----------------------#
class BertLabelAttentionCRFNER(nn.Module):
    """
    <图像-辅助文本-文本>图建模 + CRF 模型
    """
    def __init__(self, bert_model_path, tag2idx, aux_encoder=None):
        super(BertLabelAttentionCRFNER, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.hidden2tag = nn.Linear(hidden_size, len(tag2idx))
        # CRF层，batch_first=True 表示输入形状为 (batch, seq_len, feat_dim)
        self.crf = CRF(num_tags=len(tag2idx), batch_first=True)
        self.fc_layer = nn.Linear(hidden_size, hidden_size)
        self.graph_fusion = GraphFusion(hidden_size)
        self.text_encoder = TextEncoder(hidden_size, num_layers=2, dropout=0.3)
        self.image_encoder = ImageFeatureExtractor(hidden_size) # 图像编码器模块
        self.image_encoder_gat = ImageFeatureEncoder(hidden_size)
        self.maskrcnn_image_encoder = MaskRCNNGuidedEncoder(hidden_size) # 图像编码器模块
        # 辅助文本编码器模块
        if aux_encoder is None:
            self.aux_encoder = nn.Identity()   #消融实验可移除项
        else:
            self.aux_encoder = aux_encoder


    def forward(self, input_ids, attention_mask, label_embeddings, labels=None, image_tensor=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.text_encoder(sequence_output)  # [B, T, H]

        enhanced_aux = self.aux_encoder(label_embeddings)

        attn_scores = torch.matmul(sequence_output, enhanced_aux.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        label_context = torch.matmul(attn_weights, enhanced_aux)

        if image_tensor is not None:
            image_features = self.maskrcnn_image_encoder(image_tensor, enhanced_aux) # [B, 4, hidden_size]
            # image_features = self.image_encoder_gat(image_features)  # [B, 4, hidden_size]  #消融实验可移除项
        else:
            image_features = None

        enhanced_output = self.graph_fusion(sequence_output, label_context, attention_mask, image_features)
        num_layers = 3
        for _ in range(num_layers):
            # 交互层可以采用一个全连接层再加激活，再加残差连接
            temp = self.dropout(torch.relu(self.fc_layer(enhanced_output)))
            enhanced_output = enhanced_output + temp                           #消融实验可移除项

        emissions = self.hidden2tag(enhanced_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss, emissions
        else:
            pred = self.crf.decode(emissions, mask=attention_mask.bool())
            return pred, emissions

###############################################################################

bert_model_path = "./bert-base-cased"
aux_encoder = AuxiliaryEncoder(hidden_size=768)
model = BertLabelAttentionCRFNER(bert_model_path, tag2idx, aux_encoder=aux_encoder)

# 加载 tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)

def save_model(model, optimizer, scheduler, epoch, path="./model.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)
    print("Checkpoint saved at epoch", epoch)


def predict(model, dataloader, dataset, mode="val", logger=None):
    """
    对给定 dataloader 进行预测，并返回预测结果（token级别的预测索引列表）。
    """
    model.eval()
    all_preds = []
    all_gold = []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting " + mode):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold = batch["label_ids"].to(device)
            text_segments = batch["text_segments"]
            label_embeddings = LabelEmbbeding(text_segments, tokenizer=tokenizer, model=model, device=device)
            image_tensor = batch["image_tensor"].to(device)
            loss, preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_embeddings=label_embeddings,
                labels=gold,
                image_tensor=image_tensor
            )
            total_loss += loss.item()
            count += 1
            preds, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_embeddings=label_embeddings,
                labels=None,
                image_tensor=image_tensor
            )
            # preds 为列表，每个元素为该批次某样本的 token 级预测
            all_preds.extend(preds)
            all_gold.extend(gold.cpu().tolist())
    arr = ["PER", "LOC", "ORG", "OTHER"]
    dict_class_P = {}
    dict_class_R = {}
    for class_type in arr:
        class_f1, class_p, class_r = evaluate_each_class(all_preds, all_gold, tag2idx, class_type)
        msg = "{}: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(class_type, class_p, class_r, class_f1)
        dict_class_P[class_type] = class_p
        dict_class_R[class_type] = class_r
        logger.info(msg)
    precision, recall, f1 = evaluate(dict_class_P, dict_class_R, dataset, mode)
    log_str = "{} set evaluation : Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(mode, precision, recall, f1)
    logger.info(log_str)
    logger.info("\n")
        
    avg_val_loss = total_loss / count
    return f1, avg_val_loss, dict_class_P, dict_class_R

import torch.backends.cudnn as cudnn
cudnn.enabled = False
import csv

def train(args):
    train_dataset = MMNerDataset(args.data, args.segments, args.images, split="train", max_length=max_len)
    valid_dataset = MMNerDataset(args.data, args.segments, args.images, split="valid", max_length=max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger = init_logger("train")
    logger.info("开始训练")

    torch.manual_seed(args.seed)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    best_f1 = 0.0
    best_epoch = -1
    start_time = time.time()

    start_epoch = 0
    # 断点恢复训练
    if args.resume and os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    # 创建 CSV 文件保存路径
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "train_metrics.csv")

    write_header = True
    if args.resume and os.path.exists(csv_path):
        write_mode = "a"   # 追加写入
        write_header = False  # 不要重复写入表头
    else:
        write_mode = "w"   # 新建文件

    with open(csv_path, mode=write_mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        if write_header:
            header = [
                "epoch", "train_loss", "val_loss", "val_f1",
                "PER_precision", "PER_recall", "PER_f1",
                "LOC_precision", "LOC_recall", "LOC_f1",
                "ORG_precision", "ORG_recall", "ORG_f1",
                "OTHER_precision", "OTHER_recall", "OTHER_f1"
            ]
            writer.writerow(header)

        for epoch in range(start_epoch, args.epoch):
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label_ids = batch["label_ids"].to(device)
                text_segments = batch["text_segments"]
                label_embeddings = LabelEmbbeding(text_segments, tokenizer=tokenizer, model=model, device=device)
                image_tensor = batch["image_tensor"].to(device)

                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label_embeddings=label_embeddings,
                    labels=label_ids,
                    image_tensor=image_tensor
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                epoch_loss += loss.item()

                if (train_loader.batch_size > 0) and (train_loader.batch_size != 0) and (train_loader.batch_size and train_loader.batch_size % log_fre == 0):
                    print(f"Epoch: {epoch} Loss: {loss.item():.4f}")

            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} average train loss: {avg_train_loss:.4f}")

            # 验证阶段
            val_f1, val_loss, val_precisions, val_recalls = predict(
                model, valid_loader, args.data, mode="val", logger=logger
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                save_model(model, optimizer, scheduler, epoch, args.ckpt_path)

            print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")

            # 记录每一类的 P, R, F1
            row = [epoch, avg_train_loss, val_loss, val_f1]
            for cls in ["PER", "LOC", "ORG", "OTHER"]:
                p = val_precisions.get(cls, 0.0)
                r = val_recalls.get(cls, 0.0)
                f1_cls = 2 * p * r / (p + r + 1e-8) if (p + r) > 0 else 0.0
                row.extend([p, r, f1_cls])
            writer.writerow(row)
            csvfile.flush()

    logger.info(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")
    
def test(args):
    test_dataset  = MMNerDataset(args.data, args.segments, args.images, split="test", max_length=max_len)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logger = init_logger("test")
    logger.info("开始测试")
    # 加载最佳模型
    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(args.ckpt_path, map_location=map_location), strict=False)
    model.to(device)
    print("Evaluating on test set...")
    test_f1, test_loss, _, _= predict(model, test_loader, args.data, mode="test", logger=logger)
    print("Test set F1: {:.4f}, Loss: {:.4f}".format(test_f1, test_loss))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true', help="Run training.")
    parser.add_argument("--do_test", action='store_true', help="Run testing.")
    parser.add_argument("--resume", action='store_true', help="Resume training from checkpoint.")
    parser.add_argument("--data",type=str ,default="./my_data/twitter2015", help="Run prediction.")
    parser.add_argument("--segments",type=str ,default="./data/data_I2T_2015", help="Run prediction.")
    parser.add_argument("--images",type=str ,default="./data/twitter2015/image", help="Run prediction.")
    parser.add_argument("--epoch", type=int, default=30, help="Total training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for Adam.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--ckpt_path", type=str, default="./model.pt", help="Path to save or load model.")
    parser.add_argument("--output_dir", default="./train_metric", help="Path to save csv file.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for initialization.")
    args = parser.parse_args()
    
    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)
    else:
        raise ValueError("At least one of do_train or do_test must be True.")

if __name__ == "__main__":
    main()