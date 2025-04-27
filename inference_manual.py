import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from vlm_mner import BertLabelAttentionCRFNER, LabelEmbbeding, AuxiliaryEncoder, MaskRCNNGuidedEncoder, ImageFeatureExtractor, ImageFeatureEncoder, GraphFusion, TextEncoder
from transformers import BertTokenizerFast
from config import tag2idx, idx2tag

# --------—— 配置部分 ——--------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/home/dell3/zjh1/VLM-MNER/BestModel/Twitter2017/4_15/model.pt"                # 你的模型检查点
BERT_PATH = "./bert-base-cased"           # 和训练时一致
TOP_K = 5                               # 每张图选 top_k 区域框
FONT = ImageFont.load_default()
# -------------------------------

# 1) 初始化模型和 tokenizer
tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
aux_encoder = AuxiliaryEncoder(hidden_size=768)
model = BertLabelAttentionCRFNER(BERT_PATH, tag2idx, aux_encoder=aux_encoder)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    sd = checkpoint['model_state_dict']
else:
    sd = checkpoint
model.load_state_dict(sd)

model.to(DEVICE).eval()

# 2) 手动输入这一条样本的数据
raw_text = "At Oracle Arena, the home of the Golden State Warriors, Kyrie Irving is facing off against Stephen Curry."   # 你的原文本
aux_texts = [
    "person：The image depicts a basketball game in progress. Two players are actively engaged in the game, with one player wearing a white jersey with the number 30 and the name 'Curry' on the back, and the other player wearing a black jersey with the number 30. The player in the white jersey is attempting to block the player in the black jersey, who is in mid-air, holding the basketball. The background shows a crowd of spectators, many of whom are wearing yellow, indicating support for the team in white jerseys.\n"
    "location：The image depicts a basketball game in progress, with players from two teams actively engaged in the play. The crowd in the background is predominantly dressed in yellow, suggesting they are fans of the home team. The setting appears to be an indoor arena, likely a professional basketball venue given the large crowd and the intensity of the game. The players are wearing uniforms that indicate they are part of professional teams, with one team in white and the other in black. The scene captures a moment of action, with one player attempting a shot while the other defends.\n"
    "miscellaneous：The image does not contain information about the entity 'miscellaneous'. It depicts a basketball game with players in action, with a crowd of spectators in the background.\n"
    "organization：The image depicts a basketball game with players from two different teams. The player in white is from the Golden State Warriors, as indicated by the team's logo and colors. The player in black is from the Cleveland Cavaliers, as indicated by the team's logo and colors. The crowd in the background is predominantly wearing yellow, suggesting support for the Golden State Warriors. The scene is likely from a playoff game, given the intensity and the packed stadium."
]
image_paths = [
    "/home/dell3/zjh1/VLM-MNER/test_single/test_person.jpg",
    "/home/dell3/zjh1/VLM-MNER/test_single/test_location.jpg",
    "/home/dell3/zjh1/VLM-MNER/test_single/test_miscellaneous.jpg",
    "/home/dell3/zjh1/VLM-MNER/test_single/test_organization.jpg",
]

# 3) 准备输入：tokenize 原文本
tokens = tokenizer.tokenize(raw_text)
tokens = ["[CLS]"] + tokens + ["[SEP]"]
input_ids = tokenizer.convert_tokens_to_ids(tokens)
attention_mask = [1] * len(input_ids)
# pad 到最大长度
max_len = 128
pad_len = max_len - len(input_ids)
input_ids += [0]*pad_len
attention_mask += [0]*pad_len
input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=DEVICE)

# 4) 构造辅助文本 embeddings
aux_text = "\n".join(aux_texts)

text_segments_list = [aux_text]
label_embeddings = LabelEmbbeding(aux_texts, tokenizer=tokenizer, model=model, device=DEVICE)

# 5) 载入并 preprocess 四张图
preprocess = lambda img: TF.resize(img, (224,224))
imgs = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    imgs.append(TF.to_tensor(preprocess(img)))
# 变成 (1,4,3,224,224)
image_tensor = torch.stack(imgs, dim=0).unsqueeze(0).to(DEVICE)

# 6) 推理
with torch.no_grad():
    preds, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        label_embeddings=label_embeddings,
        labels=None,
        image_tensor=image_tensor
    )
# preds 是 [[idx1, idx2, ...]]
pred_tags = [idx2tag[i] for i in preds[0]]

# 7) 打印 token + 预测（合并子词，取第一个标注）

# 排除 Padding，只保留实际 token 的预测
tokens_clean = tokens[:len(pred_tags)]
pred_tags_clean = pred_tags[:len(tokens_clean)]

# 合并子词（## 开头的）并取第一个标注
merged_tokens = []
merged_labels = []

current_token = ""
current_label = ""
for tok, tag in zip(tokens_clean[1:-1], pred_tags_clean[1:-1]):  # 去掉 [CLS] 和 [SEP]
    if tok.startswith("##"):
        current_token += tok[2:]
    else:
        if current_token:
            merged_tokens.append(current_token)
            merged_labels.append(current_label)
        current_token = tok
        current_label = tag
# 处理最后一个 token
if current_token:
    merged_tokens.append(current_token)
    merged_labels.append(current_label)

# 打印控制台
print(f"{'Token':15s}Prediction")
print("-"*30)
for tok, tag in zip(merged_tokens, merged_labels):
    print(f"{tok:15s}{tag}")

# 保存为行格式
with open("predictions.txt", "w", encoding="utf-8") as f:
    f.write("Tokens:   " + " ".join(merged_tokens) + "\n")
    f.write("Labels:   " + " ".join(merged_labels) + "\n")


# 8) 可视化每张图被选的 top_k 区域
#    我们复用 MaskRCNNGuidedEncoder 内的 mask_rcnn 和 top_k 筛选逻辑
mrcnn_encoder = model.maskrcnn_image_encoder
maskrcnn = mrcnn_encoder.mask_rcnn
maskrcnn.to(DEVICE).eval()

# 对 4 张图平铺调用一次 maskrcnn
flat = image_tensor.view(4,3,224,224)
with torch.no_grad():
    dets = maskrcnn(flat)

# 依次对每张图，画出 scores 排名前 TOP_K 的 boxes 并保存
for i, det in enumerate(dets):
    img_path = image_paths[i]
    img = preprocess(Image.open(img_path).convert("RGB"))
    draw = ImageDraw.Draw(img)
    scores = det["scores"].cpu()
    boxes = det["boxes"].cpu()
    # 取 top_k highest-score 区域
    topk = min(TOP_K, scores.size(0))
    idxs = scores.topk(topk).indices
    for idx in idxs:
        x1, y1, x2, y2 = boxes[idx]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{scores[idx]:.2f}", font=FONT, fill="red")
    # 保存图像
    img.save(f"image_{i}_top_{TOP_K}_regions.jpg")
