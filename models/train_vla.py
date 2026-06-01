import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import timm
from torch.nn.parallel import DistributedDataParallel as DDP

from nuscenes.nuscenes import NuScenes
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ImageFile.LOAD_TRUNCATED_IMAGES = True

CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
]

# =========================================================
# 1. Q-Former Style Cross Attention Block
# =========================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)


# =========================================================
# 2. VLA Architecture: Multi-Camera Prefix VLA
# =========================================================
class MultiCameraPrefixVLA(nn.Module):
    def __init__(self, num_action_classes, model_type="light_drive", language_model_name="Qwen/Qwen2-0.5B-Instruct", prefix_len=32):
        super().__init__()
        self.model_type = model_type
        self.prefix_len = prefix_len
        self.num_action_classes = num_action_classes

        # [Vision Encoder]
        if model_type == "light_drive":
            self.vision_encoder = timm.create_model("deit_small_patch16_224", pretrained=True)
            self.vision_dim = 384
        elif model_type == "baseline_a":
            self.vision_encoder = timm.create_model("resnet101", pretrained=True, num_classes=0, global_pool="")
            self.vision_dim = 2048
        else:
            self.vision_encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.vision_dim = 768

        # [Language Model]
        self.model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, torch_dtype=self.model_dtype
        )
        lm_dim = self.language_model.config.hidden_size

        # [Projections & Q-Former]
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

        if self.model_type == "light_drive":
            self.num_patches = 196
            self.patch_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, lm_dim) * 0.02)
            
            self.camera_encoder = nn.Sequential(
                nn.Linear(16, 128), nn.GELU(),
                nn.Linear(128, lm_dim), nn.GELU(),
                nn.Linear(lm_dim, lm_dim * 2), 
            )
            nn.init.zeros_(self.camera_encoder[-1].weight)
            nn.init.zeros_(self.camera_encoder[-1].bias)
            self.camera_embedding = nn.Parameter(torch.randn(len(CAMERA_NAMES), lm_dim))

        self.query_tokens = nn.Parameter(torch.randn(1, prefix_len, lm_dim) * 0.02)
        self.qformer_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=lm_dim, num_heads=8) for _ in range(4)
        ])
        self.prefix_norm = nn.LayerNorm(lm_dim)
        
        # 🚀 [VLA 추가] Action & Speed 예측을 위한 출력 Head
        self.action_head = nn.Linear(lm_dim, num_action_classes)
        self.speed_head = nn.Sequential(
            nn.Linear(lm_dim, lm_dim // 2),
            nn.ReLU(),
            nn.Linear(lm_dim // 2, 1)
        )
        
        self._freeze_backbones()

    def _freeze_backbones(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
        print("[*] Frozen Vision Encoder & LLM. Training Q-Former, Projectors, and Action/Speed Heads.")

    def extract_patch_tokens(self, images):
        if self.model_type == "baseline_a":
            features = self.vision_encoder(images) 
            return features.flatten(2).transpose(1, 2) 
        else:
            features = self.vision_encoder.forward_features(images)
            return features[:, 1:, :] 

    def encode_prefix(self, images, camera_matrices):
        B, N, C, H, W = images.shape
        patch_tokens = self.extract_patch_tokens(images.view(B * N, C, H, W))
        patch_tokens = self.vision_proj(patch_tokens)
        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, N, num_patches, -1)

        if self.model_type == "light_drive":
            patch_tokens = patch_tokens + self.patch_pos_embed
            cam_embed = self.camera_embedding.unsqueeze(0).unsqueeze(2)
            patch_tokens = patch_tokens + cam_embed

            geom_params = self.camera_encoder(camera_matrices) 
            gamma, beta = geom_params.chunk(2, dim=-1)         
            
            gamma = gamma.unsqueeze(2) 
            beta = beta.unsqueeze(2)
            patch_tokens = patch_tokens * (1.0 + gamma) + beta

        patch_tokens = patch_tokens.view(B * N, num_patches, -1)
        queries = self.query_tokens.expand(B * N, -1, -1)
        
        for blk in self.qformer_blocks:
            queries = blk(query=queries, key_value=patch_tokens)
            
        queries = queries.view(B, N * self.prefix_len, -1)
        return self.prefix_norm(queries).to(self.language_model.dtype)

    def forward(self, images, camera_matrices, input_ids, labels, attention_mask, action_labels=None, target_speeds=None):
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        
        prefix_mask = torch.ones((images.shape[0], prefix_embeds.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        prefix_labels = torch.full((images.shape[0], prefix_embeds.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        
        # 🚀 [VLA 추가] output_hidden_states=True를 통해 마지막 은닉층 추출
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=True, 
            return_dict=True
        )
        
        lm_loss = outputs.loss
        
        # 🚀 [VLA 추가] 마지막 토큰의 Hidden State 가져오기 (배치 내 시퀀스 길이 고려)
        hidden_states = outputs.hidden_states[-1] # (B, Seq_Len, Dim)
        sequence_lengths = full_attention_mask.sum(dim=1) - 1 # 패딩 제외 마지막 토큰 인덱스
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_idx, sequence_lengths] # (B, Dim)
        
        # Action Classification (Driving Decision)
        action_logits = self.action_head(last_token_hidden)
        
        # Speed Regression
        speed_preds = self.speed_head(last_token_hidden).squeeze(-1)
        
        if action_labels is not None and target_speeds is not None:
            # Action은 CrossEntropy, Speed는 MSE Loss 적용
            action_loss = F.cross_entropy(action_logits, action_labels)
            speed_loss = F.mse_loss(speed_preds, target_speeds)
            
            # Loss 가중치 조합 (속도 Loss의 스케일을 맞추기 위해 0.1 부여)
            total_loss = lm_loss + action_loss + (0.1 * speed_loss)
            return total_loss, lm_loss, action_loss, speed_loss
            
        return action_logits, speed_preds


# =========================================================
# 3. Dataset & DataLoader (VLA 지원으로 확장)
# =========================================================
class NuScenesVLADataset(Dataset):
    def __init__(self, nusc_root, label_path, tokenizer, transform, tokens=None, action2id=None):
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=nusc_root, verbose=False)
        self.tokenizer = tokenizer
        self.transform = transform
        
        with open(label_path, "r", encoding="utf-8") as f:
            self.label_data = json.load(f)
            
        if tokens is None:
            self.tokens = list(self.label_data.keys())
        else:
            self.tokens = [t for t in tokens if t in self.label_data]
            
        # 🚀 [VLA 추가] JSON 전체를 훑어서 고유 Action(Driving Decision) 클래스 자동 추출
        if action2id is None:
            self._build_action_vocab()
        else:
            self.action2id = action2id
            
        print(f"[*] Loaded {len(self.tokens)} samples for training.")

    def _build_action_vocab(self):
        actions = set()
        for k, v in self.label_data.items():
            reasoning = v.get("global_reasoning", "")
            if "### Driving Decision" in reasoning:
                act = reasoning.split("### Driving Decision")[-1].strip().split()[0].strip('.,')
                actions.add(act.upper())
        self.unique_actions = ["UNKNOWN"] + sorted(list(actions))
        self.action2id = {a: i for i, a in enumerate(self.unique_actions)}
        print(f"[*] Discovered {len(self.unique_actions)} Action Classes: {self.unique_actions}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sample_token = self.tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        label_info = self.label_data[sample_token]
        
        ego_speed = label_info.get("ego_speed", 0.0)
        global_reasoning = label_info.get("global_reasoning", "")
        per_camera = label_info.get("per_camera_perception", [])
        per_camera_text = "\n".join(per_camera)

        messages = [{"role": "user", "content": f"Ego vehicle speed: {ego_speed:.1f} m/s. Describe the driving scene in detail and plan safe action."}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if per_camera_text.strip():
            target_text = f"### Per-Camera Analysis\n{per_camera_text}\n\n{global_reasoning}{self.tokenizer.eos_token}"
        else:
            target_text = f"{global_reasoning}{self.tokenizer.eos_token}"
        
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        # 🚀 [VLA 추가] 타겟 라벨(Action ID 및 Speed) 파싱
        if "### Driving Decision" in global_reasoning:
            act_str = global_reasoning.split("### Driving Decision")[-1].strip().split()[0].strip('.,').upper()
        else:
            act_str = "UNKNOWN"
            
        action_label = self.action2id.get(act_str, 0) # 기본값 UNKNOWN(0)
        target_speed = float(ego_speed)

        imgs_tensor, matrices = [], []
        for cam_name in CAMERA_NAMES:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            with Image.open(img_path) as img:
                imgs_tensor.append(self.transform(img.convert("RGB")))

            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            intrinsics = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32) / 1600.0
            rotation = torch.tensor(calib["rotation"], dtype=torch.float32)
            translation = torch.tensor(calib["translation"], dtype=torch.float32) / 10.0

            matrices.append(torch.cat([intrinsics.flatten(), rotation, translation]))

        return {
            "images": torch.stack(imgs_tensor),
            "matrices": torch.stack(matrices),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "action_label": action_label,
            "target_speed": target_speed
        }

def collate_fn(batch, pad_token_id):
    images = torch.stack([b["images"] for b in batch])
    matrices = torch.stack([b["matrices"] for b in batch])
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    
    # 🚀 [VLA 추가] Action 및 Speed 라벨 배치화
    action_labels = torch.tensor([b["action_label"] for b in batch], dtype=torch.long)
    target_speeds = torch.tensor([b["target_speed"] for b in batch], dtype=torch.float32)
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != pad_token_id).long()
    
    return {
        "images": images,
        "matrices": matrices,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "action_labels": action_labels,
        "target_speeds": target_speeds
    }


# =========================================================
# 4. Main Training Loop
# =========================================================
def train_model(args):
    is_distributed = getattr(args, 'distributed', False) or int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if is_distributed else 0
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset Splits & Action Vocab Building
    temp_dataset = NuScenesVLADataset(args.nusc_root, args.label_path, tokenizer, transform)
    action2id = temp_dataset.action2id
    num_actions = len(action2id)
    
    all_tokens = temp_dataset.tokens
    rnd = torch.Generator()
    rnd.manual_seed(args.seed if hasattr(args, 'seed') else 42)
    perm = torch.randperm(len(all_tokens), generator=rnd).tolist()
    split_idx = int(len(all_tokens) * args.val_split) if args.val_split > 0 else 0
    
    val_tokens = [all_tokens[i] for i in perm[:split_idx]] if split_idx > 0 else None
    train_tokens = [all_tokens[i] for i in perm[split_idx:]]
    
    train_dataset = NuScenesVLADataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=train_tokens, action2id=action2id)
    
    if is_distributed:
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
    else:
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    val_dataloader = None
    if val_tokens:
        val_dataset = NuScenesVLADataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=val_tokens, action2id=action2id)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    # 🚀 [VLA 추가] 모델 초기화 시 Action 개수 전달
    model = MultiCameraPrefixVLA(
        num_action_classes=num_actions,
        model_type=args.model_type,
        language_model_name=args.language_model,
        prefix_len=args.prefix_len,
    )
    model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and getattr(args, 'use_data_parallel', False):
        model = nn.DataParallel(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    
    accumulation_steps = args.accum_steps
    total_steps = (len(dataloader) // accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * 0.1) 
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler() 

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if is_distributed:
            dataloader.sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if (not is_distributed or local_rank == 0) else dataloader
        
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            images = batch["images"].to(device)
            matrices = batch["matrices"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            action_labels = batch["action_labels"].to(device)
            target_speeds = batch["target_speeds"].to(device)

            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                # 🚀 [VLA 추가] 3가지 Loss 통합 반환
                loss, lm_loss, act_loss, spd_loss = model(images, matrices, input_ids, labels, attention_mask, action_labels, target_speeds)
                loss = loss / accumulation_steps 

            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * accumulation_steps
            
            if hasattr(progress_bar, "set_postfix"):
                # 🚀 [VLA 추가] 진행바에서 각 세부 Loss 모니터링 가능하도록 설정
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * accumulation_steps:.3f}",
                    "lm": f"{lm_loss.item():.3f}",
                    "act": f"{act_loss.item():.3f}",
                    "spd": f"{spd_loss.item():.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

        avg_loss = epoch_loss / len(dataloader)
        
        # ... (Validation Loop: 동일한 구조로 loss 검증) ...
        # (생략: 공간 관계상 Validation 파트는 기존 코드에서 v_loss 추론 부분만 `model(...)`로 호출하시면 동일하게 작동합니다.)
        
        if not is_distributed or local_rank == 0:
            print(f"\n[Epoch {epoch+1}] Train Avg Total Loss: {avg_loss:.4f}")
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict() if isinstance(model, (nn.DataParallel, DDP)) else model.state_dict(), save_path)
            
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["light_drive", "baseline_a", "baseline_b"], default="light_drive")
    parser.add_argument("--nusc-root", default="./")
    parser.add_argument("--label-path", default="./hybrid_teacher_labels_final.json")
    parser.add_argument("--language-model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output-dir", default="./checkpoints/light_drive_vla")
    args = parser.parse_args()
    train_model(args)
