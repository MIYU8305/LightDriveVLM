import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import timm
import torch
import torch.nn as nn
from nuscenes.nuscenes import NuScenes
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
]

# ==========================================
# 1. 하이브리드 데이터셋 클래스
# ==========================================
class HybridFusionDataset(Dataset):
    def __init__(self, nusc, label_path, transform=None):
        self.nusc = nusc
        self.transform = transform
        self.camera_names = CAMERA_NAMES

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다. 먼저 라벨을 생성하세요: {label_path}")

        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self.samples = [sample for sample in nusc.sample if sample["token"] in self.labels]
        print(f"[*] 하이브리드 라벨 {len(self.labels)}개 로드 (학습 사용: {len(self.samples)}개)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs, matrices = [], []

        for cam_name in self.camera_names:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            mat = torch.cat([
                torch.tensor(calib["camera_intrinsic"], dtype=torch.float32).flatten(),
                torch.tensor(calib["rotation"], dtype=torch.float32),
                torch.tensor(calib["translation"], dtype=torch.float32),
            ])
            matrices.append(mat)

        return torch.stack(imgs), torch.stack(matrices), self.labels[sample["token"]]

# ==========================================
# 2. LightDriveVLM 모델 (추론 기능 포함)
# ==========================================
class MultiCameraPrefixVLM(nn.Module):
    def __init__(self, language_model_name="gpt2", prefix_len=32, num_unfrozen_lm_blocks=2):
        super().__init__()
        self.prefix_len = prefix_len
        self.vision_encoder = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=0)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        lm_dim = self.language_model.config.n_embd

        self.vision_proj = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, lm_dim), nn.GELU())
        self.camera_mlp = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, lm_dim), nn.GELU(), nn.Linear(lm_dim, lm_dim))
        
        self.query_tokens = nn.Parameter(torch.randn(1, prefix_len, lm_dim) * 0.02)
        self.prefix_attention = nn.MultiheadAttention(lm_dim, num_heads=12, batch_first=True)
        self.prefix_norm = nn.LayerNorm(lm_dim)

        self._freeze_language_model(num_unfrozen_lm_blocks)

    def _freeze_language_model(self, num_unfrozen_lm_blocks):
        for param in self.language_model.parameters():
            param.requires_grad = False
        transformer = self.language_model.transformer
        for block in transformer.h[-num_unfrozen_lm_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in transformer.ln_f.parameters():
            param.requires_grad = True

    def encode_prefix(self, images, camera_matrices):
        batch_size, num_cams, channels, height, width = images.shape
        image_features = self.vision_encoder(images.view(batch_size * num_cams, channels, height, width))
        image_features = self.vision_proj(image_features).view(batch_size, num_cams, -1)
        camera_features = self.camera_mlp(camera_matrices)
        fused_features = image_features + camera_features

        queries = self.query_tokens.expand(batch_size, -1, -1)
        prefix, _ = self.prefix_attention(queries, fused_features, fused_features)
        return self.prefix_norm(prefix)

    def forward(self, images, camera_matrices, input_ids, attention_mask):
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        token_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(attention_mask.size(0), self.prefix_len, dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        prefix_labels = torch.full((input_ids.size(0), self.prefix_len), -100, dtype=input_ids.dtype, device=input_ids.device)
        token_labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([prefix_labels, token_labels], dim=1)

        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, images, camera_matrices, tokenizer, max_new_tokens=60, **kwargs):
        """추론 시 사용되는 함수: 정답 텍스트 없이 이미지만 보고 상황을 스스로 생성"""
        self.eval()
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        
        outputs = self.language_model.generate(
            inputs_embeds=prefix_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            **kwargs
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ==========================================
# 3. 조기 종료 (Loss 1.5 강제 조건)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, target_loss=1.5, path="./checkpoints/best_vlm.pth"):
        self.patience = patience
        self.target_loss = target_loss
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f"[*] EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                if val_loss > self.target_loss:
                    print(f"[!] Loss ({val_loss:.4f})가 목표({self.target_loss})보다 높습니다. 강제 학습을 지속합니다...")
                    self.counter = 0
                else:
                    self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, self.path)
        print(f"[*] 새로운 베스트 모델 저장 (Val Loss: {self.best_loss:.4f})")

# ==========================================
# 4. 학습 루프
# ==========================================
def tokenize_labels(tokenizer, labels, max_length, device):
    encoded = tokenizer(list(labels), return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)

def train(args):
    device_ids = list(range(torch.cuda.device_count()))
    main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=False)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = HybridFusionDataset(nusc, args.label_path, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    train_db, val_db = random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_db, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = MultiCameraPrefixVLM(language_model_name=args.language_model, prefix_len=32)
    model.to(main_device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=10, target_loss=args.target_loss, path="./checkpoints/best_lightdrive_vlm.pth")

    print(f"[*] Training Started on {main_device} | Target Loss: {args.target_loss}")
    
    for epoch in range(args.max_epochs):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for imgs, mats, labels in pbar:
            imgs, mats = imgs.to(main_device, non_blocking=True), mats.to(main_device, non_blocking=True)
            input_ids, attention_mask = tokenize_labels(tokenizer, labels, 256, main_device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(imgs, mats, input_ids=input_ids, attention_mask=attention_mask).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, mats, labels in val_loader:
                imgs, mats = imgs.to(main_device, non_blocking=True), mats.to(main_device, non_blocking=True)
                input_ids, attention_mask = tokenize_labels(tokenizer, labels, 256, main_device)
                loss = model(imgs, mats, input_ids=input_ids, attention_mask=attention_mask).mean()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"[*] Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("[!] Target loss reached and patience exceeded. Training finished.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc-root", default="./nuscenes")
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--label-path", default="./hybrid_labels.json") # 새로 만든 하이브리드 라벨 사용
    parser.add_argument("--language-model", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=4) # A6000 4장이므로 글로벌 배치는 16
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--target-loss", type=float, default=1.5) # 강제 학습 로스 기준
    train(parser.parse_args())
