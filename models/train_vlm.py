import argparse
import json
import os
import time

import torch
import torch.nn as nn
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

# matplotlib 임시 디렉토리 설정 및 잘린 이미지 로드 허용 (데이터 로딩 안정성 확보)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# NuScenes 데이터셋의 6개 카메라 뷰
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

# =========================================================
# 1. Q-Former Style Cross Attention Block
# =========================================================
# Vision 특성(Key, Value)을 Language 모델의 차원으로 압축 및 정렬하기 위한 Cross Attention 블록
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # batch_first=True 설정으로 (B, Seq, Dim) 형태의 텐서를 입력받음
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # 특징 변환을 위한 FFN (GELU 활성화 함수 사용)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # Query: Learnable tokens (또는 이전 레이어의 출력)
        # Key/Value: Vision Encoder에서 추출된 패치 토큰들
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        
        # Residual Connection 및 LayerNorm 적용
        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)


# =========================================================
# 2. Unified Multi-Camera Prefix VLM (Training Ready)
# =========================================================
# Vision-Language 융합 모델 정의. Vision과 LLM 백본은 동결(Freeze)하고,
# Projector와 Q-Former만 학습하여 효율적인 파라미터 튜닝(PEFT)을 수행함.
class MultiCameraPrefixVLM(nn.Module):
    def __init__(
        self,
        model_type="light_drive",
        language_model_name="Qwen/Qwen2-0.5B-Instruct",
        prefix_len=32,
        vision_backbone="vit_base_patch16_224",
        use_camera_embedding=True,
        use_geometry_conditioning=True,
    ):
        super().__init__()
        self.model_type = model_type
        self.vision_backbone = vision_backbone
        self.use_camera_embedding = use_camera_embedding
        self.use_geometry_conditioning = use_geometry_conditioning
        
        # [🔥FIX 5 참고] 카메라 1대당 추출할 시각 토큰의 수. 
        # 최종적으로 LLM에 전달되는 토큰 수는 6 (카메라 수) * 32 (prefix_len) = 192개
        self.prefix_len = prefix_len

        # -------------------------------------
        # Vision Encoder 초기화
        # -------------------------------------
        if model_type == "light_drive":
            self.vision_encoder = timm.create_model(self.vision_backbone, pretrained=True)
            self.vision_dim = self._get_vision_dim(self.vision_encoder)
        elif model_type == "baseline_a":
            # ResNet의 경우 Global Average Pooling을 비활성화하여 공간적(Spatial) 피처 맵을 유지
            self.vision_encoder = timm.create_model("resnet101", pretrained=True, num_classes=0, global_pool="")
            self.vision_dim = 2048
        else:
            self.vision_encoder = timm.create_model(self.vision_backbone, pretrained=True)
            self.vision_dim = self._get_vision_dim(self.vision_encoder)

        # -------------------------------------
        # Language Model 초기화 (bfloat16 지원 시 메모리 최적화 및 학습 안정성 확보)
        # -------------------------------------
        self.model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, torch_dtype=self.model_dtype
        )
        lm_dim = self.language_model.config.hidden_size

        # -------------------------------------
        # Projections & Q-Former 초기화
        # -------------------------------------
        # Vision 채널 차원을 LLM 채널 차원으로 매핑하는 MLP
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

        if self.model_type == "light_drive":
            # ViT 패치 수 (224x224 입력, patch size 16 -> 14x14 = 196)
            self.num_patches = 196
            # 학습 가능한 2D Position Embedding
            self.patch_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, lm_dim) * 0.02)
            
            # 카메라의 Extrinsics/Intrinsics 매트릭스를 기반으로 공간 정보를 주입(FiLM Style)하기 위한 인코더
            if self.use_geometry_conditioning:
                self.camera_encoder = nn.Sequential(
                    nn.Linear(16, 128), nn.GELU(),
                    nn.Linear(128, lm_dim), nn.GELU(),
                    nn.Linear(lm_dim, lm_dim * 2), # Gamma와 Beta 생성을 위해 출력 차원을 2배로 설정
                )
                # 초기화 시 원본 피처에 영향을 주지 않도록 Identity 매핑(zero init) 유도
                nn.init.zeros_(self.camera_encoder[-1].weight)
                nn.init.zeros_(self.camera_encoder[-1].bias)
            else:
                self.camera_encoder = None
            
            # 6개 뷰에 대한 개별적인 카메라 ID 임베딩
            if self.use_camera_embedding:
                self.camera_embedding = nn.Parameter(torch.randn(len(CAMERA_NAMES), lm_dim))
            else:
                self.camera_embedding = None

        # Q-Former용 Learnable Query 생성: Shape (1, prefix_len, lm_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, prefix_len, lm_dim) * 0.02)
        
        # 4-Layer Q-Former 스택 구성
        self.qformer_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=lm_dim, num_heads=8) for _ in range(4)
        ])
        self.prefix_norm = nn.LayerNorm(lm_dim)
        
        # 파라미터 동결 적용
        self._freeze_backbones()

    def _freeze_backbones(self):
        """Vision Encoder와 Language Model의 파라미터를 동결하여 학습 속도 향상 및 과적합 방지"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        print("[*] Frozen Vision Encoder & LLM. Training only Projectors & Q-Former.")

    def _get_vision_dim(self, model):
        """timm 모델의 종류에 따라 달라지는 출력 차원 속성을 동적으로 탐색"""
        if hasattr(model, "embed_dim"):
            return model.embed_dim
        if hasattr(model, "num_features"):
            return model.num_features
        if hasattr(model, "feature_dim"):
            return model.feature_dim
        raise ValueError(f"Cannot infer vision dimension for backbone {self.vision_backbone}")

    def extract_patch_tokens(self, images):
        """Vision 백본을 통과시켜 패치 단위의 특징 맵 추출"""
        if self.model_type == "baseline_a":
            # ResNet의 경우 (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C) 형태로 변환
            features = self.vision_encoder(images) 
            return features.flatten(2).transpose(1, 2) 
        else:
            # ViT의 경우 CLS 토큰을 제외한 패치 토큰만 추출 (B, num_patches, C)
            features = self.vision_encoder.forward_features(images)
            return features[:, 1:, :] 

    def encode_prefix(self, images, camera_matrices):
        """
        6대의 멀티 카메라 이미지를 LLM에 입력하기 위한 Prefix Token으로 변환하는 핵심 파이프라인.
        입력: images (B, N, C, H, W) / camera_matrices (B, N, 16)
        출력: prefix_embeds (B, N * prefix_len, lm_dim)
        """
        B, N, C, H, W = images.shape
        
        # 연산 효율성을 위해 B와 N 차원을 합쳐서(Batch * 6) Vision Encoder 통과
        patch_tokens = self.extract_patch_tokens(images.view(B * N, C, H, W))
        patch_tokens = self.vision_proj(patch_tokens)
        num_patches = patch_tokens.shape[1]
        
        # 다시 B, N 형태로 복구: (B, N, num_patches, lm_dim)
        patch_tokens = patch_tokens.view(B, N, num_patches, -1)

        if self.model_type == "light_drive":
            # 1. Spatial Positional Embedding 추가
            patch_tokens = patch_tokens + self.patch_pos_embed
            
            # 2. Camera ID Embedding 추가
            if self.use_camera_embedding and self.camera_embedding is not None:
                cam_embed = self.camera_embedding.unsqueeze(0).unsqueeze(2)
                patch_tokens = patch_tokens + cam_embed

            # 3. Geometry Conditioning (FiLM 적용)
            # 카메라 파라미터(행렬) 기반으로 각 패치 특징에 affine transformation 수행
            if self.use_geometry_conditioning and self.camera_encoder is not None:
                geom_params = self.camera_encoder(camera_matrices) 
                gamma, beta = geom_params.chunk(2, dim=-1) # (B, N, lm_dim) 2개로 분할
                
                gamma = gamma.unsqueeze(2) 
                beta = beta.unsqueeze(2)

                # Feature Modulation: 피처 값의 스케일(gamma)과 쉬프트(beta) 조정
                patch_tokens = patch_tokens * (1.0 + gamma) + beta

        # [🔥FIX 5] 카메라별 독립적 Q-Former 연산 (Per-Camera Q-Former)
        # GPU 병렬 연산을 위해 다시 Batch와 Camera 차원을 병합 (B*N)
        # 즉, 6개 이미지를 통째로 압축하지 않고 개별 카메라당 32개의 토큰으로 각각 압축함
        patch_tokens = patch_tokens.view(B * N, num_patches, -1)
        queries = self.query_tokens.expand(B * N, -1, -1)
        
        # Q-Former Cross-Attention 연산 (Vision -> Learnable Queries)
        for blk in self.qformer_blocks:
            queries = blk(query=queries, key_value=patch_tokens)
            
        # 카메라 6대에서 나온 32개 토큰들을 시퀀스 차원(Sequence dimension)으로 이어붙임 (Concat)
        # 최종 형태: (B, N * prefix_len, lm_dim) -> (B, 192, 1536)
        queries = queries.view(B, N * self.prefix_len, -1)
        
        # LLM의 입력 데이터 타입(bfloat16 등)에 맞추어 타입 캐스팅
        return self.prefix_norm(queries).to(self.language_model.dtype)

    def forward(self, images, camera_matrices, input_ids, labels, attention_mask):
        # 1. Vision 정보 압축하여 Prefix 임베딩 생성
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        
        # 2. 텍스트 정보 임베딩
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 3. Vision Prefix와 텍스트 임베딩을 시퀀스 축(dim=1)을 기준으로 결합 (Soft Prompting)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        
        # 4. Attention Mask 확장: Prefix 부분은 항상 Attention 하도록(1) 마스크 추가
        prefix_mask = torch.ones((images.shape[0], prefix_embeds.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 5. Labels 확장: Prefix 임베딩 구간은 Loss 계산에서 제외하기 위해 -100 할당 (CrossEntropyLoss의 기본 ignore_index)
        prefix_labels = torch.full((images.shape[0], prefix_embeds.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        
        # 6. LLM Forward 및 Loss 계산
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True
        )
        return outputs.loss


# =========================================================
# 3. Dataset & DataLoader (for JSON Teacher Labels)
# =========================================================
# NuScenes 데이터를 로드하고 Qwen 모델에 적합한 Chat Format 텐서를 생성하는 데이터셋
class NuScenesTeacherDataset(Dataset):
    def __init__(self, nusc_root, label_path, tokenizer, transform, tokens=None):
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=nusc_root, verbose=False)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Teacher VLM(e.g., GPT-4V) 등에 의해 미리 생성된 라벨링 데이터 로드
        with open(label_path, "r", encoding="utf-8") as f:
            self.label_data = json.load(f)
            
        if tokens is None:
            self.tokens = list(self.label_data.keys())
        else:
            self.tokens = [t for t in tokens if t in self.label_data]
            missing = set(tokens) - set(self.tokens)
            if missing and self.nusc.verbose:
                print(f"[!] Warning: {len(missing)} tokens from provided list not found in label file")
        print(f"[*] Loaded {len(self.tokens)} samples for training.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sample_token = self.tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        label_info = self.label_data[sample_token]
        
        ego_speed = label_info.get("ego_speed", 0.0)
        global_reasoning = label_info.get("global_reasoning", "")
        
        # [🔥FIX 4] CoT(Chain of Thought) 적용: 모델이 최종 행동 전 개별 카메라의 인지 상황을 먼저 분석하도록 유도
        per_camera = label_info.get("per_camera_perception", [])
        per_camera_text = "\n".join(per_camera)

        # [🔥FIX 1] Qwen-Instruct 모델 학습을 위한 시스템/유저 템플릿 적용(ChatML)
        messages = [
            {"role": "user", "content": f"Ego vehicle speed: {ego_speed:.1f} m/s. Describe the driving scene in detail and plan safe action."}
        ]
        
        # `add_generation_prompt=True`를 통해 어시스턴트의 답변 시작 토큰까지 포맷팅 완료
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # [🔥FIX 4 계속] Target 텍스트를 CoT 형태로 구성
        if per_camera_text.strip():
            target_text = f"### Per-Camera Analysis\n{per_camera_text}\n\n{global_reasoning}{self.tokenizer.eos_token}"
        else:
            target_text = f"{global_reasoning}{self.tokenizer.eos_token}"
        
        # Prompt와 Target 분리 토크나이징
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        
        input_ids = prompt_ids + target_ids
        # Loss는 모델이 생성해야 할 부분(target)에만 걸리도록 Prompt 구간을 -100으로 마스킹
        labels = [-100] * len(prompt_ids) + target_ids

        imgs_tensor, matrices = [], []
        for cam_name in CAMERA_NAMES:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])

            with Image.open(img_path) as img:
                imgs_tensor.append(self.transform(img.convert("RGB")))

            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            
            # [🔥FIX 2] 카메라 파라미터 정규화
            # 스케일이 매우 큰 파라미터(특히 Intrinsics)가 Linear 레이어를 통과할 때 Gradient Exploding이 발생하는 것을 방지
            intrinsics = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32) / 1600.0
            rotation = torch.tensor(calib["rotation"], dtype=torch.float32)
            translation = torch.tensor(calib["translation"], dtype=torch.float32) / 10.0

            # 16차원의 평탄화된(Flattened) 기하학 파라미터 생성
            matrices.append(torch.cat([
                intrinsics.flatten(),
                rotation,
                translation,
            ]))

        return {
            "images": torch.stack(imgs_tensor),
            "matrices": torch.stack(matrices),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch, pad_token_id):
    """DataLoader에서 배치 내 가변 길이 시퀀스를 맞춰주는 커스텀 콜레이트 함수"""
    images = torch.stack([b["images"] for b in batch])
    matrices = torch.stack([b["matrices"] for b in batch])
    
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    
    # NLP 시퀀스 패딩 처리 (배치 내 최대 길이에 맞춰)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    # 패딩 된 부분은 Loss가 계산되지 않도록 -100으로 처리
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    # 실제 유효 토큰 구간을 표시하는 Attention Mask 생성
    attention_mask = (input_ids_padded != pad_token_id).long()
    
    return {
        "images": images,
        "matrices": matrices,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask
    }


# =========================================================
# 4. Main Training Loop
# =========================================================
def train_model(args):
    # 분산 학습(DDP) 환경 설정
    is_distributed = getattr(args, 'distributed', False) or int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if is_distributed else 0
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 토크나이저 초기화 및 pad 토큰 설정 (Qwen 등은 pad 토큰이 없는 경우가 있어 eos_token 활용)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = MultiCameraPrefixVLM(
        model_type=args.model_type,
        language_model_name=args.language_model,
        prefix_len=args.prefix_len,
        vision_backbone=args.vision_backbone,
    )
    model.to(device)

    # 병렬 처리 래퍼(Wrapper) 적용
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if local_rank == 0:
            print(f"[*] Initialized DistributedDataParallel on {dist.get_world_size()} processes")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and getattr(args, 'use_data_parallel', False):
        print(f"[*] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[*] Resuming from checkpoint: {args.resume_checkpoint}")
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device), strict=False)
        
    model.train()

    # [🔥FIX 3] 이미지 전처리 (비율 유지)
    # 단순 224x224 Resize는 왜곡을 유발해 Vision 모델의 사전 학습 피처를 망가뜨릴 수 있으므로, 짧은 축 Resize 후 CenterCrop 적용
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset Splits (Validation 분할 논리)
    all_tokens = None
    if args.val_label_path:
        with open(args.val_label_path, 'r', encoding='utf-8') as vf:
            val_labels = json.load(vf)
        val_tokens = list(val_labels.keys())
        with open(args.label_path, 'r', encoding='utf-8') as lf:
            main_labels = json.load(lf)
        all_tokens = list(main_labels.keys())
        train_tokens = [t for t in all_tokens if t not in set(val_tokens)]
    elif args.val_split and args.val_split > 0.0:
        with open(args.label_path, 'r', encoding='utf-8') as lf:
            main_labels = json.load(lf)
        all_tokens = list(main_labels.keys())
        # 재현성 보장을 위해 Seed 고정 후 데이터 섞기
        rnd = torch.Generator()
        rnd.manual_seed(args.seed if hasattr(args, 'seed') else 42)
        perm = torch.randperm(len(all_tokens), generator=rnd).tolist()
        split_idx = int(len(all_tokens) * args.val_split)
        val_tokens = [all_tokens[i] for i in perm[:split_idx]]
        train_tokens = [all_tokens[i] for i in perm[split_idx:]]
    else:
        train_tokens = None
        val_tokens = None

    # 데이터 로더 준비
    train_dataset = NuScenesTeacherDataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=train_tokens)
    if is_distributed:
        # DDP에서는 각 GPU 노드에 데이터를 분산 배정하는 DistributedSampler 필수
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
    else:
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    val_dataloader = None
    if val_tokens:
        val_dataset = NuScenesTeacherDataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=val_tokens)
        val_dataloader = DataLoader(val_dataset, batch_size=getattr(args, 'val_batch_size', args.batch_size), shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    # Optimizer (동결되지 않은 Projector와 Q-Former 파라미터만 최적화)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    
    # 스케줄러 및 Accumulation 스텝 계산
    accumulation_steps = args.accum_steps
    total_steps = (len(dataloader) // accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * 0.1)  # 초기 10%를 웜업 스텝으로 할당 (Transformer 학습 안정화)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    # Mixed Precision(FP16) 환경에서 기울기 소실 방지를 위한 GradScaler
    scaler = GradScaler() 

    print(f"[*] Starting Training...")
    print(f"    - Epochs: {args.epochs}")
    print(f"    - Batch Size: {args.batch_size} (Accumulation: {args.accum_steps}, Effective Batch Size: {args.batch_size * args.accum_steps})")
    print(f"    - Total Steps: {total_steps} (Warmup: {warmup_steps})")
    print(f"    - Trainable Params: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    best_epoch = -1

    # 에포크 루프 시작
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if is_distributed:
            # 매 에포크마다 셔플 시드를 변경하기 위해 호출
            dataloader.sampler.set_epoch(epoch)
        # 메인 프로세스(Rank 0)에서만 진행 상황을 로깅
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if (not is_distributed or local_rank == 0) else dataloader
        
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            images = batch["images"].to(device)
            matrices = batch["matrices"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Mixed Precision 연산 활성화 영역
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss = model(images, matrices, input_ids, labels, attention_mask)
                # Gradient Accumulation을 위해 Loss 값을 스텝 수로 나눔
                loss = loss / accumulation_steps 

            # 역전파 수행
            scaler.scale(loss).backward()
            
            # Accumulation Step 도달 시 가중치 업데이트
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                # Gradient Clipping 적용 전 스케일 원복
                scaler.unscale_(optimizer)
                # 기울기 폭주 방지(Gradient Clipping)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 학습률 스케줄러 업데이트 (Step 단위 갱신)
                scheduler.step()

            # 표기를 위한 실제 스텝 로스 계산 복구
            epoch_loss += loss.item() * accumulation_steps
            
            # 로깅 인터페이스
            if hasattr(progress_bar, "set_postfix"):
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}", "lr": f"{current_lr:.2e}"})
            else:
                if not is_distributed or local_rank == 0:
                    if (step + 1) % 100 == 0:
                        print(f"step {step} loss={loss.item() * accumulation_steps:.4f}")

        # Validation 평가 루틴
        avg_loss = epoch_loss / len(dataloader)
        val_avg_loss = None
        
        if val_dataloader and (not is_distributed or local_rank == 0):
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad(): # 추론 모드이므로 그래디언트 연산 및 메모리 절약
                for val_batch in tqdm(val_dataloader, desc="Validation"):
                    v_images = val_batch["images"].to(device)
                    v_matrices = val_batch["matrices"].to(device)
                    v_input_ids = val_batch["input_ids"].to(device)
                    v_labels = val_batch["labels"].to(device)
                    v_attention_mask = val_batch["attention_mask"].to(device)
                    
                    with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        v_loss = model(v_images, v_matrices, v_input_ids, v_labels, v_attention_mask)
                    val_loss_sum += v_loss.item()
            val_avg_loss = val_loss_sum / len(val_dataloader)

        # 모델 저장 루틴 (체크포인트)
        if not is_distributed or local_rank == 0:
            print(f"\n[Epoch {epoch+1}] Train Avg Loss: {avg_loss:.4f}")
            if val_avg_loss is not None:
                print(f"[Epoch {epoch+1}] Val   Avg Loss: {val_avg_loss:.4f}")

            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
            # DDP 또는 DP 래퍼에 싸여 있다면 내부의 실제 모델(module) state_dict를 저장
            state_to_save = model.module.state_dict() if isinstance(model, (nn.DataParallel, DDP)) else model.state_dict()
            torch.save(state_to_save, save_path)
            print(f"[*] Checkpoint saved to: {save_path}")

            # 베스트 모델 갱신 (Val Loss가 있으면 기준으로, 없으면 Train Loss 기준)
            compare_loss = val_avg_loss if val_avg_loss is not None else avg_loss
            if compare_loss < best_loss:
                best_loss = compare_loss
                best_epoch = epoch + 1
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(state_to_save, best_path)
                print(f"[*] New best model saved to: {best_path} (epoch {best_epoch}, loss={best_loss:.4f})")

        # 평가 이후 다시 학습 모드로 전환
        if val_dataloader and (not is_distributed or local_rank == 0):
            model.train()

    # 분산 환경 정리
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["light_drive", "baseline_a", "baseline_b"], default="light_drive")
    parser.add_argument("--vision-backbone", default="vit_base_patch16_224", help="timm vision backbone to use for light_drive or baseline_b")
    parser.add_argument("--use-camera-embedding", dest="use_camera_embedding", action="store_true", help="Enable camera embedding for light_drive")
    parser.add_argument("--no-camera-embedding", dest="use_camera_embedding", action="store_false", help="Disable camera embedding for light_drive")
    parser.add_argument("--use-geometry-conditioning", dest="use_geometry_conditioning", action="store_true", help="Enable geometry conditioning for light_drive")
    parser.add_argument("--no-geometry-conditioning", dest="use_geometry_conditioning", action="store_false", help="Disable geometry conditioning for light_drive")
    parser.set_defaults(use_camera_embedding=True, use_geometry_conditioning=True)
    parser.add_argument("--nusc-root", default="./")
    parser.add_argument("--label-path", default="./hybrid_teacher_labels_final.json")
    parser.add_argument("--language-model", default="Qwen/Qwen2-0.5B-Instruct")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient Accumulation Steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # 이제 prefix-len 32를 설정하면 6대 카메라를 합쳐 최종적으로 LLM에 192개의 시각 토큰이 전달됩니다.
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")

    # Validation / Split options
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction of data to hold out for validation (0.0 disables)")
    parser.add_argument("--val-label-path", default="", help="Path to a JSON file containing validation labels (keys are sample tokens)")
    parser.add_argument("--val-batch-size", type=int, default=0, help="Validation batch size (0 will use --batch-size)")
    
    # Checkpointing
    parser.add_argument("--output-dir", default="./checkpoints/light_drive")
    parser.add_argument("--resume-checkpoint", default="", help="이어서 학습할 체크포인트 경로")
    parser.add_argument("--use-data-parallel", action="store_true", help="Use torch.nn.DataParallel if multiple GPUs available")
    parser.add_argument("--distributed", action="store_true", help="Enable torch.distributed training (use with torchrun)")
    
    args = parser.parse_args()
    if args.val_batch_size == 0:
        args.val_batch_size = args.batch_size
    train_model(args)
