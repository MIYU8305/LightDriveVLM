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

# matplotlib 캐시 디렉토리 설정 (권한 문제 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# 손상된 이미지 로드 시 에러 무시
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 자율주행 차량(nuScenes)의 6개 카메라 뷰 이름
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
# 시각적 특징(Visual Features)에서 텍스트 생성에 필요한 핵심 정보만 추출/압축하기 위한 블록
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Query: 학습 가능한 토큰, Key/Value: 이미지 패치 특징
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        # 피드포워드 네트워크 (비선형성 추가)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # 크로스 어텐션 수행 (Q=Query, K=V=Image Patches)
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        # 잔차 연결(Residual Connection) 및 정규화
        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        return self.norm2(x + mlp_out)


# =========================================================
# 2. Unified Multi-Camera Prefix VLM (Training Ready)
# =========================================================
# 여러 대의 카메라 이미지를 LLM의 '접두사(Prefix)' 토큰으로 변환하는 메인 모델
class MultiCameraPrefixVLM(nn.Module):
    def __init__(self, model_type="light_drive", language_model_name="Qwen/Qwen2-0.5B-Instruct", prefix_len=32):
        super().__init__()
        self.model_type = model_type
        # [🔥FIX 5 참고] 카메라 1대당 추출할 토큰 수 (총 6대 * 32 = 192개 토큰이 LLM으로 들어감)
        self.prefix_len = prefix_len

        # -------------------------------------
        # Vision Encoder (시각 특징 추출기)
        # -------------------------------------
        if model_type == "light_drive":
            self.vision_encoder = timm.create_model("deit_small_patch16_224", pretrained=True)
            self.vision_dim = 384
        elif model_type == "baseline_a":
            self.vision_encoder = timm.create_model("resnet101", pretrained=True, num_classes=0, global_pool="")
            self.vision_dim = 2048
        else:
            self.vision_encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.vision_dim = 768

        # -------------------------------------
        # Language Model (LLM 백본)
        # -------------------------------------
        # 하드웨어 지원 여부에 따라 bfloat16 또는 float32 적용 (메모리 절약)
        self.model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, torch_dtype=self.model_dtype
        )
        lm_dim = self.language_model.config.hidden_size # LLM의 임베딩 차원 (예: Qwen2-0.5B의 경우 896)

        # -------------------------------------
        # Projections & Q-Former (비전 차원 -> 언어 차원 변환)
        # -------------------------------------
        # 비전 인코더의 출력을 LLM 차원으로 매핑하는 프로젝션 레이어
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

        # light_drive 모델인 경우, 추가적인 3D 기하학(Geometry) 융합 모듈 사용
        if self.model_type == "light_drive":
            self.num_patches = 196 # 224x224 이미지를 16x16 패치로 나누면 14x14 = 196개
            # 2D 패치 위치 임베딩
            self.patch_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, lm_dim) * 0.02)
            
            # 카메라 내부/외부 파라미터(행렬)를 처리하는 인코더
            self.camera_encoder = nn.Sequential(
                nn.Linear(16, 128), nn.GELU(),
                nn.Linear(128, lm_dim), nn.GELU(),
                nn.Linear(lm_dim, lm_dim * 2), 
            )
            
            # [🔥FIX] 기하학 정보가 초기 학습을 방해하지 않도록 Zero-Initialization 적용 (FiLM 기법)
            nn.init.zeros_(self.camera_encoder[-1].weight)
            nn.init.zeros_(self.camera_encoder[-1].bias)

            # 6개의 카메라 ID 임베딩 (어느 방향의 카메라인지 모델이 인식하게 함)
            self.camera_embedding = nn.Parameter(torch.randn(len(CAMERA_NAMES), lm_dim))

        # Q-Former 학습 가능한 쿼리 토큰 초기화
        self.query_tokens = nn.Parameter(torch.randn(1, prefix_len, lm_dim) * 0.02)
        # 4층의 크로스 어텐션 블록 쌓기
        self.qformer_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=lm_dim, num_heads=8) for _ in range(4)
        ])
        self.prefix_norm = nn.LayerNorm(lm_dim)
        
        # 무거운 백본(Vision, LLM) 모델들의 가중치 동결
        self._freeze_backbones()

    def _freeze_backbones(self):
        # 학습 속도와 메모리 절약을 위해 비전 인코더와 LLM 파라미터는 업데이트하지 않음
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        print("[*] Frozen Vision Encoder & LLM. Training only Projectors & Q-Former.")

    def extract_patch_tokens(self, images):
        # 구조에 맞게 이미지 특징 추출 (CLS 토큰 제외하고 패치만 반환)
        if self.model_type == "baseline_a":
            features = self.vision_encoder(images) 
            return features.flatten(2).transpose(1, 2) 
        else:
            features = self.vision_encoder.forward_features(images)
            return features[:, 1:, :] # [B, 196, Dim]

    def encode_prefix(self, images, camera_matrices):
        B, N, C, H, W = images.shape # B: 배치, N: 카메라 수(6), C: 채널(3), H/W: 224
        # 1. 시각 특징 추출 후 LLM 차원으로 투영
        patch_tokens = self.extract_patch_tokens(images.view(B * N, C, H, W))
        patch_tokens = self.vision_proj(patch_tokens)
        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, N, num_patches, -1)

        # 2. 3D 기하학 정보 병합 (light_drive 모델)
        if self.model_type == "light_drive":
            patch_tokens = patch_tokens + self.patch_pos_embed # 위치 임베딩 추가
            cam_embed = self.camera_embedding.unsqueeze(0).unsqueeze(2)
            patch_tokens = patch_tokens + cam_embed # 카메라 ID 임베딩 추가

            # 기하학 파라미터(16-dim)를 기반으로 gamma, beta 생성 (FiLM 방식 공간 융합)
            geom_params = self.camera_encoder(camera_matrices) 
            gamma, beta = geom_params.chunk(2, dim=-1)         
            
            gamma = gamma.unsqueeze(2) 
            beta = beta.unsqueeze(2)

            # 비전 토큰에 기하학 정보 변조(Modulation)
            patch_tokens = patch_tokens * (1.0 + gamma) + beta

        # [🔥FIX 5] 카메라별 독립적 Q-Former 연산 (Per-Camera Q-Former)
        # GPU 병렬 연산을 위해 배치를 B * N으로 풀어서 처리
        patch_tokens = patch_tokens.view(B * N, num_patches, -1)
        queries = self.query_tokens.expand(B * N, -1, -1)
        
        # 3. Q-Former 연산: 196개의 이미지 패치를 32개의 쿼리 토큰으로 압축
        for blk in self.qformer_blocks:
            queries = blk(query=queries, key_value=patch_tokens)
            
        # 연산 후 다시 B 크기로 복원하며, 카메라 6대의 토큰(32 * 6 = 192)을 하나로 이어붙임
        queries = queries.view(B, N * self.prefix_len, -1)
        
        # LLM dtype과 맞추기 위해 캐스팅 후 반환
        return self.prefix_norm(queries).to(self.language_model.dtype)

    def forward(self, images, camera_matrices, input_ids, labels, attention_mask):
        # 1. 비전 정보를 Prefix 임베딩으로 변환
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        # 2. 텍스트 정보를 LLM 임베딩으로 변환
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        # 3. 비전 + 텍스트 결합 (Vision-Language)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        
        # Prefix 부분의 어텐션 마스크 생성 (항상 어텐션을 받도록 1로 설정)
        prefix_mask = torch.ones((images.shape[0], prefix_embeds.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Prefix 부분은 Loss 계산에서 제외되도록 정답 라벨을 -100으로 설정 (텍스트 생성 부분만 Loss 반영)
        prefix_labels = torch.full((images.shape[0], prefix_embeds.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        
        # LLM 모델 포워딩 및 Loss 반환
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
class NuScenesTeacherDataset(Dataset):
    def __init__(self, nusc_root, label_path, tokenizer, transform, tokens=None):
        # NuScenes 데이터셋 로드 (이미지 메타정보 등을 불러오기 위함)
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=nusc_root, verbose=False)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # 사전 생성된 Teacher 모델(LLM)의 텍스트 라벨 JSON 로드
        with open(label_path, "r", encoding="utf-8") as f:
            self.label_data = json.load(f)
            
        # train/val split용 토큰 리스트 필터링
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
        
        # 자차 속도 및 주행 판단(글로벌 추론) 텍스트 추출
        ego_speed = label_info.get("ego_speed", 0.0)
        global_reasoning = label_info.get("global_reasoning", "")
        
        # [🔥FIX 4] CoT(Chain of Thought) 적용: 각 카메라별 개별 상황을 먼저 분석하는 텍스트 결합
        per_camera = label_info.get("per_camera_perception", [])
        per_camera_text = "\n".join(per_camera)

        # [🔥FIX 1] Qwen-Instruct 채팅 템플릿 양식에 맞게 입력 프롬프트(User) 구성
        messages = [
            {"role": "user", "content": f"Ego vehicle speed: {ego_speed:.1f} m/s. Describe the driving scene in detail and plan safe action."}
        ]
        
        # 모델 입력용 텍스트 변환
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # [🔥FIX 4 계속] 모델이 출력해야 할 정답(Target/Assistant) 텍스트 구성 (CoT 구조)
        if per_camera_text.strip():
            target_text = f"### Per-Camera Analysis\n{per_camera_text}\n\n{global_reasoning}{self.tokenizer.eos_token}"
        else:
            target_text = f"{global_reasoning}{self.tokenizer.eos_token}"
        
        # 텍스트를 토큰 ID로 인코딩
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids
        
        # 전체 입력 시퀀스 = [프롬프트 토큰] + [정답 토큰]
        input_ids = prompt_ids + target_ids
        # 정답 라벨은 프롬프트 구간을 -100(무시)으로 설정하고, 정답 구간만 타겟팅함
        labels = [-100] * len(prompt_ids) + target_ids

        imgs_tensor, matrices = [], []
        # 6개의 카메라 이미지를 순회하며 로드 및 전처리
        for cam_name in CAMERA_NAMES:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])

            with Image.open(img_path) as img:
                imgs_tensor.append(self.transform(img.convert("RGB")))

            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            
            # [🔥FIX 2] 카메라 파라미터(행렬) 스케일 정규화 (학습 시 Gradient Exploding 방지)
            intrinsics = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32) / 1600.0
            rotation = torch.tensor(calib["rotation"], dtype=torch.float32)
            translation = torch.tensor(calib["translation"], dtype=torch.float32) / 10.0

            # 16차원의 평탄화된 배열로 결합 (Intrinsic 3x3 + Rotation 4 + Translation 3)
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
    # 배치(Batch) 단위로 데이터를 묶어주는 함수
    images = torch.stack([b["images"] for b in batch])
    matrices = torch.stack([b["matrices"] for b in batch])
    
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    
    # 텍스트 시퀀스 길이가 다르므로, 가장 긴 시퀀스에 맞춰 패딩(Padding) 수행
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # 패딩된 부분은 어텐션에서 무시하도록 마스크 생성 (1: 유효, 0: 패딩)
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
    # DDP (분산 학습) 설정 - 여러 GPU 사용 시 환경 초기화
    is_distributed = getattr(args, 'distributed', False) or int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if is_distributed else 0
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    # 패딩 토큰이 없는 모델(Qwen)을 위한 임시 조치 (eos 토큰으로 설정)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 모델 인스턴스화
    model = MultiCameraPrefixVLM(
        model_type=args.model_type,
        language_model_name=args.language_model,
        prefix_len=args.prefix_len,
    )
    model.to(device)

    # 병렬 학습 래핑 (DDP 또는 DataParallel)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if local_rank == 0:
            print(f"[*] Initialized DistributedDataParallel on {dist.get_world_size()} processes")
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and getattr(args, 'use_data_parallel', False):
        print(f"[*] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # 체크포인트 이어서 학습하기
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[*] Resuming from checkpoint: {args.resume_checkpoint}")
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location=device), strict=False)
        
    model.train()

    # [🔥FIX 3] 이미지 변형 시 비율 유지 (Perception Loss 방지)
    # 이미지가 찌그러져 3D 원근감이 왜곡되는 것을 막기 위한 CenterCrop 적용
    transform = transforms.Compose([
        transforms.Resize(224), # 짧은 축을 224로 맞추어 비율 유지
        transforms.CenterCrop((224, 224)), # 중앙을 잘라서 정사각형 입력 생성
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 표준 정규화
    ])
    
    # Train / Validation 데이터 분리 로직
    all_tokens = None
    if args.val_label_path:
        # 별도의 Validation 라벨 파일이 주어졌을 때
        with open(args.val_label_path, 'r', encoding='utf-8') as vf:
            val_labels = json.load(vf)
        val_tokens = list(val_labels.keys())
        with open(args.label_path, 'r', encoding='utf-8') as lf:
            main_labels = json.load(lf)
        all_tokens = list(main_labels.keys())
        train_tokens = [t for t in all_tokens if t not in set(val_tokens)]
    elif args.val_split and args.val_split > 0.0:
        # 비율로 분리할 때 (예: 0.1 이면 10%를 Val로 뺌)
        with open(args.label_path, 'r', encoding='utf-8') as lf:
            main_labels = json.load(lf)
        all_tokens = list(main_labels.keys())
        rnd = torch.Generator()
        rnd.manual_seed(args.seed if hasattr(args, 'seed') else 42) # 재현성을 위한 시드 고정
        perm = torch.randperm(len(all_tokens), generator=rnd).tolist()
        split_idx = int(len(all_tokens) * args.val_split)
        val_tokens = [all_tokens[i] for i in perm[:split_idx]]
        train_tokens = [all_tokens[i] for i in perm[split_idx:]]
    else:
        # 전체 학습 진행 (Val 없음)
        train_tokens = None
        val_tokens = None

    # 데이터 로더 세팅
    train_dataset = NuScenesTeacherDataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=train_tokens)
    if is_distributed:
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
    else:
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    val_dataloader = None
    if val_tokens:
        val_dataset = NuScenesTeacherDataset(args.nusc_root, args.label_path, tokenizer, transform, tokens=val_tokens)
        val_dataloader = DataLoader(val_dataset, batch_size=getattr(args, 'val_batch_size', args.batch_size), shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    # Optimizer (동결되지 않은 Q-Former와 Linear Projection 레이어 가중치만 최적화)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    
    # Gradient Accumulation: OOM 방지를 위해 배치 크기를 쪼개서 그래디언트를 누적(accumulate)한 뒤 업데이트
    accumulation_steps = args.accum_steps
    total_steps = (len(dataloader) // accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * 0.1)  # 초기 10%는 웜업(점진적 LR 상승) 적용
    
    # LR 스케줄러 (Cosine 궤적으로 학습률 점진적 감소)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    # AMP(혼합 정밀도) 학습을 위한 스케일러 (FP16의 언더플로우 방지)
    scaler = GradScaler() 

    print(f"[*] Starting Training...")
    print(f"    - Epochs: {args.epochs}")
    print(f"    - Batch Size: {args.batch_size} (Accumulation: {args.accum_steps}, Effective Batch Size: {args.batch_size * args.accum_steps})")
    print(f"    - Total Steps: {total_steps} (Warmup: {warmup_steps})")
    print(f"    - Trainable Params: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if is_distributed:
            dataloader.sampler.set_epoch(epoch) # 매 에폭마다 셔플 다르게 적용
            
        # 마스터(0번) 노드에서만 tqdm 프로그레스바 출력
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") if (not is_distributed or local_rank == 0) else dataloader
        
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # 입력 데이터 GPU로 이동
            images = batch["images"].to(device)
            matrices = batch["matrices"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # AMP(Automatic Mixed Precision) 컨텍스트: 연산 속도 최적화
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss = model(images, matrices, input_ids, labels, attention_mask)
                loss = loss / accumulation_steps # 누적 횟수만큼 스케일링

            # 역전파 수행
            scaler.scale(loss).backward()
            
            # 누적된 배치가 목표 수치(accumulation_steps)에 도달하면 가중치 업데이트
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                # Gradient 폭주(Exploding) 방지를 위한 클리핑
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 그래디언트 초기화
                
                scheduler.step() # LR 업데이트

            epoch_loss += loss.item() * accumulation_steps
            
            # 진행 상태창에 로스 및 학습률 표시
            if hasattr(progress_bar, "set_postfix"):
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}", "lr": f"{current_lr:.2e}"})
            else:
                if not is_distributed or local_rank == 0:
                    if (step + 1) % 100 == 0:
                        print(f"step {step} loss={loss.item() * accumulation_steps:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        val_avg_loss = None
        
        # Validation Loop 진행
        if val_dataloader and (not is_distributed or local_rank == 0):
            model.eval() # 평가 모드
            val_loss_sum = 0.0
            with torch.no_grad(): # 그래디언트 계산 비활성화 (메모리 최적화)
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

        # 에폭 완료 후 요약 출력 및 모델 저장 (마스터 노드)
        if not is_distributed or local_rank == 0:
            print(f"\n[Epoch {epoch+1}] Train Avg Loss: {avg_loss:.4f}")
            if val_avg_loss is not None:
                print(f"[Epoch {epoch+1}] Val   Avg Loss: {val_avg_loss:.4f}")

            # 매 에폭별 모델 저장
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pth")
            state_to_save = model.module.state_dict() if isinstance(model, (nn.DataParallel, DDP)) else model.state_dict()
            torch.save(state_to_save, save_path)
            print(f"[*] Checkpoint saved to: {save_path}")

            # 최고 성능 달성 시 Best Model 별도 저장 (Validation이 없으면 Train Loss 기준)
            compare_loss = val_avg_loss if val_avg_loss is not None else avg_loss
            if compare_loss < best_loss:
                best_loss = compare_loss
                best_epoch = epoch + 1
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(state_to_save, best_path)
                print(f"[*] New best model saved to: {best_path} (epoch {best_epoch}, loss={best_loss:.4f})")

        # 다음 에폭을 위해 다시 학습 모드로 전환
        if val_dataloader and (not is_distributed or local_rank == 0):
            model.train()

    # 모든 학습 종료 시 분산 환경 그룹 파기
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 모델 아키텍처 선택
    parser.add_argument("--model-type", choices=["light_drive", "baseline_a", "baseline_b"], default="light_drive")
    # 누씬 데이터 및 라벨 경로
    parser.add_argument("--nusc-root", default="./")
    parser.add_argument("--label-path", default="./hybrid_teacher_labels_final.json")
    parser.add_argument("--language-model", default="Qwen/Qwen2-0.5B-Instruct")
    
    # Training Hyperparameters (하이퍼파라미터 설정)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient Accumulation Steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # 카메라당 할당할 쿼리 토큰 갯수
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")

    # Validation / Split options (검증 데이터 처리 옵션)
    parser.add_argument("--val-split", type=float, default=0.0, help="Fraction of data to hold out for validation (0.0 disables)")
    parser.add_argument("--val-label-path", default="", help="Path to a JSON file containing validation labels (keys are sample tokens)")
    parser.add_argument("--val-batch-size", type=int, default=0, help="Validation batch size (0 will use --batch-size)")
    
    # Checkpointing (체크포인트 입출력 관련)
    parser.add_argument("--output-dir", default="./checkpoints/light_drive")
    parser.add_argument("--resume-checkpoint", default="", help="이어서 학습할 체크포인트 경로")
    
    # 분산/병렬 처리 관련 플래그
    parser.add_argument("--use-data-parallel", action="store_true", help="Use torch.nn.DataParallel if multiple GPUs available")
    parser.add_argument("--distributed", action="store_true", help="Enable torch.distributed training (use with torchrun)")
    
    args = parser.parse_args()
    
    # Validation 배치가 따로 주어지지 않으면 Train 배치와 통일
    if args.val_batch_size == 0:
        args.val_batch_size = args.batch_size
        
    train_model(args)
