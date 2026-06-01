import os
import sys
import json
import argparse
import re

import torch
import numpy as np

from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================================================
# Runtime Environment (실행 환경 설정)
# =========================================================

# HuggingFace 다운로드 미러 사이트 설정 (네트워크 이슈 방지)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 사용할 GPU 지정 (0번부터 7번까지 총 8개)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
# matplotlib 임시 폴더 권한 문제 방지
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 손상되거나 잘린 이미지를 불러올 때 발생하는 에러 무시
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("=" * 80)
print(f"[*] Python: {sys.executable}")
print(f"[*] Torch: {torch.__version__}")
try:
    import transformers
    print(f"[*] Transformers: {transformers.__version__}")
except Exception as e:
    print(f"[!] Transformers import failed: {e}")
print("=" * 80)


# =========================================================
# Utils (유틸리티 함수 모음)
# =========================================================

def cleanup_text(text):
    """
    LLM이 생성한 텍스트 중 반복되거나 깨진 출력을 정리하는 함수
    """
    text = text.strip()

    # 모델이 특정 단어를 무한 반복하는 환각(Hallucination) 현상 제거 (예: A A A A...)
    patterns = [
        r'(A\s+){8,}',
        r'(The\s+){8,}',
        r'(In\s+){8,}',
    ]

    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # 중복된 띄어쓰기(공백)를 하나로 통일
    text = re.sub(r'\s+', ' ', text)

    # 문장이 마침표로 끝나지 않고 잘린 경우 마침표 추가
    if len(text) > 10 and text[-1].isalnum():
        if not text.endswith((".", "!", "?")):
            text += "."

    return text.strip()


def get_cuda_input_device(model):
    # 모델의 첫 번째 임베딩 레이어가 올라가 있는 GPU 디바이스 반환
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device("cuda:0")


def require_cuda_gpus(required_gpus=2):
    # 필요한 최소 GPU 개수가 충족되는지 확인하고 디바이스 정보 출력
    print(f"[*] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    gpu_count = torch.cuda.device_count()

    print(f"[*] Detected GPUs: {gpu_count}")

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"    - cuda:{i}: {props.name} ({props.total_memory / (1024**3):.1f} GiB)")

    if gpu_count < required_gpus:
        raise RuntimeError(f"Need at least {required_gpus} GPUs.")

    return gpu_count


def load_rgb_image(path):
    # 이미지를 열고 RGB 포맷으로 변환하여 반환
    with Image.open(path) as img:
        return img.convert("RGB").copy()


# =========================================================
# NuScenes GT Helper (NuScenes 정답 데이터 추출기)
# =========================================================

def get_camera_specific_annotations(
    nusc,
    sample,
    cam_name,
    target_classes,
    max_dist=40.0
):
    """
    특정 카메라 뷰(cam_name)에 존재하는 3D 바운딩 박스(GT) 정보를 추출하여
    LLaVA 모델에게 '힌트(Hint)'로 제공할 텍스트를 생성합니다.
    """
    _, boxes, _ = nusc.get_sample_data(sample["data"][cam_name])
    objects = []

    for box in boxes:
        # 박스 카테고리 추출 (예: vehicle.car -> car)
        category = box.name.split(".")[-1]

        # 관심 있는 클래스(자동차, 보행자 등)가 아니면 무시
        if category not in target_classes:
            continue

        # 객체의 x, y, z 좌표 확인
        x, y, z = box.center

        # z값이 0 이하(카메라 뒤쪽)이거나 최대 탐지 거리(40m) 밖이면 무시
        if z <= 0 or z > max_dist:
            continue

        objects.append(f"{category} ({z:.1f}m)")

    if len(objects) == 0:
        return "No nearby critical objects."

    return "Nearby objects: " + ", ".join(objects)


# =========================================================
# Fusion Reasoning (다중 카메라 융합 및 주행 판단)
# =========================================================

def call_global_fusion(
    fusion_model,
    fusion_tokenizer,
    llava_descriptions,
    ego_speed
):
    """
    LLaVA가 개별적으로 분석한 6개의 카메라 뷰 텍스트를 모아서,
    단일 언어 모델(Qwen)이 전역적인 상황을 판단하고 주행 행동을 결정하도록 합니다.
    """
    # 6개 카메라 분석 결과를 하나의 텍스트로 병합
    formatted_views = "\n".join(llava_descriptions)

    # 퓨전 모델(Qwen)에게 부여할 시스템 역할 및 규칙
    system_prompt = """
You are the global reasoning module for an autonomous driving system.

You receive synchronized descriptions from 6 surround-view cameras.

CRITICAL SPATIAL RULES:
1. Objects and traffic lights detected in BACK, BACK_LEFT, and BACK_RIGHT cameras are BEHIND the ego vehicle.
2. DO NOT stop or slow down for red lights or stop signs seen in the REAR cameras.
3. Focus on forward and lateral hazards for driving decisions.

Tasks:
1. Build unified 360-degree understanding
2. Detect moving risks
3. Detect cut-in / occlusion / pedestrian risks
4. Decide safest driving action

Be concise and factual. Avoid repetition.
"""

    # 사용자 프롬프트: 현재 차량 속도와 6개 카메라 분석 결과 전달
    user_prompt = f"""
Ego vehicle speed: {ego_speed:.1f} m/s

Camera observations:

{formatted_views}

Generate EXACTLY this format:

### Global Scene
- summarize overall scene

### Risk Assessment
- moving hazards
- pedestrian risks
- cut-in risks
- occlusion risks

### Driving Decision
- STOP / SLOW_DOWN / MAINTAIN_SPEED / YIELD

### Reason
- concise explanation
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 모델에 맞게 채팅 템플릿 적용 (프롬프트 포매팅)
    text = fusion_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = get_cuda_input_device(fusion_model)

    # 텍스트 토큰화
    inputs = fusion_tokenizer(
        [text],
        return_tensors="pt"
    ).to(device)

    # 텍스트 생성 (Inference)
    with torch.inference_mode():
        outputs = fusion_model.generate(
            **inputs,
            max_new_tokens=220,
            min_new_tokens=80,
            do_sample=False, # 일관된 판단을 위해 Greedy Search 사용 (temperature=0)
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1, # 반복 억제
            no_repeat_ngram_size=5,
            pad_token_id=fusion_tokenizer.eos_token_id
        )

    # 프롬프트 부분을 제외하고 순수하게 생성된 텍스트만 추출
    outputs = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, outputs)
    ]

    response = fusion_tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    return cleanup_text(response)


# =========================================================
# Main Pipeline (전체 데이터 생성 파이프라인)
# =========================================================

def generate_data(
    nusc_root,
    save_path,
    model_id,
    fusion_model_id
):
    skipped_path = save_path + ".skipped.json" # 에러가 난 샘플들을 따로 저장할 경로

    print(f"[*] Loading NuScenes from: {nusc_root}")

    # NuScenes 데이터셋 초기화
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=nusc_root,
        verbose=False
    )

    print(f"[*] Total samples: {len(nusc.sample)}")

    camera_names = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT"
    ]

    # 분석 대상 객체 클래스 목록
    target_classes = {
        "car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"
    }

    hybrid_labels = {}

    # 기존에 만들어둔 라벨 파일이 있다면 이어서 작업하기 위해 로드 (Resume 기능)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            hybrid_labels = json.load(f)

    skipped_samples = {}

    if os.path.exists(skipped_path):
        with open(skipped_path, "r") as f:
            skipped_samples = json.load(f)

    # 아직 처리되지 않은(JSON에 없는) 샘플들만 필터링
    pending_samples = [
        sample for sample in nusc.sample
        if sample["token"] not in hybrid_labels
    ]

    print(f"[*] Pending Samples: {len(pending_samples)}")

    require_cuda_gpus(2)

    # =========================================================
    # Quantization (VRAM 절약을 위한 4비트 양자화 설정)
    # =========================================================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True, # 이중 양자화로 메모리 추가 절약
        bnb_4bit_quant_type="nf4", # NormalFloat4 타입
        bnb_4bit_compute_dtype=torch.bfloat16 # 연산은 bfloat16 정밀도로 수행
    )

    # =========================================================
    # Load LLaVA (이미지를 텍스트로 묘사해줄 Vision-Language Model)
    # =========================================================
    print(f"[*] Loading LLaVA: {model_id}")

    processor = LlavaProcessor.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto", # 여러 GPU에 모델을 자동으로 분산 배치
        low_cpu_mem_usage=True
    )

    input_device = get_cuda_input_device(model)
    print(f"[*] LLaVA device: {input_device}")

    # =========================================================
    # Load Fusion LLM (6개 시점을 모아 판단을 내릴 Language Model)
    # =========================================================
    print(f"[*] Loading Fusion LLM: {fusion_model_id}")

    fusion_tokenizer = AutoTokenizer.from_pretrained(
        fusion_model_id,
        use_fast=False
    )

    fusion_model = AutoModelForCausalLM.from_pretrained(
        fusion_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # =========================================================
    # Main Loop (데이터 생성 시작)
    # =========================================================
    for sample in tqdm(pending_samples):
        token = sample["token"]

        images = []
        prompts = []

        skip_sample = False

        # 1. 6개의 카메라 이미지를 돌면서 분석 준비
        for cam_name in camera_names:
            try:
                # NuScenes에서 해당 카메라의 이미지 경로 찾기
                sample_data = nusc.get(
                    "sample_data",
                    sample["data"][cam_name]
                )

                img_path = os.path.join(
                    nusc.dataroot,
                    sample_data["filename"]
                )

                image = load_rgb_image(img_path)
                images.append(image)

            except Exception as e:
                # 이미지가 없거나 손상된 경우 건너뛰기 큐에 저장
                skipped_samples[token] = {
                    "camera": cam_name,
                    "error": str(e)
                }
                skip_sample = True
                break

            # 정답 데이터(GT)로부터 해당 카메라 뷰에 있는 객체 정보를 텍스트 힌트로 가져옴
            gt_info = get_camera_specific_annotations(
                nusc,
                sample,
                cam_name,
                target_classes
            )

            # LLaVA에게 내릴 프롬프트 작성 (GT 힌트를 포함하여 더 정확한 묘사 유도)
            prompt = f"""
USER: <image>

You are an expert autonomous driving perception system.
Analyze the `{cam_name}` image.

Ground truth 3D hint:
{gt_info}

Respond STRICTLY in this bulleted format:
- Vehicles: (describe nearby vehicles and positions)
- Pedestrians / Cyclists: (describe vulnerable road users)
- Road Layout: (describe lanes, traffic lights, intersections)
- Driving Hazards: (describe any visible hazards)

Keep it concise, factual, and direct. Do not write introductory sentences.
ASSISTANT:
"""
            prompts.append(prompt)

        if skip_sample:
            continue

        # 2. LLaVA를 사용해 6개 카메라 동시 분석 (Batch Inference)
        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(input_device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=20,
                do_sample=True, # 묘사의 다양성을 위해 샘플링 켬
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.05,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        generated = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        per_camera_outputs = []

        # 생성된 응답 후처리 및 정리
        for i, text in enumerate(generated):
            clean = text.split("ASSISTANT:")[-1].strip()
            clean = cleanup_text(clean)

            # 내용이 없는 깡통 응답 필터링 처리
            if not clean.replace("*", "").replace("-", "").strip():
                clean = "- Vehicles: None detected\n- Pedestrians / Cyclists: None detected\n- Road Layout: No visible features\n- Driving Hazards: None"

            per_camera_outputs.append(
                f"[{camera_names[i]}]:\n{clean}"
            )

        # =====================================================
        # Global Fusion (다중 시점 융합)
        # =====================================================

        ego_speed = 0.0

        # 3. 현재 차량의 속도(m/s) 계산
        try:
            sd_current_token = sample["data"]["CAM_FRONT"]
            sd_current = nusc.get("sample_data", sd_current_token)
            ego_current = nusc.get("ego_pose", sd_current["ego_pose_token"])

            # 이전 프레임(prev)이 존재한다면 이동 거리와 시간 차이를 통해 속도 계산
            if sd_current["prev"]:
                sd_prev = nusc.get("sample_data", sd_current["prev"])
                ego_prev = nusc.get("ego_pose", sd_prev["ego_pose_token"])

                # X, Y 평면 기준 이동 거리 계산 (유클리드 거리)
                dist = np.linalg.norm(
                    np.array(ego_current["translation"][:2]) - np.array(ego_prev["translation"][:2])
                )
                # Timestamp(마이크로초)를 초(Seconds)로 변환
                time_diff = (ego_current["timestamp"] - ego_prev["timestamp"]) / 1e6

                if time_diff > 0:
                    ego_speed = dist / time_diff
        except Exception as e:
            pass

        # 4. Qwen 모델을 호출하여 6개 카메라 묘사 텍스트를 종합하고 최종 운전 판단 생성
        global_reasoning = call_global_fusion(
            fusion_model,
            fusion_tokenizer,
            per_camera_outputs,
            ego_speed
        )

        # 5. 최종 결과를 딕셔너리에 저장
        hybrid_labels[token] = {
            "per_camera_perception": per_camera_outputs,
            "global_reasoning": global_reasoning,
            "num_cameras": 6,
            "ego_speed": round(ego_speed, 2)
        }

        # 10개 샘플이 처리될 때마다 진행 상황 중간 저장 (크래시 대비)
        if len(hybrid_labels) % 10 == 0:
            with open(save_path, "w") as f:
                json.dump(
                    hybrid_labels,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            with open(skipped_path, "w") as f:
                json.dump(
                    skipped_samples,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

    # 루프 종료 후 최종 저장
    with open(save_path, "w") as f:
        json.dump(
            hybrid_labels,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 80)
    print("[*] DONE")
    print(f"[*] Saved to: {save_path}")
    print("=" * 80)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # NuScenes 데이터셋이 위치한 최상위 폴더 경로
    parser.add_argument(
        "--nusc-root",
        default="/workspace"
    )

    # 결과물이 저장될 JSON 파일 경로
    parser.add_argument(
        "--save-path",
        default="./hybrid_teacher_labels_final.json"
    )

    # 이미지 분석용 VLM 모델(LLaVA) 경로
    parser.add_argument(
        "--model-id",
        default="./models/llava-1.5-7b-hf"
    )

    # 글로벌 추론 및 융합을 위한 LLM(Qwen) 경로
    parser.add_argument(
        "--fusion-model-id",
        default="./models/Qwen2.5-7B-Instruct"
    )

    args = parser.parse_args()

    # 스크립트 실행
    generate_data(
        nusc_root=args.nusc_root,
        save_path=args.save_path,
        model_id=args.model_id,
        fusion_model_id=args.fusion_model_id
    )
