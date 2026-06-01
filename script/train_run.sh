#!/usr/bin/env bash
# 호환성을 위해 현재 셸이 bash가 아닐 경우 bash로 재실행
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# 파이프라인 에러 방지 (Robust Bash Strict Mode)
# -e: 파이프라인 내 명령어 실패 시 즉시 종료
# -u: 정의되지 않은 변수 사용 시 에러 발생
# -o pipefail: 파이프라인 중간에서 발생한 에러를 무시하지 않고 반환
set -euo pipefail

# ==============================================================================
# light_drive Ablation Study 매트릭스 구성
# 목적: ViT/DeiT 백본 차이 및 Multi-Camera 퓨전을 위한 2가지 조건화 기법의 영향도 평가
# 총 실험 횟수: 2 (Backbones) x 2 (Cam Emb) x 2 (Geom) = 8회 순차 실행
# ==============================================================================

LOG_DIR=./logs
mkdir -p "$LOG_DIR" # 실험 기록용 로그 디렉토리 보장

# 실험 하이퍼파라미터 검색 공간 정의
BACKBONES=("deit_small_patch16_224" "vit_base_patch16_224")
CAM_EMB=("true" "false")
GEOM=("true" "false")

for backbone in "${BACKBONES[@]}"; do
  for cam_emb in "${CAM_EMB[@]}"; do
    for geom in "${GEOM[@]}"; do
      
      # 1. 동적 실험 식별자(Experiment Name) 생성
      # 체크포인트 저장 폴더명 및 로그 파일명으로 사용되어 실험 결과 추적을 용이하게 함
      exp_name="light_drive_${backbone}"
      
      if [[ "$cam_emb" == "true" ]]; then
        exp_name+="_cam_on"
      else
        exp_name+="_cam_off"
      fi
      
      if [[ "$geom" == "true" ]]; then
        exp_name+="_geom_on"
      else
        exp_name+="_geom_off"
      fi

      OUT_DIR="./checkpoints/${exp_name}"
      # 타임스탬프를 부여하여 이전 실험 로그 덮어쓰기 방지
      LOGFILE="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

      echo "============================================================="
      echo "Running experiment: $exp_name"
      echo "Backbone: $backbone"
      echo "Camera Embedding: $cam_emb"
      echo "Geometry Conditioning: $geom"
      echo "Output dir: $OUT_DIR"
      echo "Log file: $LOGFILE"
      echo "============================================================="

      # 2. Base Command 구성 (DDP 런처 적용)
      # 단일 노드 내 8개의 GPU를 사용하는 분산 학습 명령
      # 배열(cmd)을 사용하여 파라미터를 관리하면 공백이나 특수문자 처리가 안전함
      cmd=(python3 -m torch.distributed.run --nproc_per_node=8 train_new.py
        --distributed
        --nusc-root ./
        --label-path hybrid_teacher_labels_final.json
        --batch-size 2             # GPU당 배치 사이즈 (Global Batch Size = 2 * 8 = 16)
        --val-split 0.05
        --val-batch-size 2
        --epochs 30
        --vision-backbone "$backbone"
      )

      # 3. Ablation 조건에 따른 동적 Flag 주입 (argparse action="store_true/false" 대응)
      if [[ "$cam_emb" == "true" ]]; then
        cmd+=(--use-camera-embedding)
      else
        cmd+=(--no-camera-embedding)
      fi

      if [[ "$geom" == "true" ]]; then
        cmd+=(--use-geometry-conditioning)
      else
        cmd+=(--no-geometry-conditioning)
      fi

      cmd+=(--output-dir "$OUT_DIR")

      # 4. 명령어 실행 및 로깅
      # 나중에 디버깅이나 재현을 위해 실제 실행되는 전체 명령어를 로그 최상단에 기록
      echo "${cmd[*]}" | tee "$LOGFILE"
      
      # 표준 출력(stdout)과 표준 에러(stderr)를 모두 로그 파일에 누적 기록(-a)하면서 콘솔에도 출력
      "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"

      echo ""
      echo "Finished experiment: $exp_name"
      echo "-------------------------------------------------------------"
      
      # (Optional) 실험 간 GPU 메모리 릴리즈를 확실히 하기 위해 짧은 sleep을 줄 수도 있습니다.
      # sleep 5 
    done
  done
 done
