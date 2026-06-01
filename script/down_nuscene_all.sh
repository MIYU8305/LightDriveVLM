#!/bin/bash
# 이 스크립트는 bash 셸을 사용하여 실행됨을 선언합니다.

# ==============================================================================
# 1. 데이터 저장 경로 설정 및 디렉토리 생성
# ==============================================================================
# 홈 디렉토리(~) 아래에 nuscenes 폴더를 데이터 저장 최상위 경로로 지정합니다.
# (수정됨: 기존 $HOME./nuscenes 에서 슬래시(/)를 추가하여 정상적인 경로로 수정했습니다)
DATA_ROOT="${HOME}/nuscenes"

# 설정한 경로에 폴더를 생성합니다. 
# -p 옵션: 이미 폴더가 존재하면 에러를 무시하고, 상위 폴더가 없으면 함께 생성합니다.
mkdir -p ${DATA_ROOT}

# 생성한 데이터 저장 폴더로 이동합니다.
cd ${DATA_ROOT}

echo "===== nuScenes download start ====="

# ==============================================================================
# 2. 데이터셋 다운로드 (wget 활용)
# -c 옵션: 다운로드가 중간에 끊겼을 경우 처음부터 다시 받지 않고 이어서 다운로드(continue)합니다.
# -O 옵션: 다운로드된 파일을 저장할 이름을 지정합니다.
# ==============================================================================

# [Mini dataset]
# 전체 데이터셋을 다운로드하기 전에 코드 테스트 용도로 쓸 수 있는 소규모 샘플 데이터셋입니다.
wget -c "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz" -O v1.0-mini.tgz

# [Trainval metadata]
# 이미지나 라이다(LiDAR) 원본 데이터가 아닌, 3D 바운딩 박스 라벨, 카메라 파라미터(캘리브레이션) 등이 포함된 메타데이터입니다.
wget -c "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz" -O v1.0-trainval_meta.tgz

# [Trainval blobs (센서 원본 데이터)]
# 카메라 이미지, 라이다, 레이더 등 실제 무거운 센서 데이터들입니다.
# 파일 용량이 매우 커서 10개의 파트로 나뉘어 있습니다.
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz" -O blobs_trainval01.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz" -O blobs_trainval02.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval03_blobs.tgz" -O blobs_trainval03.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz" -O blobs_trainval04.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval05_blobs.tgz" -O blobs_trainval05.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval06_blobs.tgz" -O blobs_trainval06.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval07_blobs.tgz" -O blobs_trainval07.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz" -O blobs_trainval08.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz" -O blobs_trainval09.tgz
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz" -O blobs_trainval10.tgz

echo "===== extract ====="

# ==============================================================================
# 3. 압축 해제
# ==============================================================================
# 현재 폴더(DATA_ROOT)에 있는 확장자가 .tgz인 모든 파일을 순회(for문)하며 압축을 풉니다.
for f in *.tgz
do
    echo "extracting $f"
    # tar 옵션 설명:
    # -x : 압축 해제 (eXtract)
    # -v : 압축 해제 진행 과정을 화면에 출력 (Verbose)
    # -z : gzip으로 압축된 파일을 처리 (gZip)
    # -f : 대상 파일 이름을 지정 (File)
    tar -xvzf $f
done

echo "===== done ====="