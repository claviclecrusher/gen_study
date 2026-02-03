# Gen_Study: Generative Models Visualization

1D 신세틱 데이터를 사용한 생성 모델(Decoder, Autoencoder, VAE) 학습 및 시각화 프로젝트

## 프로젝트 개요

이 프로젝트는 매우 작은 1차원 신세틱 데이터(2-Gaussian mixture)를 생성하고, 세 가지 생성 모델을 학습하여 그 동작을 시각화합니다.

### 구현된 모델

1. **Non-identifiable Decoder**: 랜덤 노이즈를 입력받아 데이터를 생성하는 단순한 디코더 (mode collapse 현상 관찰)
2. **Autoencoder**: 데이터를 잠재 공간으로 인코딩하고 다시 복원하는 모델
3. **VAE (Variational Autoencoder)**: KL regularization을 통해 잠재 공간을 정규화하는 생성 모델

## 프로젝트 구조

```
Gen_Study/
├── data/
│   ├── __init__.py
│   └── synthetic.py          # 2-Gaussian mixture 데이터 생성
├── models/
│   ├── __init__.py
│   ├── base_mlp.py          # Base MLP 클래스
│   ├── decoder.py           # Non-identifiable Decoder
│   ├── autoencoder.py       # Autoencoder
│   └── vae.py               # VAE
├── training/
│   ├── __init__.py
│   ├── train_decoder.py     # Decoder 학습 스크립트
│   ├── train_ae.py          # Autoencoder 학습 스크립트
│   └── train_vae.py         # VAE 학습 스크립트
├── visualization/
│   ├── __init__.py
│   ├── config.py            # 색상 및 스타일 설정
│   ├── viz_decoder.py       # Decoder 시각화
│   ├── viz_ae.py            # Autoencoder 시각화
│   └── viz_vae.py           # VAE 시각화
├── outputs/                  # 학습된 모델 및 시각화 결과
├── main.py                   # 전체 실험 실행 스크립트
├── requirements.txt
└── README.md
```

## 설치 방법

### 1. Conda 환경 생성 및 활성화

```bash
# Gen_Study 디렉토리로 이동
cd /home/user/Desktop/Gen_Study

# Conda 환경 생성 (Python 3.10)
conda create -n gen_study python=3.10 -y

# 환경 활성화
conda activate gen_study
```

### 2. 패키지 설치

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 환경 비활성화 (작업 종료 시)

```bash
conda deactivate
```

## 사용 방법

**주의**: 모든 명령은 `gen_study` conda 환경이 활성화된 상태에서 실행해야 합니다.

```bash
# 환경 활성화 확인
conda activate gen_study
```

### 전체 실험 실행

```bash
python main.py
```

이 명령은 다음을 수행합니다:
1. 500개의 2-Gaussian mixture 샘플 생성
2. 네 가지 모델 각각 2000 에폭 학습 (Decoder, Autoencoder, VAE β=1.0, VAE β=10.0)
3. 학습된 모델 저장 (`outputs/` 디렉토리)
4. 각 모델의 시각화 결과 저장 (`outputs/` 디렉토리)

### 개별 모델 학습

```bash
# Decoder만 학습
python training/train_decoder.py

# Autoencoder만 학습
python training/train_ae.py

# VAE만 학습
python training/train_vae.py
```

### 데이터 생성 테스트

```bash
python data/synthetic.py
```

## 설정

### 데이터 설정
- **2-Gaussian mixture**
  - Gaussian 1: mean=-2.0, std=0.8 (70%)
  - Gaussian 2: mean=2.0, std=0.6 (30%)
- **샘플 수**: 500

### 모델 설정
- **MLP 구조**: [32, 64, 32] hidden layers
- **잠재 차원**: 1D
- **입출력 차원**: 1D

### 학습 설정
- **Optimizer**: Adam
- **Learning rate**: 1e-3
- **Batch size**: 64
- **Epochs**: 2000

### 시각화 색상 설정

색상은 [`visualization/config.py`](visualization/config.py)에서 중앙 관리됩니다:

```python
COLORS = {
    'prior_z': '#1f77b4',        # blue - 사전 분포 z
    'data_x': '#ff7f0e',         # orange - 데이터 x
    'latent_z_hat': '#2ca02c',   # green - 인코딩된 ẑ
    'output_x_hat': '#d62728',   # red - 복원된 x̂
    'coupling_train': '#7f7f7f', # gray - 학습 커플링
    'coupling_infer': '#9467bd', # purple - 추론 매핑
    'encode_line': '#8c564b',    # brown - 인코딩 선
    'decode_line': '#e377c2',    # pink - 디코딩 선
}
```

## 시각화 설명

각 시각화는 하나의 이미지에 두 공간을 나타냅니다:
- **왼쪽 패널**: 잠재/소스 공간 (z, ẑ)
- **오른쪽 패널**: 데이터/타겟 공간 (x, x̂)
- **연결선**: 매핑 관계 시각화

### 1. Decoder 시각화
- 학습 시 랜덤 커플링 (회색 선)
- 추론 시 매핑 (보라색 선)
- 출력이 데이터의 평균으로 수렴하는 mode collapse 관찰 가능

### 2. Autoencoder 시각화
- 인코딩: x → ẑ (갈색 선)
- 디코딩: ẑ → x̂ (분홍색 선)
- 잠재 공간 ẑ가 사전 분포와 다를 수 있음

### 3. VAE 시각화 (β=1.0)
- Autoencoder와 동일한 구조
- KL regularization으로 ẑ가 사전 분포 N(0,1)에 가까워짐
- 재구성 품질과 정규화 사이의 트레이드오프 관찰 가능

### 4. VAE with High Beta 시각화 (β=10.0)
- 높은 β 값으로 KL regularization을 더욱 강화
- ẑ가 사전 분포 N(0,1)에 더욱 가까워짐
- β=1.0과 비교하여 정규화 효과가 더 강함을 관찰 가능
- 재구성 품질은 다소 저하될 수 있으나 생성 품질은 향상

## 출력 파일

실행 후 `outputs/` 디렉토리에 다음 파일들이 생성됩니다:
- `nid_decoder_model.pt` - 학습된 Non-identifiable Decoder 모델
- `nid_decoder_visualization.png` - Non-identifiable Decoder 시각화
- `autoencoder_model.pt` - 학습된 Autoencoder 모델
- `autoencoder_visualization.png` - Autoencoder 시각화
- `vae_beta1.0_model.pt` - 학습된 VAE 모델 (β=1.0)
- `vae_beta1.0_visualization.png` - VAE 시각화 (β=1.0)
- `vae_beta10.0_model.pt` - 학습된 VAE 모델 (β=10.0)
- `vae_beta10.0_visualization.png` - VAE 시각화 (β=10.0)

## 요구 사항

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.
