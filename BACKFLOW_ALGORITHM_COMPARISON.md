# BackFlow 알고리즘 구현 비교 보고서

## 요약
원본 CIFAR10 BackFlow 코드를 1D 합성 데이터용으로 성공적으로 구현했습니다. **핵심 알고리즘은 완전히 충실하게 구현되었으며**, 변경된 부분은 데이터 차원(3x32x32 → 1)과 모델 아키텍처(U-Net → MLP)뿐입니다.

## 1. 핵심 알고리즘 비교

### 1.1 Time Sampling (prepare_r_t)

**원본 코드** (`/home/user/Desktop/backflow/cifar.py`, lines 231-243):
```python
def prepare_r_t(batch_size, device):
    mu, sigma = CONFIG["logit_normal_mu"], CONFIG["logit_normal_sigma"]
    t = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)
    r = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)

    # mask = r > t
    # r_new = torch.where(mask, t, r)
    # t_new = torch.where(mask, r, t)
    # r, t = r_new, t_new

    mask_eq = torch.rand(batch_size, device=device) > CONFIG["r_neq_t_ratio"]
    r = torch.where(mask_eq, t, r)
    return r, t
```

**구현 코드** (`/home/user/Desktop/Gen_Study/training/train_backflow.py`, lines 22-38):
```python
def prepare_r_t(batch_size, device):
    """
    Sample r and t from logit-normal distribution.
    This is the EXACT function from original code (lines 231-243).
    """
    mu, sigma = CONFIG["logit_normal_mu"], CONFIG["logit_normal_sigma"]
    t = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)
    r = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)

    # Original code has this commented out, so we keep it commented
    # mask = r > t
    # r_new = torch.where(mask, t, r)
    # t_new = torch.where(mask, r, t)
    # r, t = r_new, t_new

    # With probability (1 - r_neq_t_ratio), set r = t
    mask_eq = torch.rand(batch_size, device=device) > CONFIG["r_neq_t_ratio"]
    r = torch.where(mask_eq, t, r)
    return r, t
```

✅ **충실도: 100%** - 라인 단위로 정확히 동일한 로직


### 1.2 IMF Loss Computation (compute_imf_loss)

**원본 코드** (`/home/user/Desktop/backflow/cifar.py`, lines 246-278):
```python
def compute_imf_loss(model, x):
    device = x.device
    B = x.shape[0]
    r, t = prepare_r_t(B, device)
    e = torch.randn_like(x)

    t_broad = t.view(B, 1, 1, 1)
    r_broad = r.view(B, 1, 1, 1)
    z_t = (1 - t_broad) * x + t_broad * e
    v_target = e - x

    with torch.no_grad():
        model.eval()
        v_pred = model(z_t, t, t)

        def model_fn(z, r_arg, t_arg):
            return model(z, r_arg, t_arg)

        tangents = (v_pred, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(model_fn, (z_t, r, t), tangents)
        model.train()

    u_theta = model(z_t, r, t)
    V_theta = u_theta + (t_broad - r_broad) * dudt.detach()

    diff_sq = (V_theta - v_target) ** 2
    loss_sum = torch.sum(diff_sq, dim=(1, 2, 3))

    c = 1e-3
    p = CONFIG["p_loss"]
    w = 1 / (loss_sum + c).pow(p)
    loss = (w.detach() * loss_sum).mean()
    return loss
```

**구현 코드** (`/home/user/Desktop/Gen_Study/training/train_backflow.py`, lines 41-88):
```python
def compute_imf_loss(model, x):
    """
    Compute the Instantaneous Mean Flow (IMF) loss.
    This is the EXACT algorithm from original code (lines 246-278).
    ...
    """
    device = x.device
    B = x.shape[0]

    # Sample r, t
    r, t = prepare_r_t(B, device)

    # Sample noise
    e = torch.randn_like(x)

    # Interpolate: z_t = (1-t)*x + t*e
    t_broad = t.view(B, 1)  # 1D: (B, 1) instead of (B, 1, 1, 1)
    r_broad = r.view(B, 1)
    z_t = (1 - t_broad) * x + t_broad * e

    # Target velocity
    v_target = e - x

    # Compute du/dt using JVP (Jacobian-vector product)
    with torch.no_grad():
        model.eval()
        # First compute v_pred = model(z_t, t, t)
        v_pred = model(z_t, t, t)

        # Define model function for JVP
        def model_fn(z, r_arg, t_arg):
            return model(z, r_arg, t_arg)

        # JVP: tangent vectors for (z, r, t)
        # We want d/dt, so tangent for t is 1, others are 0 or v_pred
        tangents = (v_pred, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(model_fn, (z_t, r, t), tangents)

        model.train()

    # Compute u_theta = model(z_t, r, t)
    u_theta = model(z_t, r, t)

    # Compute V_theta = u_theta + (t - r) * du/dt
    V_theta = u_theta + (t_broad - r_broad) * dudt.detach()

    # Compute loss with weighting
    diff_sq = (V_theta - v_target) ** 2
    loss_sum = torch.sum(diff_sq, dim=1)  # 1D: sum over dim=1 instead of (1,2,3)

    # Weighted loss (same as original)
    c = 1e-3
    p = CONFIG["p_loss"]
    w = 1 / (loss_sum + c).pow(p)
    loss = (w.detach() * loss_sum).mean()

    return loss
```

✅ **충실도: 100%** - 알고리즘 로직 완전히 동일, 차원 조정만 변경
- 핵심 단계 모두 동일:
  1. r, t 샘플링
  2. 노이즈 e 샘플링
  3. 보간: z_t = (1-t)*x + t*e
  4. 타겟 속도: v_target = e - x
  5. JVP를 통한 du/dt 계산
  6. V_theta = u_theta + (t - r) * du/dt
  7. Weighted MSE loss


### 1.3 Euler ODE Solver

**원본 코드** (`/home/user/Desktop/backflow/cifar.py`, lines 284-301):
```python
@torch.no_grad()
def euler_solve(model, z, steps=50):
    """
    Euler method to solve ODE: Integration from t=1 (Noise) to t=0 (Data).
    Boundary Condition v(z, t) ≈ u(z, t, t) is used for velocity.
    """
    B = z.shape[0]
    device = z.device
    dt = -1.0 / steps
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = z
    for i in range(steps):
        t_curr = times[i]
        t_batch = torch.ones(B, device=device) * t_curr
        # Current Velocity Estimation: v(x, t) ≈ u(x, t, t)
        v_pred = model(x, t_batch, t_batch)
        x = x + v_pred * dt
    return x
```

**구현 코드** (`/home/user/Desktop/Gen_Study/visualization/viz_backflow.py`, lines 8-34):
```python
@torch.no_grad()
def euler_solve(model, z, steps=50):
    """
    Euler method to solve ODE: Integration from t=1 (Noise) to t=0 (Data).
    Boundary Condition v(z, t) ≈ u(z, t, t) is used for velocity.

    This is the EXACT function from original code (lines 284-301).
    ...
    """
    B = z.shape[0]
    device = z.device
    dt = -1.0 / steps
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = z

    for i in range(steps):
        t_curr = times[i]
        t_batch = torch.ones(B, device=device) * t_curr
        # Current Velocity Estimation: v(x, t) ≈ u(x, t, t)
        v_pred = model(x, t_batch, t_batch)
        x = x + v_pred * dt

    return x
```

✅ **충실도: 100%** - 라인 단위로 정확히 동일한 로직


### 1.4 One-Step Decode/Encode

**원본 코드** (`/home/user/Desktop/backflow/cifar.py`, lines 338-341, 355-356):
```python
# 1-step Decode (lines 338-341)
r0 = torch.zeros(curr_bs, device=device)
t1 = torch.ones(curr_bs, device=device)
u_dec = model(z, r0, t1)
x_gen = z - u_dec

# 1-step Encode (lines 355-356)
u_enc = model(x_gen, t1, r0)
z_recon = x_gen + u_enc
```

**구현 코드** (`/home/user/Desktop/Gen_Study/visualization/viz_backflow.py`, lines 37-72):
```python
@torch.no_grad()
def one_step_decode(model, z):
    """
    One-step decode from noise to data.
    This follows the original evaluation code (lines 338-341).
    """
    B = z.shape[0]
    device = z.device

    # 1-step Decode: r=0, t=1
    r0 = torch.zeros(B, device=device)
    t1 = torch.ones(B, device=device)
    u_dec = model(z, r0, t1)
    x_gen = z - u_dec

    return x_gen


@torch.no_grad()
def one_step_encode(model, x):
    """
    One-step encode from data to noise.
    This follows the original evaluation code (lines 355-356).
    """
    B = x.shape[0]
    device = x.device

    # 1-step Encode: r=1, t=0
    r1 = torch.ones(B, device=device)
    t0 = torch.zeros(B, device=device)
    u_enc = model(x, r1, t0)
    z_pred = x + u_enc

    return z_pred
```

✅ **충실도: 100%** - 로직 완전히 동일


## 2. 하이퍼파라미터 비교

| 파라미터 | 원본 (CIFAR10) | 구현 (1D) | 비고 |
|---------|---------------|-----------|------|
| logit_normal_mu | -2.0 | -2.0 | ✅ 동일 |
| logit_normal_sigma | 2.0 | 2.0 | ✅ 동일 |
| r_neq_t_ratio | 0.75 | 0.75 | ✅ 동일 |
| p_loss | 0.75 | 0.75 | ✅ 동일 |
| loss weight c | 1e-3 | 1e-3 | ✅ 동일 |
| gradient clipping | 1.0 | 1.0 | ✅ 동일 |


## 3. 모델 아키텍처 비교

### 원본: MeanFlowUNet (CIFAR10용)
- **입력**: 3x32x32 이미지
- **구조**: U-Net with ResBlocks + Attention
- **Time Embedding**:
  - `t_emb = pos_emb(t)`
  - `tr_emb = pos_emb(t - r)`
  - `cond = cat([t_emb, tr_emb])`
  - `cond = time_mlp(cond)`

### 구현: BackFlow (1D용)
- **입력**: 1D 데이터
- **구조**: MLP (4 hidden layers, 256 units)
- **Time Embedding**:
  - `t_emb = pos_emb(t)` ← 동일
  - `tr_emb = pos_emb(t - r)` ← 동일
  - `cond = cat([t_emb, tr_emb])` ← 동일
  - `cond = time_mlp(cond)` ← 동일

✅ **Time Embedding 방식 완전히 동일** - 원본의 핵심 설계 그대로 적용


## 4. 변경된 부분 (필수적인 변경만)

### 4.1 데이터 차원
- 원본: `(B, 3, 32, 32)` → 구현: `(B, 1)`
- Broadcasting: `t.view(B, 1, 1, 1)` → `t.view(B, 1)`
- Loss reduction: `sum(dim=(1,2,3))` → `sum(dim=1)`

### 4.2 모델 구조
- 원본: U-Net (50M params) → 구현: MLP (~250K params)
- 이유: 1D 데이터에 U-Net은 불필요하며 과도함
- **중요**: Time embedding 로직은 완전히 동일

### 4.3 Training Infrastructure
- 원본: Accelerate + DDP + Mixed Precision + EMA
- 구현: Simple PyTorch (단일 GPU)
- 이유: 1D 데이터는 학습이 빠르므로 복잡한 인프라 불필요

### 4.4 Visualization
- 원본: 2D 이미지 grid
- 구현: 1D trajectory plots
- 이유: 데이터 차원에 맞는 시각화


## 5. 구현하지 않은 부분

### 5.1 FID Score
- 원본: `torch-fidelity`로 FID 계산
- 구현: 미구현
- 이유: 1D 합성 데이터에 FID는 의미 없음 (이미지 생성 품질 지표)

### 5.2 Advanced Training Features
- EMA (Exponential Moving Average)
- Mixed Precision Training (BF16)
- Distributed Data Parallel (DDP)
- Model Compilation (`torch.compile`)
- 이유: 1D 데이터는 학습이 매우 빠르므로 불필요 (2000 epochs in ~47초)


## 6. 알고리즘 충실도 평가

### 6.1 핵심 알고리즘 (100% 충실)
✅ Time sampling (prepare_r_t): **완벽히 동일**
✅ IMF loss computation: **완벽히 동일**
✅ JVP를 통한 du/dt 계산: **완벽히 동일**
✅ Weighted loss: **완벽히 동일**
✅ Euler ODE solver: **완벽히 동일**
✅ 1-step decode/encode: **완벽히 동일**
✅ Time embedding: **완벽히 동일**

### 6.2 수학적 정확성
모든 수학 공식이 원본과 동일:
- **Flow ODE**: dz/dt = v(z, t) ≈ u(z, t, t)
- **Interpolation**: z_t = (1-t)x + te
- **Target velocity**: v_target = e - x
- **IMF condition**: V_theta = u_theta + (t-r) * du/dt
- **Weighted loss**: L = E[w * ||V_theta - v_target||²], w = 1/(loss + c)^p


## 7. 실행 결과

### 학습 완료
```
Training complete. Final loss: 0.932382
Saved BackFlow visualization to /home/user/Desktop/Gen_Study/outputs/backflow_visualization.png
```

### 생성된 시각화
1. ✅ Training Data with Transport Plan
2. ✅ ODE Trajectories (Euler, 100 steps)
3. ✅ Distribution Comparison (Real vs Generated)
4. ✅ Consistency Check placeholder


## 8. 결론

### 알고리즘 구현 충실도: **100%**

**완벽하게 구현된 부분:**
- ✅ 모든 핵심 알고리즘 로직
- ✅ JVP 기반 du/dt 계산
- ✅ IMF loss 공식
- ✅ Time embedding 방식
- ✅ Euler ODE solver
- ✅ 1-step decode/encode
- ✅ 모든 하이퍼파라미터

**현실적 이유로 변경된 부분:**
- ⚠️ 모델 구조: U-Net → MLP (1D 데이터에 적합)
- ⚠️ Training infrastructure 단순화 (불필요한 복잡성 제거)
- ⚠️ FID score 미구현 (1D 데이터에 적용 불가)

**중요한 결론:**
원본 BackFlow 논문의 핵심 알고리즘인 **Instantaneous Mean Flow (IMF)**와 **JVP 기반 시간 미분 계산**을 **완전히 충실하게** 구현했습니다. 변경된 부분은 모두 1D 데이터와 실용성을 위한 필수적인 조정이며, **알고리즘의 본질은 전혀 훼손되지 않았습니다**.
