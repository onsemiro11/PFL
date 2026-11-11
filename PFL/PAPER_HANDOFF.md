# PFL — Current Model and Paper Handoff

이 문서는 현재 논문 모델 후보인 **two-stage PFL(Stage2 PFL)**을 논문 작성자가
코드와 일치하게 기술할 수 있도록 정리한 handoff다. 모델 본체는 relational
ego-conditioned encoder와 conditional rectified-flow decoder이며, deterministic mean
head는 encoder warm-start와 보조 학습에만 사용한다. 실제 추론 trajectory는 mean head가
아니라 flow sample로만 생성한다.

`fair_*_pfl_full_seed*` 실행은 two-stage 학습의 필요성을 판단하기 위해 처음부터
flow-only로 학습한 비교군이다. 논문용 최종 후보는 Stage2 PFL이며 NGSIM/highD 모두
고정 validation 기준 seed 0, 1, 2 재현과 전체 test 평가가 완료됐다. 최종 Stage2와
relation을 제거한 모델을 비교하는 `Stage2_REL0`도 동일한 warm-start, 보조 loss와 평가
조건으로 3-seed 학습·전체 test 평가·bootstrap 분석을 완료했다. 전체 파이프라인은
2026-10-02 16:40 KST에 완료됐으며 상태는 `experiments/stage2_rel0_status.json`에 기록돼
있다.

모델 내부 좌표 단위는 feet이며, trajectory channel 순서는
`(lateral, longitudinal)`이다. 저장되는 평가 지표는 `0.3048`을 곱해 metre로 통일한다.

---

## 1. 논문에서 검증하는 가설

현재 실험은 다음 가설을 검증하도록 구성되어 있다.

> Relational ego-conditioning은 ego plan과 각 주변 target 차량 사이의 미래
> 상대관계를 명시적으로 표현해 target trajectory prediction을 개선한다. PFL은
> 전체 미래 trajectory 분포를 flow matching으로 직접 생성하고, 실제 배포 가능한
> GT-free 선택에서 EPN보다 낮은 오차를 얻을 수 있는지를 검증한다.

EPN의 endpoint oracle이 더 좋더라도 PFL의 GT-free sample mean/medoid가 더 좋을 수
있다. 이 경우 논문 주장은 "oracle 동등"이 아니라 **EPN은 oracle coverage에서 우세하고,
PFL은 GT 없이 사용할 수 있는 prediction에서 우세하다**로 기술한다. EPN 대비 성능은
완료된 Stage2 3-seed 결과로 판단하고, relational encoder의 독립 효과는 동일한 two-stage
학습 조건으로 완료한 `Stage2_REL0` 3-seed 평가를 근거로 판단한다.

---

## 2. 현재 코드 구조

| 파일 | 역할 |
|---|---|
| `PFL/data.py` | NGSIM/highD 공통 `.mat` loader, target 단위 batching |
| `PFL/encoder.py` | CV baseline, temporal attention, social pooling, relational encoder |
| `PFL/vf.py` | conditional rectified-flow vector field와 1D U-Net |
| `PFL/model.py` | `PFLNet`, residual normalization, loss, sampling |
| `PFL/train.py` | mean warm-start/flow 학습, EMA, validation, checkpoint 저장 |
| `experiments/common.py` | 공통 metric, GT-free/oracle selector, 차량 단위 집계 |
| `experiments/scripts/evaluate_model.py` | CV/EPN/PFL 통합 평가 |
| `experiments/scripts/summarize_results.py` | seed 요약, paired bootstrap, equivalence, A–D 판정 |
| `experiments/configs/pfl_*.yaml` | PLAN=0, REL=0, Full 사전 지정 설정 |

`PFLNet`은 `PFLEncoder`가 만든 112차원 context를 `VectorField`에 조건으로 전달한다.
Ego trajectory는 조건 입력이고 예측 대상이 아니다. 생성 대상은 ego grid 안의 주변
target 차량 future trajectory뿐이다.

현재 Stage2 PFL의 전체 구조는 다음과 같다.

```text
target history ── TemporalAttentionEncoder ───────────────┐
neighbour histories ─ encoder + 25×5 social grid ─ Conv2d ├─ concat ─ c_i (112-D)
ego internal plan ─ plan encoder/grid + plan token ───────┤
ego plan vs target CV ─ RelationalEgoEncoder ─────────────┘
                                                            │
Gaussian noise x_0 + flow time u ────────────────────────────┤
                                                            ▼
                                      conditional 1D U-Net vector field
                                                            │ Euler ODE
                                                            ▼
                               25-step CV-residual trajectory samples
                                                            │ de-normalise + target CV
                                                            ▼
                                      target future trajectory candidates
```

학습 시에만 같은 112차원 context에서 maneuver head와 deterministic mean head가
분기한다. Mean head는 encoder representation을 먼저 안정화하고 Stage 2에서 작은 보조
gradient를 제공하지만, 평가와 배포에서는 그 출력을 사용하지 않는다.

```text
training only: c_i ─ maneuver heads (3 lateral + 2 longitudinal classes)
               c_i ─ MLP mean head (25×2 deterministic residual)

inference:     c_i ─ vector field ─ flow samples
```

---

## 3. 데이터 표현

### 3.1 시간 해상도와 shape

NGSIM 원본은 10 Hz다. highD는 preprocessing에서 25 Hz를 10 Hz로 변환한 뒤 같은
loader를 사용한다. Loader 설정은 다음과 같다.

```text
t_h = 30, t_f = 50, d_s = 2
history = 16 states, current 포함, 3 s, 0.2 s 간격
future  = 25 states, 5 s, 0.2 s 간격
ego plan = 25 states, 5 s
social grid = 25 longitudinal × 5 lateral cells
```

한 scene은 `(dataset segment, ego vehicle, frame)`으로 식별된다. Ego grid의 모든
occupied vehicle이 별도의 예측 target이 되며, 여러 scene의 target들은 collate에서
하나의 batch axis로 합쳐진다.

주요 tensor shape은 다음과 같다. 여기서 `B`는 scene 수가 아니라 batch 안의 전체
target 수다.

| tensor | shape |
|---|---|
| target history | `[16, B, 2]` |
| target future | `[25, B, 2]` |
| future mask | `[25, B, 2]` |
| ego plan relative to target | `[25, B, 2]` |
| neighbour history | `[16, N_nbrs, 2]` |
| neighbour grid mask | `[B, 5, 25, 64]` |

Target history와 future는 target의 현재 위치를 원점으로 표현한다. 따라서
`target_history[-1] = (0, 0)`이고 future는 현재 위치로부터의 누적 변위다.

### 3.2 Ego-plan 입력의 중요한 한계

현재 `.mat` 파일에는 실제 온라인 planner가 출력한 trajectory가 없다.
`getPlanFuture`는 **기록된 ego 차량의 실제 future trajectory**를 target 현재 위치에
대한 상대좌표로 만든다. Validation/test에서는 이 trajectory의 waypoint를 quintic
spline으로 fitting한다.

따라서 현재 입력은 `executed-future plan proxy`이며 실제 배포 가능한 planner signal로
해석하면 안 된다. Candidate 선택이 GT-free여도 ego-plan 입력 자체에는 기록된 미래가
사용된다. 논문에는 다음을 명확히 구분해야 한다.

- 현재 NGSIM/highD 실험: recorded ego future를 plan proxy로 사용한 통제 실험
- 실제 배포 조건: perception 시점에 ego 내부 planner가 출력한 trajectory를 사용

실차 배포 성능을 주장하려면 동일 tensor contract에 실제 planner output을 넣고 다시
평가해야 한다. 주변 target의 실제 future는 encoder 입력에 사용하지 않는다.

`getPlanFuture`는 호환성 때문에 sequence를 time-reversed 형태로 반환한다. Grid plan
branch와 plan token은 기존 표현을 사용하고, relational branch는 `flip`으로 다시
정방향 정렬한 뒤 target CV와 같은 horizon끼리 비교한다.

---

## 4. Constant-velocity reference

Target CV는 마지막 세 개의 0.2초 interval에서 평균 per-step velocity를 계산한다.

```python
h = target_history.permute(1, 0, 2)
v0 = (h[:, -1] - h[:, -4]) / 3
cv[t] = (t + 1) * v0
```

기본 `cv_gate=[0,1]`이므로 longitudinal velocity만 extrapolation하고 lateral reference는
0으로 둔다. `--cv_lateral`을 명시할 때만 두 축을 모두 사용한다. Fair config에서는
기본 longitudinal-only CV를 사용한다.

PFL은 raw future가 아니라 다음 standardised CV residual을 생성한다.

```text
x1 = (Y_target - Y_CV - residual_mean) / residual_std
```

`residual_mean`과 `residual_std`는 training set 300 batch에서 horizon/axis별로 계산하며
checkpoint buffer에 저장한다. 최소 standard deviation은 0.05 ft다.

---

## 5. PFL encoder

### 5.1 Temporal encoder

Target와 neighbour motion은 같은 `TemporalAttentionEncoder`를 공유하고 ego plan은
별도 instance를 사용한다. Motion 입력은 `order=3`에서 다음 5개 channel이다.

```text
[lateral position, longitudinal position, velocity, acceleration, TTC]
```

각 temporal encoder는 다음 순서로 64차원 representation을 만든다.

```text
Linear → GELU → Linear
+ learned positional embedding
self-attention, 4 heads
residual LayerNorm
64→128→64 FFN
learned-query attention pooling
```

Target history representation은 `Linear(64→32)`를 거쳐 target dynamics embedding이
된다.

### 5.2 Grid social pooling

Neighbour representation은 target 중심의 `25×5` grid에 scatter한 뒤 다음 social
convolution으로 처리한다.

```text
Conv2d(64→64, 3×3)
MaxPool2d(3×3, stride=2)
Conv2d(64→16, 3×1)
```

Ego plan을 사용하는 variant는 별도 동일 형태의 plan convolution branch를 사용하고
neighbour branch와 합친다. Plan 사용 여부와 관계없이 pooling 결과는 80차원이다.

### 5.3 Social cross-attention

기본 설정에서 target history representation이 query가 되고, neighbour grid token과
ego-plan token이 key/value가 된다. PLAN=0에서는 plan token이 없다. 출력은 32차원
target dynamics embedding에 residual addition된다.

Stage2 Full의 `use_attention=true`이므로 이 social cross-attention은 실제로 학습과
추론에 사용된다. 뒤에서 설명하는 `relation_cross_attention=false`는 이 모듈을 끈다는
뜻이 아니다. `relation_cross_attention`은 25시점 relation token을 flow U-Net의 각
residual block에 주입하는 별도의 실험 adapter다.

마찬가지로 `temporal_relational=false`는 위 5.1의 temporal self-attention을 끄는 설정이
아니다. Target/neighbour history와 ego plan을 처리하는 `TemporalAttentionEncoder`는
Stage2 Full에서 항상 사용된다. 이 flag가 끄는 것은 relation sequence를 25시점 token
형태로 flow decoder에 직접 넘기는 선택적 TemporalREL 경로뿐이다.

### 5.4 Relational ego-conditioning

Full variant는 정방향으로 정렬된 ego plan `P_t`와 target CV rollout `C_t`를 같은
horizon에서 비교한다. 좌표축은 lateral/longitudinal road coordinate다.

```text
relative position: P_t - C_t
ego plan velocity: P_t - P_(t-1)
target CV velocity: C_t - C_(t-1)
relative velocity: ego velocity - target CV velocity
ego/CV heading: atan2(lateral velocity, longitudinal velocity)
Δψ: wrap(ego heading - CV heading)

r_t = [Δs, Δd, Δv_parallel, Δv_perp, Δψ]
```

첫 velocity는 첫 available difference를 복제한다. Feature scale은 다음과 같다.

```text
[50 ft, 12 ft, 5 ft/step, 1 ft/step, 1 rad]
```

Relation sequence encoder는 다음 구조다.

```text
Linear(5→64) → LayerNorm → GELU
GRU(64→64)
MLP(64→64→32)
target_dynamics = target_dynamics + relation_output
```

MLP의 마지막 `Linear(64→32)` weight와 bias는 0으로 초기화한다. 따라서 초기 forward는
relation branch를 추가하기 전 모델과 같고, relation signal은 학습으로 필요한 만큼만
증가한다.

최종 context는 다음과 같다.

```text
social context 80 + target dynamics 32 = 112 dimensions
```

---

## 6. Ablation variants

세 variant는 같은 dataset split, flow decoder, context width, optimizer, LR schedule,
batch size, validation rule, sampling 설정을 사용한다.

| Variant | Grid plan | Plan token | Relation encoder | Context |
|---|---:|---:|---:|---:|
| PLAN=0 | off | off | off | 112 |
| REL=0 | on | on | off | 112 |
| Full | on | on | on | 112 |

논문에서 사용하는 **최종 제안 모델은 Stage2 Full**이다. 여기서 `Full`은 grid-plan,
plan-token과 기존 relational encoder를 모두 사용하는 핵심 PFL 구성을 뜻한다.
`Stage2_REL0`은 이 최종 모델을 대체하는 후보가 아니라, Full에서 relational encoder만
제거해 그 기여도를 공정하게 측정하는 ablation이다. 따라서 현재 REL=0 학습 결과가
나오더라도 최종 모델의 명칭과 기준 checkpoint는 Stage2 Full로 유지한다.

`Full`이라는 이름은 코드에 남아 있는 모든 선택적 실험 flag를 켰다는 뜻이 아니다.
TemporalREL, relation cross-attention과 mode conditioning은 Full을 정의한 뒤 성능 향상
가능성을 시험하기 위해 별도로 추가한 opt-in pilot 모듈이다.

현재 수행 중인 Stage2 REL=0과 최종 Stage2 Full의 실제 경로 차이는 다음 하나로
제한한다.

| 구성 요소 | Stage2 REL=0 | Stage2 Full |
|---|---:|---:|
| Target/neighbour/plan temporal self-attention | 사용 | 사용 |
| Grid social pooling | 사용 | 사용 |
| Ego-plan grid와 plan token | 사용 | 사용 |
| Target-to-social cross-attention | 사용 | 사용 |
| 32차원 target dynamics embedding | 사용 | 사용 |
| GRU relational ego encoder | 제거 | 사용 |
| Relation output의 target dynamics residual addition | 없음 | 사용 |
| Conditional rectified-flow decoder | 동일 | 동일 |
| Stage 1 deterministic mean warm-start | 사용 | 사용 |
| Stage 2 mean/maneuver auxiliary loss | 사용 | 사용 |
| Validation, K/NFE와 checkpoint 선택 | 동일 | 동일 |

CLI의 `--use_relational`은 `argparse`에서 `action="store_false"`로 정의된 disable flag다.
따라서 Stage2 REL=0 실행 명령에 이 문자열이 나타나는 것이 정상이며, 저장된
`args.json`에는 `use_relational=false`로 기록된다. Full 실행에는 이 disable flag를 넘기지
않아 기본값 `use_relational=true`를 사용한다.

파라미터 수는 다음과 같다.

| Variant | Parameters |
|---|---:|
| PLAN=0 | 3,138,711 |
| REL=0 | 3,178,727 |
| Full | 3,210,439 |
| REL=0 + training-only mean head (Stage2_REL0) | 3,286,297 |
| Full + training-only mean head (Stage2) | 3,318,009 |

같은 head 구성끼리 비교하면 Full과 REL=0의 차이는 항상 31,712개다. Mean head를 제외한
기본 모델 기준으로 Full의 약 0.99%이며, Stage2 비교에서의 차이는 relational encoder
자체에서만 생긴다. 주 ablation은 relation 정보를 깨끗하게 제거한 REL=0을 사용한다.
선택적 `relation_control=target_cv`는 relation encoder와 정확히 같은 구조를 두되 ego
path를 zero로 만들어 Full과 동일한 3,210,439개 파라미터를 갖게 한다. 이는 추가 capacity
확인용이며 주 실험 matrix에는 포함하지 않는다.

Planning gain과 relational gain은 오차가 낮을수록 좋다는 방향으로 정의한다.

```text
Planning gain   = Metric(PLAN=0) - Metric(REL=0)
Relational gain = Metric(REL=0)  - Metric(Full)
```

양수이면 뒤의 모델이 개선된 것이다.

---

## 7. Flow-matching decoder

### 7.1 생성 대상과 "직접 생성"의 정확한 의미

현재 Stage2 PFL의 **flow 생성 경로**에는 discrete maneuver selector, endpoint generator,
endpoint-conditioned decoder가 없다. 생성 대상은 하나의 endpoint나 시점별 increment가
아니라 다음 전체 tensor다.

```text
x1 ∈ R^(25×2) = 25개 미래 시점의 standardised target-CV residual
```

따라서 "전체 trajectory를 직접 생성한다"는 말은 raw absolute coordinate를 생성한다는
뜻이 아니다. Target의 longitudinal-only CV rollout을 먼저 deterministic reference로
계산하고, flow가 그 기준선에 대한 25시점 residual을 공동 생성한다는 뜻이다. 생성 후
residual statistics를 복원하고 CV를 더해 target-relative future coordinate를 얻는다.

```text
Y_future = generated_residual × residual_std + residual_mean + Y_CV
```

25개 시점은 1D U-Net의 horizon dimension으로 함께 입력되고 downsampling, dilated
convolution과 skip connection을 통해 서로 정보를 교환한다. 시점 1을 생성한 뒤 시점 2를
생성하는 autoregressive 구조가 아니며, 마지막 endpoint를 만든 뒤 중간 경로를 보간하는
구조도 아니다.

현재 최종 checkpoint의 관련 flag는 다음과 같다.

```text
mode_conditioned = false
intent_condition = false
recon_weight = 0
temporal_relational = false
relation_cross_attention = false
```

`config.py`에 남은 destination-size 인자와 `model.py`의 endpoint selector 함수는 EPN 호환
및 평가용 코드이며, PFL vector field 앞에 endpoint network가 존재한다는 의미가 아니다.

Source는 target별 독립 표준 Gaussian noise다.

```text
x0 ~ N(0, I)
t  ~ Uniform(0,1)
xt = (1-t)x0 + t x1
v* = x1 - x0
L_FM = masked mean ||v_theta(xt,t,c) - v*||²
```

`VectorField`는 trajectory horizon을 spatial axis로 보는 conditional 1D U-Net이다.
Generation time `t∈[0,1]`과 trajectory time 25 steps는 서로 다른 tensor dimension이다.
모델은 horizon을 autoregressive하게 생성하지 않고 `[B,25,2]` 전체를 동시에 갱신한다.

Context와 sinusoidal flow-time embedding은 `CondEncoder`에서 다음처럼 결합된다.

```text
u ∈ [0,1] → sinusoidal embedding → MLP → 128-D
context 112-D → Linear → 64-D
concat(128,64) → MLP → condition 128-D
```

Context는 별도로 `Linear(112→16)`을 거쳐 25개 horizon에 반복되고, noisy trajectory
`[B,25,2]`와 channel 방향으로 연결된다. U-Net channel은 `[64,128,256]`, encoder
dilation은 `[1,2,4]`다. 각 level에 residual block 두 개가 있고 두 번의 stride-2
downsampling, bottleneck block 두 개, skip connection을 사용한 대칭 decoder가 있다.

Residual block은 `Conv1d → GroupNorm → SiLU → Conv1d → GroupNorm` 뒤에 128차원
condition으로 예측한 scale/shift를 적용하고 residual skip을 더한다. 따라서 논문에는
구조를 다음처럼 표기한다.

```text
Conditional dilated 1D U-Net
+ sinusoidal flow-time embedding
+ context-conditioned scale-shift residual blocks
+ forward-Euler ODE solver
```

코드의 modulation은 LayerNorm이 아니라 GroupNorm 위에 적용되므로, 엄밀한 구조명으로는
`AdaLN`보다 **conditional scale-shift GroupNorm**이 정확하다.

Stage 2 training loss는 다음과 같다.

```text
L_stage2 = L_FM + 0.3 × L_maneuver + 0.01 × L_mean
```

Maneuver 보조 head는 context에서 lateral 3-class와 longitudinal 2-class를 예측한다.
Ground-truth maneuver는 vector field의 condition으로 전달되지 않는다.

여기서 표현을 주의해야 한다. **PFL 학습이 maneuver category를 전혀 사용하지 않는 것은
아니다.** Data loader가 제공하는 lateral 3-class와 longitudinal 2-class GT label은
`L_maneuver`의 supervision으로 사용되고, 그 gradient가 shared encoder까지 전달된다.
그러나 `mode_conditioned=false`이므로 category one-hot, argmax 또는 sampled category가
vector field 입력으로 전달되지 않는다. 따라서 category는 encoder representation을
regularise하는 보조 과제이지 trajectory 생성 전에 선택되는 latent mode가 아니다.

`L_mean`은 mean head가 예측한 trajectory와 GT 사이의 masked squared error다. Mean
head와 flow가 같은 encoder를 사용하므로(`separate_encoders=false`) 이 작은 loss의
gradient가 encoder에도 전달된다. Mean prediction 자체는 flow condition으로 들어가지
않으며 추론 결과에도 사용하지 않는다.

---

## 8. Two-stage 학습과 mean head의 역할

Stage2 PFL은 하나의 추론 모델을 두 학습 단계로 최적화한다. 여러 모델의 출력을 합치는
ensemble이 아니다.

### 8.1 Stage 1: encoder regression warm-start

```text
flow_weight = 0
mean_weight = 1
aux_weight = 0.3
epochs = 3

L_stage1 = L_mean + 0.3 × L_maneuver
```

Mean head는 112차원 context를 입력받는 다음 MLP다.

```text
Linear(112→256) → GELU → Linear(256→256) → GELU → Linear(256→50)
                                                → reshape [25,2]
```

출력은 flow와 동일한 standardised CV-residual 공간에 있고, loss 계산 전 residual
statistics와 target CV를 사용해 실제 trajectory로 복원한다. 이 단계에서는 vector
field를 호출하지 않으며 flow decoder는 학습되지 않는다. Encoder, relational encoder,
maneuver head와 mean head가 학습된다.

### 8.2 Stage 2: full-trajectory flow matching

Stage 1의 EMA 최종 checkpoint로 전체 `PFLNet`을 초기화한 뒤 optimizer와 scheduler를
새로 만든다.

```text
flow_weight = 1
mean_weight = 0.01
aux_weight = 0.3
epochs = 15
separate_encoders = false
```

Stage 2에서는 vector field가 주 학습 대상이고, mean loss는 shared encoder가 deterministic
trajectory 정보를 완전히 잃지 않게 하는 작은 auxiliary 역할을 한다. `mean_weight=0.01`
은 mean head 출력을 vector field에 전달한다는 뜻이 아니다.

### 8.3 추론 구조

추론에서는 다음 경로만 사용한다.

```text
history + neighbours + ego plan
        → shared PFLEncoder
        → conditional vector field
        → Euler integration of Gaussian noise
        → full trajectory samples
```

`mean_head`, maneuver GT label, target future GT는 추론에 사용하지 않는다. 공정 evaluator가
legacy Stage2 checkpoint를 읽을 때 mean-head module도 생성하는 이유는 state dictionary를
strict하게 복원하기 위해서이며, prediction은 `PFLNet.sample()`의 flow samples만 사용한다.

추론에서 maneuver head도 실행 결과 선택에 사용하지 않는다. `PFLNet.sample()` 내부의
maneuver probability 계산과 mode sampling branch는 `mode_conditioned=true`일 때만
실행되며, 최종 Stage2는 이 flag가 false다. 따라서 최종 모델은 lateral/longitudinal class를
먼저 argmax하거나 sampling한 뒤 해당 class 전용 trajectory를 생성하지 않는다.

Mean head도 checkpoint에는 존재하지만 정식 evaluator는 `mean_predict()`를 호출하지 않는다.
GT-free sample mean은 deterministic mean-head 출력이 아니라 K개 flow trajectory의 산술
평균이고, sample medoid는 K개 flow trajectory만 사용해 선택한 실제 후보 하나다. Mean
head와 maneuver head가 추론 입력으로 사용되지 않는다는 사실과, 학습 중 shared encoder
weight에 간접적으로 기여했다는 사실은 동시에 성립한다.

### 8.4 단계와 경로별 실제 사용 여부

| 구성요소 | Stage 1 warm-start | Stage 2 학습 | 배포형 GT-free 추론 | Endpoint oracle 평가 |
|---|---|---|---|---|
| history/neighbours/ego plan encoder | 학습 | 학습 | 사용 | 사용 |
| relational encoder | Full에서 학습 | Full에서 학습 | context에 사용 | context에 사용 |
| 25-step vector field | 호출·학습 안 함 | 주 loss로 학습 | Gaussian noise 적분 | 먼저 동일하게 전체 후보 생성 |
| maneuver GT label | auxiliary CE 정답 | auxiliary CE 정답 | 없음 | 없음 |
| maneuver head 출력 | CE 계산 | CE 계산 | 사용 안 함 | 사용 안 함 |
| deterministic mean head | 주 회귀 head | weight 0.01 보조 head | 사용 안 함 | 사용 안 함 |
| endpoint | 사용 안 함 | 사용 안 함 | 사용 안 함 | 생성 완료 후 GT endpoint로 후보 선택 |
| target future GT | mean/maneuver supervision | FM/mean/maneuver supervision | 없음 | 후보 생성 후 selector와 metric에만 사용 |

따라서 "학습과 추론의 모든 순간에 maneuver나 endpoint가 존재하지 않는다"고 쓰면
부정확하다. 학습에는 maneuver auxiliary supervision이 있고 평가에는 endpoint oracle이
있다. 정확한 핵심은 **maneuver 또는 endpoint를 flow 생성의 선행 latent decision으로
사용하지 않는다**는 것이다.

처음부터 flow-only로 학습한 `fair_ngsim_pfl_full_seed*`는 다음 설정의 비교군이다.

```text
flow_weight = 1, mean_weight = 0, init_from = none
```

이 비교군의 열세는 mean head가 추론에 필요하다는 증거가 아니라, 현재 조건에서
regression warm-start와 shared-encoder auxiliary를 사용한 학습 과정이 더 좋은 flow
checkpoint를 만들었다는 예비 근거다.

---

## 9. Optimisation and checkpoint selection

Stage 1과 Stage 2의 공통 최적화 조건은 다음과 같다.

| 항목 | 값 |
|---|---|
| optimizer | AdamW |
| learning rate | `1e-4` |
| weight decay | `0.0` |
| schedule | 500-step linear warmup 후 cosine decay |
| batch size | 64 scenes |
| gradient clipping | global norm 1.0 |
| EMA | 0.999 |
| Stage 1 epochs | 3 |
| Stage 2 epochs | 15 |
| Stage 1 validation metric | deterministic mean-head RMSE |
| Stage 2 validation metric | GT-free K=20 flow sample-mean trajectory RMSE |
| hyperparameter search budget | 0 |
| final comparison seeds | 0, 1, 2 |

기존 `ngsim_pfl_stage2-best.tar`는 Stage 2 중 작은 validation subset을 자주 측정해
선택한 legacy checkpoint다. 현재 evaluator로 공정 재평가했지만, 이 선택 절차까지
논문용 최종 protocol로 간주하지 않는다. 논문용 재현에서는 Stage 1을 정확히 3 epoch
수행한 뒤 최종 EMA로 Stage 2를 초기화하고, Stage 2 checkpoint는 shuffle 없는 고정
validation subset의 epoch-level K=20 sample-mean RMSE로 선택한다. Test set은 선택에
사용하지 않는다.

Validation sampling seed는 모든 epoch에서 동일하게 고정한다. Python, NumPy,
Torch/CUDA RNG와 DataLoader 순서도 seed로 고정한다. Stage 1에서 Stage 2로 넘어갈 때
model weight와 residual-normalisation buffer는 전달하지만 optimizer와 learning-rate
scheduler는 초기화한다.

저장 파일은 다음과 같다.

```text
latest.pt       raw model, EMA, optimizer, scheduler, resume position
{name}-epN.tar  epoch N EMA
{name}-best.tar 사전 지정 validation metric이 가장 낮은 EMA
{name}.tar      학습 종료 시점 EMA
```

정식 재현 평가는 Stage 2의 `{name}-best.tar`만 사용한다. Checkpoint와 같은 디렉터리의
`run_manifest.json`에서 Stage 1 parent checkpoint hash, config hash, dataset, seed와
fair-summary eligibility를 검증해야 한다.

Training log는 500 batch마다 다음을 기록한다.

```text
loss, FM, maneuver, mean, learning rate, ETA
```

Stage 1에서는 FM 값이 실제 loss가 아니라 `0.0000` placeholder다. Stage 2에서는 FM과
mean이 모두 실제 학습 loss다. Epoch 종료 시 평균 train loss, fixed validation RMSE,
spread diagnostics와 best checkpoint 갱신 여부를 기록한다.

### 9.1 채택하지 않은 선택적 구조

코드에는 다음 opt-in 실험 모듈이 남아 있지만 현재 Stage2 PFL 구조에는 포함하지 않는다.

- relation sequence를 U-Net 입력 horizon에 연결하는 TemporalREL
- 각 U-Net residual block의 relation cross-attention adapter
- maneuver mode를 vector field condition으로 넣는 mode conditioning
- endpoint-first generation

채택 여부는 test set을 보지 않고 NGSIM seed 0, 5-epoch fixed-validation pilot으로
판정했다. `Global+Temporal`은 K=20 sample-mean RMSE가 Full의 2.1436 m에서 2.3393 m로
0.1958 m(9.13%) 악화되어 사전 지정한 3% 중단 기준을 넘었다. 이에 따라 TemporalREL은
후속 multi-seed 실험으로 승격하지 않았다.

Relation cross-attention과 mode conditioning은 둘을 함께 켠 조합으로 시험했다. 이 조합은
같은 기준에서 2.2167 m로 Full보다 0.0731 m(3.41%) 나빠 채택하지 않았다. 이 실험만으로
cross-attention과 mode conditioning 중 어느 하나가 성능 저하의 원인이라고 분리해서
말할 수 없으며, 두 기능 각각이 독립적으로 효과가 없다고 결론 내릴 수도 없다. 또한 두
pilot은 mean-head warm-start가 없는 flow-only 5-epoch 조건이므로, 현재 two-stage Stage2
조건에서도 반드시 실패한다는 검증은 아니다.

이 결과는 최종 모델을 REL=0으로 바꾼다는 의미가 아니다. 최종 모델은 계속 Stage2
Full이며, 위 세 기능은 Full 이후에 시험한 확장 후보에 해당한다. 제한된 pilot에서 개선
근거를 얻지 못했기 때문에 확장 모듈을 Full에 추가하지 않고, 이미 3-seed로 재현한
Stage2 Full 구조를 논문 모델로 유지한다.
코드상 relation cross-attention은 horizon별 relation token을 요구하므로 TemporalREL을
끄면 함께 비활성화된다. Mode conditioning은 독립적으로 켤 수 있지만, 단독 채택 실험은
아직 수행하지 않았다. 따라서 논문 모델은 **GRU로 요약한 relational encoding과 25시점
전체 trajectory flow matching**을 유지한다. 이 모듈들은 명시적 flag를 켜고 처음부터
다시 학습하지 않는 한 사용되지 않는다.

---

## 10. Sampling

PFL은 forward Euler로 rectified-flow ODE를 적분한다.

```python
dt = 1 / NFE
x = randn(B*K, 25, 2)
for i in range(NFE):
    x = x + dt * vector_field(x, i/NFE, repeated_context)
```

Fair evaluation은 `NFE=10`, `K∈{6,20}`을 사용한다. K=20을 한 번 생성하고 앞 여섯
sample을 K=6 결과로 사용하므로 후보 집합이 nested된다. GPU sampling chunk 기본값은
20이다. 이는 후보 정의를 바꾸지 않고 K draw를 widened batch로 계산한다.

출력은 `[K,B,25,2]`이며 de-normalization과 target CV를 더한 feet 좌표다.

각 Euler step에서 vector field는 현재의 `[B×K,25,2]` tensor 전체에 대한 velocity를
출력한다. Flow time `u` 하나는 한 target trajectory의 25개 horizon position에 공통으로
적용되지만, U-Net 출력에는 각 horizon의 2차원 velocity가 따로 존재한다. 최종 Stage2의
`NFE=10`은 실제 미래 시점이 10개라는 뜻이 아니라 flow ODE를 10번 갱신한다는 뜻이다.

각 target의 noise와 flow trajectory는 독립적으로 생성된다. Agent interaction은 encoder
context에만 들어가며 여러 agent의 sample index `k`가 하나의 coherent joint scene을
뜻하지는 않는다.

---

## 11. GT-free와 oracle 평가

평가는 `experiments/scripts/evaluate_model.py`에서 CV, EPN, PFL에 공통으로 수행한다.

### 11.1 GT-free sample mean

```text
Y_mean = (1/K) Σ_k Y_k
```

선택 함수는 sample tensor만 받고 GT 인자를 받지 않는다. Conditional mean의 Monte
Carlo estimate이며 실제 생성 후보 trajectory는 아니다. 이 사실을 metadata에 남긴다.

### 11.2 GT-free sample medoid

```text
k* = argmin_k Σ_j distance(Y_k, Y_j)
Y_medoid = Y_k*
```

GT를 사용하지 않고 실제 생성 sample 중 하나를 반환한다. Distance는 전체 trajectory를
flatten한 뒤 candidate 사이 L2 distance를 합산한다.

### 11.3 Endpoint oracle

```text
k* = argmin_k ||Y_k(T) - Y_GT(T)||²
```

GT endpoint에 가장 가까운 후보 하나를 선택한 뒤 그 후보의 전체 trajectory RMSE,
ADE, FDE를 계산한다. Direct minADE와 direct minFDE는 각각 별도의 후보를 선택할 수
있으므로 `oracle_coverage/direct_minADE`, `direct_minFDE`라는 별도 metric으로 저장한다.
Endpoint-selected trajectory RMSE와 direct minima를 혼용하지 않는다.

이 selector는 **K개의 25-step trajectory가 모두 생성된 뒤** 적용된다. GT endpoint가
vector field의 condition으로 들어가거나, 선택된 endpoint로 trajectory를 다시 decode하지
않는다. 따라서 endpoint oracle은 후보 coverage를 측정하는 평가 규칙이며 PFL 생성 구조나
배포 추론 절차의 일부가 아니다.

세 metric은 실제 5초 endpoint까지 mask가 유효한 target에만 계산한다. 짧게 잘린
future의 padding 좌표를 GT endpoint로 취급하지 않는다. GT-free RMSE/FDE도 각
horizon의 정답이 존재하는 차량만 그 horizon 집계에 포함한다. 결과 metadata의
`metric_revision=full_horizon_mask_v2`와 `runtime.full_horizon_targets`가 이 조건과
표본 수를 기록한다. 이 수정 전 산출물은
`experiments/archive/pre_mask_fix_20260913`에 격리되어 있으며 논문 수치로 사용하지 않는다.

### 11.4 EPN GT-free 제한

현재 EPN은 sampled candidate별 probability 또는 confidence를 출력하지 않는다.
따라서 EPN top-confidence metric을 임의로 만들지 않고 metadata에
`unavailable_no_candidate_specific_confidence`로 기록한다. 공통 GT-free 비교에는
명시적으로 sample-medoid fallback을 사용하며 최종 summary에도 이 한계를 표시한다.

---

## 12. 저장 metric

각 dataset/model/seed에 대해 다음을 metre 단위로 저장한다.

- RMSE at 1, 2, 3, 4, 5 seconds
- ADE와 FDE
- GT-free sample mean과 sample medoid, K=6/20
- endpoint-selected oracle trajectory RMSE/ADE/FDE, K=6/20
- direct minADE와 minFDE, K=6/20
- PFL lateral/longitudinal sample spread
- average pairwise trajectory distance
- endpoint spread와 endpoint coverage at 0.5/1/2/3/5 m
- spread-error ratio
- parameter count
- encoding/sampling/전체 latency per scene과 per target
- K=6/K=20 생성 시간, NFE, GPU/CPU 정보

결과 CSV의 기본 열은 다음과 같다.

```text
dataset, model, variant, seed, protocol, selection_rule, K, NFE,
horizon, RMSE, ADE, FDE, minADE, minFDE, num_windows, num_vehicles,
checkpoint, recording_id, dataset_segment, vehicle_id, group
```

축약 평가나 임의 checkpoint smoke 결과는 `experiments/results/smoke`로 분리하며
`complete_test=false`로 저장한다. Summary는 `complete_test=true`인 정식 test 결과만
읽는다.

---

## 12.1 정성적 최종 경로 비교

정량 지표를 보완하기 위해 동일 test 장면에서 PFL과 EPN의 K=20 후보와 GT-free
sample-medoid 최종 경로를 비교한다. 그림에는 target history, recorded ego future로 만든
ego plan proxy, target ground truth, 두 모델의 20개 후보와 medoid를 함께 표시한다. 좌표는
target 현재 위치 기준이며 metre로 변환한다.

평면 경로에서 고속도로의 횡 이동 수 m와 종 이동 수십~수백 m를 같은 크기의 축에 강제로
늘리면 작은 횡 흔들림이 큰 곡선처럼 보인다. 최종 그림은 이 왜곡을 사용하지 않는다. 첫
열은 원본 track의 차량 중심, length/width와 진행 방향으로 그린 실제 축척의 현재 장면이고,
둘째·셋째 열은 5초 전체 경로의 lateral/longitudinal position을 시간축으로 분리해 표시한다.
선정된 모든 사례에서 cache의 GT를 원본 `Tracks`의 동일 vehicle/frame future와 직접 대조한
최대 절대차는 highD와 NGSIM 모두 0.0이었다.

사례는 예측을 본 뒤 test set 전체에서 임의로 고르지 않는다. 정렬된 test set 전체 범위에
`linspace`로 고정한 512개 scene을 먼저 선택하고 두 모델에 동일하게 사용한다. 이 표본에서
미리 정한 네 범주인 lane change, braking, direct follower, adjacent-near 각각에 대해 PFL이
이긴 사례들의 ADE 개선량 중앙값에 해당하는 장면을 대표 성공 사례로 선택한다. 또한 PFL이
진 사례들의 손실 중앙값 장면을 별도 failure figure로 반드시 함께 보존한다.
전처리의 lane-change label은 lane transition 전후 4초를 모두 포함하므로 이미 변경을 끝낸
장면이 선택되지 않도록, lane-change 그림에는 미래 lateral displacement가 1.5 m 이상인
조건을 추가한다.

- 비교 metric: K=20 GT-free sample-medoid ADE
- 표시 후보: PFL 20개 full-trajectory flow samples, EPN 20개 candidates
- 시각화 seed: 0
- highD screened targets: 1,088, PFL win rate 52.57%, mean `EPN-PFL` ADE +0.020 m
- NGSIM screened targets with future: 3,173, PFL win rate 49.67%, mean `EPN-PFL` ADE -0.010 m
- 생성 코드: `experiments/scripts/collect_qualitative.py`,
  `experiments/scripts/plot_qualitative.py`
- 결과: `experiments/figures/qualitative/`

고정 표본의 범주별 결과는 다음과 같다. 양수 `EPN-PFL`은 PFL의 낮은 ADE를 뜻한다.

| Dataset | Category | Targets | PFL win rate | Mean `EPN-PFL` ADE (m) | Median (m) |
|---|---:|---:|---:|---:|---:|
| highD | lane change | 32 | 53.13% | +0.029 | +0.003 |
| highD | braking | 66 | 60.61% | +0.067 | +0.058 |
| highD | direct follower | 118 | 61.02% | +0.085 | +0.046 |
| highD | adjacent near | 837 | 52.33% | +0.015 | +0.009 |
| NGSIM | lane change | 47 | 42.55% | +0.031 | -0.085 |
| NGSIM | braking | 528 | 60.42% | +0.180 | +0.108 |
| NGSIM | direct follower | 451 | 54.32% | +0.061 | +0.038 |
| NGSIM | adjacent near | 2,252 | 47.16% | -0.052 | -0.033 |

이 고정 표본에서는 braking과 direct-follower가 두 데이터셋에서 일관된 PFL 개선을 보인다.
lane-change는 highD에서만 약한 양의 경향이며, NGSIM lane-change와 adjacent-near 개선은
주장하지 않는다. 이 범주 수치는 seed 0의 512-scene 진단 표본이므로 본문에서는 정성적
관찰로 쓰고, 별도 3-seed 전체-test mode 분석 전에는 통계적 주 결과로 승격하지 않는다.

권장 figure caption:

> Qualitative comparison of GT-free K=20 sample-medoid trajectories. Thin lines
> show generated candidates, thick blue/red lines denote the PFL/EPN medoids,
> black denotes the target ground truth, and dashed orange denotes the ego plan.
> Examples are selected by a fixed pre-declared median-gain rule from uniformly
> sampled test scenes. PFL more closely follows the observed trajectory in the
> illustrated braking and direct-following interactions; a representative failure
> case is reported separately in the appendix.

이 정성 그림은 해당 상황에서 궤적 형상과 후보 분포가 어떻게 달라지는지 설명하는 자료다.
범주 전체의 우위를 증명하는 통계로 사용하지 않으며, 특히 NGSIM 고정 표본 평균은 PFL
우위를 보이지 않는다. 전체 성능 주장은 3-seed 정식 test 평가와 bootstrap 결과를 따른다.

---

## 13. Interaction group

그룹은 예측 시작 시점의 현재 frame 정보만 사용하며 threshold는 config에 사전 지정된
30 m다. 내부 feet 좌표의 threshold는 metre에서 변환한다.

- G1: ego와 같은 lane이고 target longitudinal position이 ego 뒤쪽
- G2: ego 인접 lane이고 절대 longitudinal distance가 30 m 이하
- G3: 나머지 target

전체 평균과 G1/G2/G3 각각에서 PLAN=0–REL=0, REL=0–Full, Full–EPN paired difference를
저장한다.

---

## 14. 통계와 주장 판정

Raw sliding window를 독립 표본으로 사용하지 않는다. 먼저 겹치는 window를
dataset-scoped target vehicle 안에서 집계하고, 같은 vehicle의 seed 반복도 vehicle
안에서 평균한다. 이후 다음을 계산한다.

- seed별 결과와 mean ± sample standard deviation
- paired vehicle bootstrap 95% confidence interval, 2,000 repetitions
- recording 단위 bootstrap 95% confidence interval
- planning gain과 relational gain
- Full–EPN GT-free 및 oracle 차이

Oracle equivalence의 사전 지정 상대 margin 3%는 보조 분석으로 유지한다. Bootstrap CI
전체가 `[-0.03,0.03]` 안에 있을 때만 실질적 동등성으로 판단하며, 유의하지 않다는
이유만으로 동등하다고 결론 내리지 않는다. 다만 최종 논문 주장의 성립 조건을 oracle
동등성으로 제한하지 않는다. Oracle은 후보 집합의 coverage, GT-free는 실제 선택 가능한
prediction 품질을 측정하기 때문이다.

최종 PFL–EPN 비교는 Stage2 PFL과 EPN을 같은 target 집합, horizon mask, K와 seed로
평가한다. 개발용 variant와 pilot은 seed 0으로 선별할 수 있지만, EPN 대비 우위처럼
논문의 주 결과는 seed 0, 1, 2의 평균과 분산을 보고한다. 결과는 다음처럼 해석한다.

- relation 개선 + PFL GT-free 우수: 핵심 주장과 구조 ablation을 모두 지지
- PFL GT-free 우수 + EPN oracle 우수: 배포 선택과 oracle coverage의 trade-off를 보고
- PFL GT-free 우수 + relation 독립 효과 불명확: 성능 주장은 가능하지만 relation 기여는 제한
- PFL GT-free 우위가 seed 간 유지되지 않음: 현재 EPN 대비 우위 주장은 보류

### 14.1 최종 Stage2 결과

Stage2 Full, Stage2 REL=0과 EPN의 seed 0, 1, 2 전체 test 평가가 모두 완료됐다. 아래
gain은 차량별 RMSE를 먼저 집계한 뒤 paired vehicle bootstrap으로 계산했으며, 양수이면
candidate가 더 좋다. GT-free 주 비교는 K=6 sample medoid다.

| Dataset | 비교 | Mean gain (m) | Vehicle-bootstrap 95% CI (m) | 판정 |
|---|---|---:|---:|---|
| NGSIM | Stage2 REL=0 → Stage2 Full | +0.1278 | [+0.1225, +0.1333] | Full 유의 우수 |
| highD | Stage2 REL=0 → Stage2 Full | -0.0011 | [-0.0018, -0.0004] | 전체 차량 평균은 REL=0 미세 우수 |
| NGSIM | EPN → Stage2 Full, GT-free | +0.2326 | [+0.2198, +0.2458] | Full 유의 우수 |
| NGSIM | EPN → Stage2 Full, endpoint oracle | -0.1063 | [-0.1140, -0.0988] | EPN 유의 우수 |
| highD | EPN → Stage2 Full, GT-free | +0.0767 | [+0.0754, +0.0780] | Full 유의 우수 |
| highD | EPN → Stage2 Full, endpoint oracle | +0.0401 | [+0.0390, +0.0413] | Full 유의 우수 |

highD relation 결과는 상황별로 다르다. Direct-follower G1에서는 Full이 +0.0262 m
개선됐지만 adjacent-near G2에서는 REL=0이 0.0060 m 좋았다. 전체 paired 평균의 차이는
1.1 mm로 작으므로 highD 전체에서 relational encoder가 일반적으로 개선한다고 강하게
주장하지 않는다. NGSIM에서는 ALL과 G1/G2/G3 모두 Full의 CI가 0보다 커 relation 기여를
일관되게 지지한다.

`summary.csv`의 전체-window aggregate RMSE와 위 paired-vehicle mean gain은 weighting이
다르다. 전자는 모든 window를 합쳐 RMSE를 계산하므로 window가 많은 차량의 비중이 크고,
후자는 각 차량을 통계 단위로 동일하게 취급한다. 예를 들어 highD K=6 sample-medoid의
aggregate RMSE는 Full 1.3069 m, REL=0 1.3298 m지만, 차량별 paired 평균은 REL=0이
0.0011 m 좋다. 논문 유의성 판정에는 사전 지정한 paired-vehicle 결과를 사용하고,
aggregate RMSE는 일반 성능 표에 함께 보고한다.

---

## 15. Split leakage audit

현재 제공된 split은 ego vehicle ID를 분리하지만 recording-disjoint가 아니다.

| Dataset | Split | Windows | Ego vehicles | Target vehicles | Recordings |
|---|---|---:|---:|---:|---:|
| NGSIM | train | 5,777,839 | 8,844 | 9,165 | 6 |
| NGSIM | val | 761,734 | 1,010 | 1,457 | 6 |
| NGSIM | test | 1,320,103 | 1,925 | 2,934 | 6 |
| highD | train | 3,796,329 | 67,583 | 67,565 | 60 |
| highD | val | 498,337 | 9,648 | 9,842 | 60 |
| highD | test | 1,010,583 | 18,756 | 18,827 | 60 |

NGSIM target overlap은 train/val 487, train/test 826, val/test 557 vehicles다. highD는
train/val 209, train/test 0, val/test 254다. 모든 recording이 모든 split에 나타난다.
Benchmark를 임의로 바꾸지 않고 기존 split을 유지했으며, 이 결과는 generalisation
주장의 제한으로 보고해야 한다.

---

## 16. 현재 실험 산출물

```text
experiments/run_status.json                 현재 stage와 완료 run
experiments/results/progress_report.json    학습/평가 중간 요약
experiments/results/{dataset}/per_seed/     seed별 정식 평가
experiments/results/{dataset}/per_vehicle/  차량 단위 평가
experiments/results/{dataset}/summary.csv   seed mean±std
experiments/results/{dataset}/summary.json  paired CI, equivalence, A–D 판정
experiments/figures/                         자동 생성 그림
experiments/stage2_rel0_status.json          Stage2_REL0 현재 단계·dataset·seed
experiments/configs/pfl_stage2_rel0*.yaml    Stage2_REL0 warm-start/flow 설정
experiments/scripts/run_stage2_rel0.sh       Stage2REL0 3-seed 실행·재개 스크립트
```

기존 공정 비교의 완료 상태는 `run_status.json`, relation ablation 완료 상태는
`stage2_rel0_status.json`에서 확인한다. 각 model은 validation-best checkpoint의 전체 test
평가가 완료되고 `complete_test=true` metadata가 생성된 결과만 정식 결과로 취급한다.

---

## 17. 논문 작성 시 지켜야 할 표현

현재 코드와 데이터가 뒷받침하는 범위에서 다음을 구분해야 한다.

1. `sample mean`은 GT-free point estimator지만 실제 sample trajectory가 아니다.
2. `sample medoid`는 GT-free이며 실제 후보 하나다.
3. `endpoint oracle`은 GT future endpoint를 선택에 사용하므로 배포 지표가 아니다.
4. EPN top-confidence는 현재 구현에서 제공되지 않는다.
5. Ego-plan은 실제 planner output이 아니라 recorded-future proxy다.
6. Stage2 PFL은 학습 중 mean head와 regression warm-start를 사용하지만, 추론 출력은
   flow samples만 사용한다.
7. 기존 split은 recording-disjoint generalisation을 검증하지 않는다.
8. EPN 대비 성능과 relation의 독립 효과는 각각 완료된 Stage2 및 `Stage2_REL0` 3-seed
   전체 test 결과와 paired bootstrap으로 판단한다.

### 17.1 전체 trajectory 생성에 대한 권장 문장

다음 문장은 학습, 추론, 평가 코드를 모두 반영한다.

> PFL은 별도의 maneuver mode 또는 endpoint를 생성·선택하여 trajectory decoder에
> 전달하지 않는다. 대신 target의 constant-velocity reference에 대한 정규화된 25-step
> trajectory residual을 conditional rectified flow로 공동 생성한다. Maneuver
> classification과 deterministic mean regression은 shared encoder 학습을 위한 보조
> 목적함수로 사용되지만, 해당 head의 출력은 최종 flow sampler에 입력되지 않는다.
> 추론에서는 Gaussian trajectory noise를 Euler integration한 뒤 CV reference를 더하며,
> GT endpoint는 생성 이후 oracle 평가에서만 후보 선택에 사용된다.

피해야 할 표현은 다음과 같다.

- `PFL은 학습에서도 maneuver label을 전혀 사용하지 않는다.`
  - 실제로는 auxiliary cross-entropy supervision에 사용한다.
- `PFL은 raw future trajectory를 flow로 생성한다.`
  - 실제 생성 공간은 horizon/axis별로 standardise한 CV residual이다.
- `PFL은 endpoint를 사용하지 않는다.`
  - 생성에는 사용하지 않지만 endpoint-oracle 평가에는 사용한다.
- `Mean head는 PFL 추론 결과다.`
  - 현재 정식 추론 결과는 flow sample mean/medoid이며 mean head는 training auxiliary다.
- `NFE=10은 미래 10개 시점을 순차 예측한다.`
  - 미래는 25시점이고 NFE는 flow-time Euler update 횟수다.

이 제약을 유지하면 논문의 method, experiment protocol, limitation과 실제 실행 코드가
서로 모순되지 않는다.
