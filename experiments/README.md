# PFL–EPN 공정 비교 실험

이 디렉터리는 NGSIM과 highD에서 CV, EPN, PFL `PLAN=0`, `REL=0`, `Full`을
같은 split과 사전 지정된 선택 규칙으로 비교한다. 테스트 결과는 설정이나
threshold를 바꾸는 데 사용하지 않는다. 전체 저장소 점검과 데이터 누수 결과는
[`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md)에 기록되어 있다.

## 사전 지정된 조건

- PFL mean head는 생성하지 않으며 flow matching으로만 학습한다.
- 모든 PFL variant는 112차원 context와 같은 decoder, optimizer, schedule,
  batch size, 검증 지표, checkpoint 규칙을 사용한다.
- 검증 지표는 고정된 validation 순서와 seed로 생성한 `K=20` sample mean의
  trajectory RMSE이다. 최대 15 epoch, 최소 5 epoch, patience 3이다.
- 평가 후보 수는 `K=6,20`이며 K=6은 같은 K=20 draw의 앞 여섯 개이다.
- PFL은 Euler NFE=10이다. seed는 0, 1, 2이고 equivalence margin은 상대 3%다.
- 단위는 loader 내부에서 feet이고 모든 결과 CSV는 metre다.
- RNG와 DataLoader 순서는 고정한다. 강제 deterministic CUDA kernel은 지원하지만
  기본값은 끈다. RTX 4070 SUPER의 Conv1d에서 수백 배 느려지는 현상을 임의 가중치
  한 배치로 확인했으며, test 성능을 보기 전에 이 실행 조건을 고정했다. 필요하면
  평가 명령에 `--deterministic-algorithms`를 추가할 수 있다.

PFL 파라미터 수는 PLAN=0 3,138,711, REL=0 3,178,727, Full 3,210,439이다.
relation branch 차이는 Full의 0.99%이다. 따라서 주 실험은 해석이 명확한 REL=0을
사용한다. `--relation_control target_cv`를 쓰는 선택적 대조군은 Full과 정확히
같은 3,210,439개 파라미터를 갖도록 분리되어 있다.

## 실행

단일 학습은 다음과 같이 실행한다.

```bash
experiments/scripts/run_train.sh \
  --config experiments/configs/pfl_full.yaml --dataset ngsim --seed 0

experiments/scripts/run_train.sh \
  --config experiments/configs/epn.yaml --dataset highd --seed 0
```

단일 평가는 best-validation checkpoint를 명시한다.

```bash
experiments/scripts/run_evaluation.sh \
  --model pfl --config experiments/configs/pfl_full.yaml \
  --dataset ngsim --seed 0 \
  --checkpoint PFL/trained_models/fair_ngsim_pfl_full_seed0/fair_ngsim_pfl_full_seed0-best.tar
```

전체 24개 학습(2 datasets × 3 seeds × 4 learned models), CV 평가, 통계 요약,
그림 생성은 다음 명령으로 순차 실행한다. 같은 run 이름의 `latest.pt`가 있으면
동일 provenance일 때만 이어서 학습한다.

```bash
experiments/scripts/run_all_experiments.sh
```

실행 순서는 split audit → dataset별 CV → seed별 PFL 세 variant → EPN → summary
→ figures이다. 각 dataset/seed의 네 학습 모델 평가가 끝날 때마다 summary, figure,
`results/progress_report.json`을 갱신한다. 전체 3-seed 행렬이 완성되기 전에는 A–D
판정을 `pending`으로 유지한다. `report_parameters.py`는 데이터 없이 모델 크기를
다시 기록한다.

## 평가 정의

- `gt_free/sample_mean`: K개 trajectory의 시점별 산술 평균. GT를 받지 않으며
  실제 생성 sample이 아닌 conditional mean 추정값이다.
- `gt_free/sample_medoid`: K개 중 다른 모든 trajectory까지의 거리 합이 가장 작은
  실제 sample. 선택 함수에는 GT 인자가 없다.
- `oracle/gt_endpoint_nearest_trajectory`: GT 5초 endpoint에 가장 가까운 endpoint를
  가진 후보 하나를 선택한 뒤 그 후보의 전체 RMSE/ADE/FDE를 계산한다. 실제 5초
  endpoint가 존재하는 target만 포함하며 padding 값은 선택이나 metric에 사용하지 않는다.
- `oracle_coverage/direct_minADE`, `direct_minFDE`: 각 지표를 직접 최소화한 값이며
  endpoint-selected oracle과 별도 열로 저장한다. 이 두 값도 완전한 5초 future가
  존재하는 동일 target 집합에서 계산한다.
- PFL은 sample spread, 평균 pairwise 거리, endpoint spread, endpoint coverage,
  spread-error ratio도 저장한다.

각 horizon의 RMSE/FDE에는 그 horizon 정답이 실제로 존재하는 차량만 포함한다.
`per_seed/*.json`의 `metric_revision=full_horizon_mask_v2`와
`runtime.full_horizon_targets`로 이 조건과 표본 수를 확인할 수 있다. 수정 전 산출물은
`experiments/archive/pre_mask_fix_20260913`에 보관하며 논문 결과로 사용하지 않는다.

EPN 구현에는 후보별 endpoint/maneuver confidence가 없다. 따라서 top-confidence
수치는 만들지 않고 metadata에 unavailable로 남긴다. 공통 GT-free 비교는 명시적으로
`sample_medoid` fallback을 사용하며 summary 경고에도 이 한계를 기록한다.

## 결과와 통계

평가 결과는 `experiments/results/{ngsim,highd}/per_seed`와 `per_vehicle`에 쌓인다.
`summarize_results.py`는 완료된 전체 test 평가만 읽어 seed mean±std, 차량 단위 paired
bootstrap 95% CI, recording 단위 bootstrap, planning/relation gain, Full–EPN 차이와
oracle equivalence를 계산한다. 차량이 seed마다 반복되므로 먼저 seed를 차량 안에서
평균한 뒤 차량을 bootstrap한다. 유의하지 않다는 사실을 동등성으로 해석하지 않는다.

## 해석 제한

현재 `.mat`에는 실제 온라인 planner 출력이 없다. ego-plan 입력은 기록된 ego의 미래
trajectory를 quintic fitting한 proxy다. 후보 선택은 GT-free여도 입력 자체는 배포 가능한
planner 신호가 아니므로, 이 실험만으로 “실차 배포 성능”을 주장할 수 없다. 내부 planner
출력이 준비되면 같은 tensor contract로 교체해 다시 평가해야 한다.

공급된 split은 recording-disjoint가 아니다. NGSIM과 highD 모두 모든 recording이 각
split에 나타나며 일부 target vehicle과 인접 시점도 겹친다. 기존 benchmark를 임의로
바꾸지 않고 audit 결과만 고정했다. PiP는 소스는 있으나 검증 가능한 compatible
checkpoint가 없어 최소 비교 행렬에서 제외했다.
