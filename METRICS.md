# Multi-Turn Training Metrics

`compute_data_metrics()`에서 리포트되는 multi-turn 전용 메트릭들의 수식과 설명.

> 표기: $N$ = 고유 trajectory 수 (traj_uid 기준 중복 제거), $T$ = max turns, $r_i^{(t)}$ = trajectory $i$의 turn $t$ reward, $G$ = GRPO group 수, $n_g$ = group $g$ 내 trajectory 수

---

## 1. Per-Turn Reward (`turn_reward/turn_{t}/`)

각 턴별로 reward 분포를 추적. 학습이 진행되면 후반 턴의 reward가 상승해야 함.

| Metric | Formula |
|--------|---------|
| `mean` | $\frac{1}{N} \sum_{i=1}^{N} r_i^{(t)}$ |
| `std`  | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i^{(t)} - \bar{r}^{(t)})^2}$ |
| `max`  | $\max_i \; r_i^{(t)}$ |
| `min`  | $\min_i \; r_i^{(t)}$ |

**해석**: turn 1 mean이 낮고 turn 5 mean이 높아지면 → 모델이 multi-turn 대화를 통해 점진적으로 공격 성공률을 높이고 있음.

---

## 2. Episode Success Rate (`episode/success_rate`)

에피소드 내 최소 하나의 턴에서 reward가 threshold 이상인 비율.

$$\text{success\_rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\max_t \; r_i^{(t)} \geq \tau\right]$$

- $\tau = 0.9$ (현재 하드코딩)
- **해석**: 이 값이 꾸준히 올라가면 학습이 정상적으로 진행되고 있음. 정체되면 reward signal이 부족하거나 policy가 collapse된 것.

---

## 3. GRPO Group Reward Std (`grpo/group_reward_std/`)

같은 prompt에 대한 GRPO group 내 trajectory들의 episode reward 분산. 이 값이 학습 신호의 품질을 결정.

$$R_i = \sum_{t=1}^{T} r_i^{(t)}$$

$$\sigma_g = \sqrt{\frac{1}{n_g} \sum_{i \in g} (R_i - \bar{R}_g)^2}$$

| Metric | Formula |
|--------|---------|
| `mean` | $\frac{1}{G} \sum_{g=1}^{G} \sigma_g$ |
| `min`  | $\min_g \; \sigma_g$ |
| `max`  | $\max_g \; \sigma_g$ |

**해석**:
- `mean ≈ 0` → group 내 모든 trajectory가 비슷한 reward를 받음 → advantage ≈ 0 → **학습 신호 없음**
- `mean`이 적당히 큰 값 유지 → 다양한 결과가 나오고 있어 gradient에 방향성이 있음
- 학습 후반에 자연스럽게 감소할 수 있음 (policy가 수렴하므로)

---

## 4. Per-Turn Response Length (`turn_response_length/turn_{t}/`)

각 턴의 응답 토큰 수. `turn_token_mask`에서 각 턴에 속하는 토큰 수를 카운트.

$$L_i^{(t)} = \left|\{k : \text{turn\_token\_mask}_i[k] = t\}\right|$$

| Metric | Formula |
|--------|---------|
| `mean` | $\frac{1}{N} \sum_{i=1}^{N} L_i^{(t)}$ |
| `std`  | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (L_i^{(t)} - \bar{L}^{(t)})^2}$ |

**해석**:
- 갑자기 극단적으로 짧아짐 → reward hacking (짧은 응답으로 높은 reward를 받는 shortcut)
- 갑자기 극단적으로 길어짐 → response length 제한에 걸리는 collapse
- 안정적으로 유지되면서 reward가 올라가는 것이 이상적

---

## 5. Reward Delta (`turn_reward/delta_last_first/`)

에피소드의 마지막 턴 reward - 첫 턴 reward. 모델이 대화를 진행하면서 개선하는지 측정.

$$\Delta_i = r_i^{(T)} - r_i^{(1)}$$

| Metric | Formula |
|--------|---------|
| `mean` | $\frac{1}{N} \sum_{i=1}^{N} \Delta_i$ |
| `std`  | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} (\Delta_i - \bar{\Delta})^2}$ |

**해석**:
- `mean > 0` → 뒤로 갈수록 reward가 올라감 → 모델이 multi-turn 전략을 학습하고 있음
- `mean ≈ 0` → 턴이 진행되어도 개선 없음 → 각 턴이 독립적으로 행동
- `mean < 0` → 뒤로 갈수록 오히려 악화 → 대화가 길어질수록 성능 저하

---

## 6. Best-of-N Episode Reward (`episode/reward/best_of_n`)

각 GRPO group(같은 prompt)에서 가장 좋은 trajectory의 episode reward. Policy가 도달 가능한 성능 상한.

$$\text{best\_of\_n} = \frac{1}{G} \sum_{g=1}^{G} \max_{i \in g} R_i$$

**해석**:
- `best_of_n`은 높은데 `episode/reward/mean`이 낮음 → policy가 잠재력은 있지만 일관되지 않음 (exploration 필요)
- 둘 다 높음 → policy가 안정적으로 좋은 성능
- `best_of_n`도 낮음 → 근본적으로 task이 어렵거나 reward signal 문제

---

## 학습 진행 판단 체크리스트

| 신호 | 정상 | 비정상 |
|------|------|--------|
| `episode/success_rate` | 점진적 상승 | 정체 or 0 유지 |
| `turn_reward/turn_{T}/mean` | 상승 | 변동 없음 |
| `grpo/group_reward_std/mean` | 적당한 값 유지 | ≈ 0 (학습 신호 없음) |
| `turn_response_length/turn_*/mean` | 안정적 | 급격한 변화 |
| `turn_reward/delta_last_first/mean` | > 0 (상승 추세) | < 0 (악화) |
| `episode/reward/best_of_n` | mean보다 높되 gap 줄어듦 | mean과 gap이 안 줄어듦 |
