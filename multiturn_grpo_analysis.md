# Multi-Turn GRPO 수식 vs 코드 검증 분석

## 대상 파일
- `agent_system/multi_turn_rollout/rollout_loop.py` — trajectory 수집 및 turn 데이터 축적
- `verl/trainer/ppo/core_algos.py` — `compute_multiturn_grpo_advantage()` 구현
- `verl/trainer/ppo/ray_trainer.py` — advantage 계산 호출 및 actor update

---

## Step 1: Diversity-Augmented Reward

### 수식

$$\hat{r}_{i,t} = r_{i,t} + \alpha \cdot d_{i,t}$$

여기서 $d_{i,t}$는 DPP 기반 leave-one-out diversity:

$$d_{i,t} = \log \frac{\det(\mathbf{L}_t + \gamma \mathbf{I}_G)}{\det(\mathbf{L}_t^{(-i)} + \gamma \mathbf{I}_{G-1})}$$

### 코드 (`core_algos.py:207-218`)

```python
def diversity(i, t):
    if lambda_div == 0:
        return 0.0
    return 1.0  # pass for now. not yet implemented.

r_hat_i = [turn_rewards[i][t] + lambda_div * diversity(i, t)
           for t in range(len(turn_rewards[i]))]
```

### 판정: DPP Diversity 미구현

- `diversity(i, t)`가 항상 `1.0`을 반환하여 실제 계산은 $\hat{r}_{i,t} = r_{i,t} + \alpha$ (상수 더하기)
- 슬라이드의 DPP 기반 leave-one-out diversity가 전혀 구현되어 있지 않음
- 상수를 더하면 group normalization의 mean에 흡수되어 **사실상 아무 효과 없음**

---

## Step 2: Group Normalization

### 수식

$$\mathcal{R} = \{\hat{r}_{i,t} : i = 1, \ldots, G,\ t = 1, \ldots, T\}$$

$$A_{i,t} = \frac{\hat{r}_{i,t} - \text{mean}(\mathcal{R})}{\text{std}(\mathcal{R})}$$

### 코드 (`core_algos.py:220-259`)

```python
id2rhat[idx].extend(r_hat[i])  # flat list에 turn rewards 추가
...
flat_tensor = torch.tensor(flat_rewards, dtype=torch.float32)
id2mean[idx] = flat_tensor.mean().item()
id2std[idx] = flat_tensor.std().item()
...
A_i = [(r - mean_R) / std_R for r in r_hat[i]]
```

### 판정: 구조적으로 올바르나, 중복 카운팅 버그 존재

**구조**: 같은 group (`uid`)에 속하는 모든 trajectory의 turn-level rewards를 flat list로 모아 mean/std 계산 → 수식과 일치

**버그**: 같은 trajectory가 active step 수만큼 batch에 반복 등장 (step 0, step 1, step 2 각각 별도 row). `compute_mean_std_cross_steps=True` (기본값)일 때 `seen_pairs`가 채워지지 않아 같은 trajectory의 rewards가 **중복 extend** 됨.

예시:
- Trajectory A의 rewards `[0.1, 0.2, 0.3]`이 3개 step으로 등장
- `id2rhat = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3]` (3배 중복)
- mean은 동일하나, `torch.std` (Bessel's correction, N-1)에서 N 값이 달라져 std가 미세하게 변함

---

## Step 3: Temporal Credit Assignment

### 수식

$$\tilde{A}_{i,t} = \sum_{k=t}^{T} \gamma^{k-t} A_{i,k}$$

재귀적으로:
$$\tilde{A}_{i,T} = A_{i,T}, \quad \tilde{A}_{i,t} = A_{i,t} + \gamma \tilde{A}_{i,t+1}$$

### 코드 (`core_algos.py:262-272`)

```python
A_tilde_i[-1] = A[i][-1]
for t in reversed(range(T - 1)):
    A_tilde_i[t] = A[i][t] + gamma * A_tilde_i[t + 1]
```

### 판정: 수식과 정확히 일치 ✓

---

## Step 4: Turn → Token 매핑 — 심각한 버그

### 수식의 의도

Turn-level advantage $\tilde{A}_{i,t}$를 해당 turn의 모든 토큰에 동일하게 할당:
- Step $k$의 response 토큰 → $\tilde{A}_{i,k}$

### `turn_token_mask` 축적 방식 (`rollout_loop.py:510-520`)

```python
# 각 step마다 해당 step의 non-pad 토큰 수만큼 step index를 추가
n_tokens = int((resp_ids != self.tokenizer.pad_token_id).sum().item())
turn_token_mask_all[i].extend([_step] * n_tokens)
```

step 0 후: `[0]*50`
step 1 후: `[0]*50 + [1]*40`
step 2 후: `[0]*50 + [1]*40 + [2]*30`

### `gather_rollout_data`에서의 할당 (`rollout_loop.py:318-320`)

```python
# 동일한 trajectory-level mask를 모든 step entry에 할당
data["turn_token_mask"] = np.array(turn_token_mask_all[bs], dtype=object)
```

### 매핑 코드 (`core_algos.py:275-289`)

```python
max_tokens = response_mask.shape[1]  # 단일 step의 response 길이
for i in range(bsz):
    mask = turn_token_mask[i]  # 전체 trajectory 레벨 mask
    for pos, t in enumerate(mask):
        if pos >= max_tokens:
            break
        if t >= 0 and t < len(A_tilde[i]):
            token_advantages[i, pos] = A_tilde[i][t]
```

### 판정: 심각한 버그

각 step entry의 `responses`는 해당 step의 생성 토큰만 포함 (pos 0부터 시작). 하지만 `turn_token_mask`는 trajectory 전체를 순서대로 축적하므로, step $k$의 entry에서 `pos 0`은 turn $k$의 첫 번째 토큰이지만 `turn_token_mask[0]`은 항상 turn 0을 가리킴.

| Step Entry | response tokens 위치 | turn_token_mask[0..max] | 할당되는 advantage | 올바른 advantage | 결과 |
|---|---|---|---|---|---|
| Step 0 (50 tokens) | pos 0..49 | `[0]*50` → turn 0 | $\tilde{A}_{i,0}$ | $\tilde{A}_{i,0}$ | ✓ |
| Step 1 (40 tokens) | pos 0..39 | `[0]*40` → **turn 0** | $\tilde{A}_{i,0}$ | $\tilde{A}_{i,1}$ | **✗ 오류** |
| Step 2 (30 tokens) | pos 0..29 | `[0]*30` → **turn 0** | $\tilde{A}_{i,0}$ | $\tilde{A}_{i,2}$ | **✗ 오류** |

**Step 0만 올바르게 작동하고, 나머지 모든 step은 잘못된 turn의 advantage를 받음.**

### 수정 방안

각 step entry에 해당 step 번호를 별도로 저장하고, 해당 turn의 advantage를 모든 active response token에 균일하게 할당:

```python
# 방법 1: step_index를 별도 저장
for i in range(bsz):
    step_idx = step_index[i]  # 이 entry가 몇 번째 turn인지
    if step_idx < len(A_tilde[i]):
        token_advantages[i, :] = A_tilde[i][step_idx]
token_advantages = token_advantages * response_mask
```

---

## Step 5: Final Objective — PPO Clipped Loss

### 수식

$$\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta}{\pi_{\theta_{old}}} \tilde{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) \tilde{A}_{i,t} \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

### 코드 (`core_algos.py:581-658`)

```python
negative_approx_kl = log_prob - old_log_prob
ratio = torch.exp(negative_approx_kl)
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

### 판정: PPO 클리핑 로직 자체는 올바름

부호 반전(`-advantages`)으로 `torch.maximum`이 원래의 `min`과 동치. 단, 입력되는 `advantages` 값이 Step 4의 버그로 인해 잘못되어 있음.

### 추가 이슈: response_mask vs loss_mask

GRPO branch에서는 multi-turn일 때 `loss_mask`로 교체하지만:
```python
# GRPO branch (ray_trainer.py)
if multi_turn:
    grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
```

MULTITURN_GRPO branch에서는 `response_mask`만 사용:
```python
# MULTITURN_GRPO branch (ray_trainer.py)
response_mask=data.batch["response_mask"],  # loss_mask 미사용
```

---

## 종합 요약

| 구성 요소 | 수식 | 코드 상태 | 심각도 |
|---|---|---|---|
| DPP Diversity $d_{i,t}$ | Leave-one-out log-det ratio | **미구현** (상수 1.0 반환) | 중 (기능 부재) |
| Group Normalization | Flat mean/std over $\mathcal{R}$ | **구조 올바름**, 중복 카운팅 이슈 | 하 (미세한 수치 차이) |
| Temporal Credit Assignment | $\tilde{A}_{i,t} = A_{i,t} + \gamma \tilde{A}_{i,t+1}$ | **정확히 일치** ✓ | - |
| Turn → Token 매핑 | Turn advantage → 해당 turn 토큰 | **심각한 버그**: Step 1+ 잘못된 advantage 할당 | **상 (학습 정확성 영향)** |
| PPO Clipped Loss | Standard PPO objective | **로직 올바름**, 잘못된 입력 | 하 (Step 4 종속) |
| response_mask 선택 | loss_mask 사용 필요 가능 | response_mask만 사용 | 중 (확인 필요) |
