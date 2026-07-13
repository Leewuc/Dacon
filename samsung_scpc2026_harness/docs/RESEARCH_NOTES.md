# 연구 노트 — 결정 harness 관련 프레임워크와 적용

benchmark의 decision 축(focal/target/control/scope/policy/plan)을 뒷받침하는 문헌을
조사해 우리 규칙과 대조. 결론: **우리 역설계 구조가 문헌 합의와 일치**하며, 문헌은
새 dev-점수 레버보다 "구조 검증 + 과적합 경고"를 준다.

## 1. Contextual Integrity (CI) — scope/policy의 이론적 뼈대
정보흐름 = (data subject, sender, **recipient**, **info type**, **transmission principle/purpose**).
규범과 일치하면 허용, 파라미터가 규범 밖으로 바뀌면 위반.
- **ConfAIde**(Mireshghallah, ICLR2024): 민감도 인식 ≠ 올바른 행동. "비밀유지" 규칙이
  helpfulness를 지배해야. → consent는 **recipient·purpose 스코프**. X를 A에게 허용 ≠ X를 B에게.
- **AirGapAgent**(Bagdasarian 2024): **task 기준 최소화** 후 recipient 요구와 교집합(합집합 금지).
  요청 필드가 최소집합 밖이면 **escalate(ask), 자동공유 금지**.
- **PrivacyLens**(Shao NeurIPS2024): 판단이 아니라 **송출 산출물**을 게이트. 맥락에서 주운
  부수정보는 task의 transmission principle이 커버할 때만 공유.
- 적용: 우리 `build_content_scope`/`build_policy`가 control+focal-contains+recipient로
  최소범위를 산출하는 것과 정합. `doctor_note_forbidden`→hold, external+민감→redacted 는 CI 규범.

## 2. Agent guardrail — control(proceed/amend/hold/ask)
- **ToolEmu**(Ruan ICLR2024): **가역성×심각도** 게이트. 비가역+심각 → 확인 필요.
- **Selectively Quitting**(Kim NeurIPS2025) — 우리 control과 직결되는 트리거:
  `대상 모호 → ask · 비가역+심각 → ask/hold · consent/전제 부재·불명 → hold · 안전한 정보수집 가능 → proceed(그 단계 먼저)`.
- 적용: 우리 control 우선순위(모호성→ask, 전제붕괴→hold, 로컬→proceed)와 동형.
  payment(비가역)→ask, health external→hold 이미 반영.

## 3. Stale-context — "전제 붕괴" 처리 (STALE, Chao 2025)
진부화 3종: **직접 override**(값 교체) / **시간 만료**(조건부 승인이 지남) / **연쇄 무효**(상위 사실 무효→의존 승인 무효).
- 적용: 우리 `_STRONG_HOLD`(무효화/전제 사라짐 → hold) vs `_ASK_CUES`(미확정 → ask) 구분이
  STALE의 "override=stop vs knowledge-gap=re-confirm"과 정합. `target_changed_after_turn`=직접 override.

## 4. 규칙 귀납 — 소표본(n=120) 과적합 관리
Molnar IML ch.10 / RIPPER / CN2. 원칙: OneR 베이스라인 → CV 필수 → **min-support≥5, ≤3조건,
규칙 수 최소, 순서형 결정목록 + 제한적 default**. 통계적 규칙은 도메인 prior로 손감사.
- **적용/경고**: 결정트리 5-fold CV로 control **0.80**(train 0.967) 측정. 내 손규칙은 train 0.983 —
  **일반화 상한(≈0.80~0.90) 근처, 추가 dev 튜닝은 과적합 위험**. target-범주 CV **0.992**(견고),
  scope-mode CV 0.70. → control은 동결, 검증은 LB로.

## 종합 결정목록(문헌 합의, 우리 구현과 대조용)
```
1 대상 모호(≥2 후보)                         → ask
2 승인/전제 진부화(override|만료|연쇄)        → ask 재확인, scope ≤ status_only
3 민감(health/financial/비밀)+외부+consent無  → scope none, hold/amend
4 비가역 & 심각                               → ask/hold
5 결과 못 정할 정보부족                        → hold
6 규범 일치 & 가역/경미                        → proceed(최소범위)
DEFAULT                                       → hold, status_only
```
scope 사다리: raw→summary→redacted→status_only→none (항상 최소 필요범위).

## 출처
ConfAIde https://confaide.github.io/ · AirGapAgent https://arxiv.org/abs/2405.05175 ·
PrivacyLens https://openreview.net/forum?id=CxNXoMnCKc · ToolEmu https://arxiv.org/abs/2309.15817 ·
Selectively Quitting https://arxiv.org/abs/2510.16492 · STALE https://arxiv.org/abs/2605.06527 ·
Decision Rules https://christophm.github.io/interpretable-ml-book/rules.html
