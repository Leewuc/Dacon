# SCPC 2026 Final — Decision Harness

DACON **[2026 삼성 대학생 프로그래밍 경진대회 : AI Challenge](https://dacon.io/competitions/official/236730)** (Final Round) 참가 솔루션.

AI 에이전트가 받은 각 task(사용자 요청 + 기기 상태 + 기록 + 대화 이력)에 대해, **무엇을/누구에게/어떤 범위로 처리할지**를 판정하는 **결정론적 규칙 기반 harness**입니다.

- **Public(screening) LB: 0.8479** — baseline 0.0882 대비 큰 향상, 궤적 0.5984 → 0.8479
- **완전 폐쇄망 · 순수 stdlib** (numpy/pandas/sklearn/외부 API/네트워크 **미사용**) → Hidden 검증환경에서 그대로 실행
- **검증 5종 통과**: 결정론 · task-id 무의존(rename-invariant) · 순서 무관 · 스키마 유효 · 외부 I/O 0

---

## 문제 구조

각 task에 대해 8개 필드로 된 answer를 생성하며, **연쇄 게이팅** 방식으로 채점됩니다:

```
focal(0.18) ──gate──► target(0.12) · control(0.18)
                          │ (곱 = dependent)
                          └──gate──► content_scope(0.17) · policy(0.13) · plan(0.18)
                                     + semantic_response(0.04)
```

`focal`이 틀리면 이하 전부 0 → **focal이 최우선 레버**. `target×control`이 하류 3축을 게이트.

## 접근

| 축 | 핵심 로직 |
|---|---|
| **focal** | marker chain(`latest_phase → marker → ref`) + refcode disambiguation(이력 서사를 절 단위로 파싱해 "승인/제외" 후보 판별) |
| **control** | 지시 cue(proceed/ask/amend/hold)의 **의미 추상화** + record 술어(`_sbu_kind`, `_auth_state`, `session_share_policy`) 분기 |
| **target** | cue → 범주(memory_store/user/recipient), resolved_target 우선 |
| **scope/policy/plan** | control·target으로 결정론 파생 (공개 ontology 값만 사용) |

**설계 원칙**: dev↔screening↔Hidden은 같은 의미의 **다른 패러프레이즈**를 쓰므로, 문자열 정확매칭이 아니라 **의미 술어(semantic predicate)**로 추상화. 규칙은 특정 task를 외우지 않고 **공개 입력을 일반 절차로 해석**하도록 작성.

## 결과 & 배운 것

전체 실험 기록은 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) 참고. 요약:

**효과가 있었던 것**
- focal disambiguation의 패러프레이즈 일반화 → screening 대량 복구 (**최대 상승, +0.155**)
- record 시그니처 기반 control 정교화 (0.90 → 0.99)
- 데이터에서 재발견한 semantic interaction (`sbu=local ∧ auth=ok → status_only`)
- **focal fallback 강건화**: cue-miss 시 fallback이 last-mentioned(승인후보의 실제 위치 분포상 최악, 18%)였던 것을 최빈위치(53%)로 교정 → 관측 점수 불변, Hidden 강건성 확보

**검증했으나 안 통한 것 (정직한 기록)**
- **학습 모델**: 순수 stdlib 로지스틱 회귀로 공정 비교 → 손규칙 압승. 120개 dev로는 학습이 과적합, 주입된 구조가 이김
- **LB 과최적화 회피**: 대회 공지(최종=Hidden calibration)에 따라 screening LB에 맞추는 튜닝을 지양하고, **dev + 의미 정당화**를 채택 기준으로 사용
- **ambiguous 케이스(41%)**: 결정론 규칙으로 푸는 신호를 찾지 못함 (여러 가설을 제출로 실측, 일반화 실패)

**핵심 교훈**: (1) 관측 데이터(dev)로 최적화 ≠ Hidden 최적화, (2) 작은 데이터에선 규칙(구조 주입) > 학습, (3) "dev↓ LB↑" 변경은 Hidden에서 안 버팀.

## 실행

```bash
# 자가 채점 + submission.csv 생성 (dev 로컬 스코어러 내장)
python run.py --data ../data --tag my_run --note "설명"

# 검증 5종
python verification/test_harness.py
```

> `run.py`는 순수 표준 라이브러리만 사용합니다. 분석/실험용 스크립트(로지스틱 회귀, LOSO-CV 등)는 별도이며 제출물이 아닙니다.

## 파일 구조

```
scpc2026-harness/
├── harness.py              # ★ 제출 핵심 — 결정론 규칙 harness (순수 stdlib)
├── run.py                  # 러너 + dev 로컬 스코어러 + submission 생성
├── submission.csv          # 최종 제출본 (700 answers)
├── verification/
│   └── test_harness.py     # 검증 5종 (결정론·rename-invariant·순서무관·스키마·no-IO)
└── docs/
    ├── EXPERIMENT_LOG.md    # 전체 실험 궤적 (버전별·LB 실측·분석)
    ├── RESEARCH_NOTES.md    # 관련 문헌 정합 (Contextual Integrity 등)
    ├── VERIFICATION.md      # 검증 방법론
    ├── run_log.jsonl        # 제출 이력 (dev·LB 점수)
    └── submission_schema.json
```

## 규정 준수

- `uses_external_api: false`, `model_id: scpc-final-fixed-slm-local-facade` 고정
- 외부 pretrained model / tokenizer / embedding **미사용**
- task-id/session-id → 정답 매핑 **없음** (rename-invariant 테스트로 검증)
- 순수 stdlib(`json`, `re`, `typing`)만 import → 검증환경 실행 보장
