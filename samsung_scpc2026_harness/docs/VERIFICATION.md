# 상위권 코드 검증 대비 — 준수 증빙

예선 종료 후 상위권 검증에서 확인하는 항목(재현성, fixed SLM interface 사용,
외부 모델/API 미사용, 하드코딩 여부, 비공개 task stream 일반화)에 대한 증빙.

## 1. 재현성 (Reproducibility)

- 결정론적 규칙 harness. 난수·시각·상태 의존 없음. `meta.temperature=0.0`, `seed`.
- `python verification/test_harness.py --data ../data` → **동일 입력 스트림에
  대해 answer가 byte-identical**([1] determinism PASS).
- 조직위 검증 스트림 실행:
  ```bash
  python run.py --stream <private_tasks>.jsonl --answers out.json
  ```
  (조직위 runner가 `FinalHarness.answer_task(task, session)`를 세션/turn 순서로
  호출하는 방식과 동일. per-task 순수성은 [3] order-independence로 보장.)

## 2. Fixed SLM interface 사용

- `FinalHarness`는 `self.slm = FixedSLMClient()`를 통해 `summarize_task(task)`를
  evidence helper로 호출한다(`harness.py`). 정답을 직접 얻는 용도가 아니라
  risk/redaction/confirmation 보조 신호로만 사용.
- `meta.fixed_slm_policy="local_fixed_slm_only"`,
  `meta.model_id="scpc-final-fixed-slm-local-facade"`.

## 3. 외부 모델/API 미사용

- `harness.py` import는 **stdlib만**: `json`, `re`, `typing`.
  네트워크/외부모델 라이브러리(requests·urllib·socket·openai·anthropic·torch·
  transformers·vllm) import 0건 (정적 grep 확인).
- `meta.uses_external_api=false`.
- 검증 테스트 [5] no-external-io PASS.

## 4. 하드코딩 없음 (일반화)

- `harness.py`에 task-id / object-id / WM 코드 / 공개 예시 문장 **리터럴 0건**
  (정적 grep 확인). gold answer 표 없음.
- 강한 동적 증빙 — 검증 테스트 [2] **no-id-dependence**:
  screening 700개의 모든 `task_/obj_/rec_/sess_` id와 `WM-####`를 새 네임스페이스로
  일괄 rename해도 control·scope·policy·plan 결정 shape이 **완전히 동일**.
  → id를 외우는 게 아니라 record 타입·값·focal 속성·지시절의 **의미 구조**만
  읽는다는 증거.
- 모든 판단 feature는 일반값(record type/value, `contains` 필드, 지시절 의미 범주)
  이며 특정 공개 항목에 종속되지 않음.

## 5. dev↔screening 일반화 설계

- dev와 screening은 지시절이 **서로 다른 표현의 패러프레이즈**(공유 tail 0개)인데
  의미 범주는 동일. control cue를 양쪽 표현을 모두 포괄하는 의미 원자로 구성해
  한쪽 표현에 과적합되지 않도록 설계(`_PROCEED_LOCAL`/`_ASK_CUES`/`_AMEND_CUES`/
  `_HOLD_CUES`). 비공개 stream의 또 다른 패러프레이즈에도 같은 원자가 매칭되도록 함.

## 검증 실행

```bash
python verification/test_harness.py --data ../data
# [1] determinism / [2] no-id-dependence / [3] order-independence
# [4] schema-validity / [5] no-external-io  → 전부 PASS
```

정적 점검:
```bash
grep -nE "final_dev|task_[0-9a-f]{6}|obj_[0-9a-f]{6}|WM-[0-9]{4}" harness.py   # 0건
grep -nE "requests|urllib|openai|anthropic|torch|transformers" harness.py       # 0건
```
