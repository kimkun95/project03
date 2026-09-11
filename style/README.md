# style

플레이스타일 진단(PCA 시각화 + K-means 군집화, 비지도학습) 파이프라인.

## 구성
- `collect_style_matches.py` — (2026-09-07부로 미사용) 유저당 최근 최대 100경기를 깊게
  수집해 `data/style/matches_style.jsonl`에 통짜 저장하는 스크립트였으나, 팀원이 이미
  팀 단위로 평탄화해 수집한 `data/style/match_team_data.csv`를 그대로 원본으로 쓰기로
  결정해 현재 파이프라인에서는 쓰지 않는다. 참고용으로만 남겨둠.
- `preprocess_style.py` — `data/style/match_team_data.csv` -> (match, ouid) 원시 카운트
  -> 유저 단위 집계 -> **패스 모델**(short_pass_ratio, long_pass_ratio, through_pass_ratio,
  driven_ground_pass_ratio) + **슛 모델**(in_penalty_shoot_ratio, heading_shoot_ratio)
  두 그룹의 비율 feature. `dribble_intensity`는 계산은 하되 군집 feature로는 쓰지 않고
  진단 문장 참고용으로만 반환한다(아래 "패스+슛 분리" 참고). 선수 카드 스탯
  (player_stats_final.csv)은 쓰지 않는다 — 팀 단위 pass/shoot/dribble 기록만으로 "유저가
  어떻게 플레이했는가"를 본다. 몰수/오류 경기(`matchEndType != 0`)는 제외한다 — 실측 결과
  `matchEndType`은 0=정상종료(승/무/패 모두 존재), 1=몰수승, 2=몰수패, 4=오류(이 경우
  나머지 스탯 필드가 전부 NULL)로 확인됨.
- `train_style.py` — 패스/헤딩/슛위치 3개 모델을 **각각 독립적으로** StandardScaler
  표준화 후 k=2~6 KMeans를 전부 시도해 실루엣 점수가 가장 높은 k를 사후에 선택
  (CLAUDE.md 원칙 #4: 유형 개수를 미리 정하지 않음). 유저마다 `pass_style_type`,
  `heading_style_type`, `shot_location_style_type` 세 개의 군집 라벨이 나온다.
  PCA(모델별로 별도, feature가 1개뿐인 헤딩/슛위치축은 1차원)는 군집화가 아니라
  시각화 좌표 추출에만 사용. `MODEL_SPECS` 리스트 하나만으로 3개 모델을 루프 처리한다.
- `utils.py` — 저장 공통 함수 (winrate/utils.py와 동일 내용, 트랙 독립성을 위해 복제).
- `position_fit.py` — 진단된 스타일 비율과 `player_stats_final.csv` 카드 스탯 사이의
  코사인 유사도로 포지션별/스쿼드 궁합 점수(position_fit_score)를 계산한다. K-means
  군집 라벨이 아니라 원래의 연속값 비율을 그대로 쓴다. 자세한 설계 근거는 모듈
  docstring 참고.
- `live_diagnosis.py` — "진단하기" 버튼의 백엔드 로직 (CLAUDE.md 핵심 기능 1~5번).
  닉네임 -> ouid -> 최근 30경기 조회 -> 스타일 비율 계산 -> 학습된 K-means로 군집 배정 ->
  가장 최근 "정상종료" 경기로 현재 스쿼드 자동구성 -> position_fit.py로 궁합 점수 ->
  진단 문장까지 `diagnose(nickname)` 하나로 처리한다. API 호출은 `fetch_recent_matches()`
  에서만 발생하고 나머지는 순수 함수라, 대시보드에서 유저가 스쿼드 슬롯을 바꾸면
  `score_squad()`만 다시 불러 API 없이 즉시 재채점할 수 있다.

### 패스/슛 모델 분리 (2026-09-07 결정)
원래 계획은 6개 feature(패스 4개 + 슛 2개)를 한 공간에서 통합 군집화하는 것이었으나,
실제 데이터로 실루엣 점수를 비교해보니:
- 6개 통합 군집화: 실루엣 0.23~0.25 (구조 거의 없음)
- **패스 4개만 단독**: 실루엣 0.33 / **슛 2개만 단독**: 실루엣 0.34 (둘 다 뚜렷이 개선)
- `dribble_intensity`, `possession`(점유율), `block_ratio`(블락/(블락+태클)),
  `shoot_intensity`(경기당 슛 시도 수), `bouncing_lob_pass_ratio`를 어느 조합에 추가해도
  실루엣이 떨어졌다 — 이 값들은 "선택"이 아니라 실력/경기 흐름과 얽혀 있어(예:
  `shoot_intensity`는 division과 상관계수 0.32) 패스/슛 선택 신호를 오히려 희석시킨다.
- K-means 외 GMM/계층적(Agglomerative)/Spectral/DBSCAN도 비교해봤다. 슛 모델은 GMM(k=2)이
  실루엣 0.43으로 더 높게 나왔지만, 소수(29명) 헤딩 특화군만 또렷하게 떼어내고 나머지
  다수를 뭉뚱그리는 결과라 "박스안형/중거리형/헤딩형" 3분류가 나오는 K-means(k=3, 0.34)가
  진단 문장에 더 쓸모 있다고 판단해 **최종적으로 패스/슛 모델 둘 다 K-means만 사용**하기로
  했다. DBSCAN은 대부분(최대 89%)을 "노이즈"로 분류해버려 전원 진단이 필요한 목적에 안
  맞았고, Spectral/Agglomerative는 K-means보다 나은 점이 없었다.

패스 모델과 슛 모델을 분리한 결과 (초기 284명 baseline 기준):
- 패스 모델 k=3: 스루패스형(90명) / 숏패스형(142명) / 드리븐그라운드패스형(52명)
- 슛 모델 k=3: 박스안슛형(113명) / 중거리슛형(104명) / 헤딩형(67명)

### 헤딩축/슛위치축 분리 (2026-09-08 결정)
슛 모델(in_penalty_shoot_ratio + heading_shoot_ratio 2개 feature 통합)도 같은 이유로
다시 쪼갰다. `in_penalty_shoot_ratio`와 `out_penalty_shoot_ratio`는 상관계수 -0.99로
사실상 완전한 여집합이라 군집화 feature로는 out_penalty 하나만 쓰면 충분하다는 것도
이때 확인했다.
- 슛 모델(in_penalty+heading 통합): 실루엣 0.36
- **헤딩축(heading_shoot_ratio 단독)**: 실루엣 0.59 / **슛위치축(out_penalty_shoot_ratio
  단독)**: 실루엣 0.56 (둘 다 뚜렷이 개선)

⚠️ "박스 안 위주"(위치)와 "헤딩 위주"(방식)는 반대 개념이 아니다 — 헤딩은 대부분 박스
안 가까운 거리에서 나오므로 두 값이 자연히 같이 높게 나올 수 있다. 실제로 두 신규
군집 모두 in_penalty_shoot_ratio가 74~83%로 높게 나왔고, 두 군집을 실제로 가르는 건
heading_shoot_ratio(0.17 vs 0.10, 거의 2배 차이)다.

현재(137명, MIN_MATCHES_PER_USER=50) 기준 결과 (`data/style/style_diagnosis_baseline_summary.txt` 참고):
- 패스 모델 k=3: 숏패스형(67명) / 스루패스형(47명) / 드리븐그라운드패스형(23명)
- 헤딩축 k=2: 헤딩을 섞어 쓰는 편(35명) / 발슛 위주(102명)
- 슛위치축 k=2: 박스 안 위주(66명) / 중거리를 섞어 쓰는 편(71명)

⚠️ 재학습마다 K-means가 배정하는 군집 번호(0/1/2...) 순서가 바뀔 수 있어, `live_diagnosis.py`의
`PASS_CLUSTER_LABELS`(하드코딩, 재학습 후 수동 갱신 필요)가 실제로 한 번 안 맞은 채
남아있던 적이 있다. 헤딩축/슛위치축은 `_ordered_cluster_labels()`로 군집 중심값 순서에서
라벨을 자동으로 뽑도록 바꿔 이 문제가 재발하지 않게 했다 — 패스는 feature가 4개라 이
방식이 안 통해 여전히 하드코딩이며, 재학습 후 반드시 라벨을 확인/갱신해야 한다.

## 확정된 설계 결정

### Feature 계산: 티어(division)별 정규화 없음
- `short_pass_ratio`, `long_pass_ratio`, `through_pass_ratio`, `dribble_intensity` 등은
  **유저 본인의 경기 데이터로만 계산한 원본 비율 그대로** 사용한다. 티어 평균을 빼거나 나누는
  정규화/보정을 절대 하지 않는다.
- **왜**: 진단 화면에서 "당신은 이런 스타일입니다"를 그 유저 **실제 플레이 데이터에 기반한
  관찰적 진단(observational diagnosis)**으로 제시하기 위함. 다른 유저 데이터가 섞인 상대값
  (상대주의적 평가)을 쓰지 않는다.
- **알려진 한계 (의도적 수용)**: `long_pass_ratio`(+0.25), `through_pass_ratio`(+0.33)가
  `match_team_data.csv` 기준 티어와 상관관계가 있다는 것을 팀이 이미 확인했다. 이는 K-means
  군집이 스타일 차이뿐 아니라 티어 분포와도 어느 정도 겹칠 수 있음을 의미한다. **이건 "고쳐야
  할 버그"가 아니라 관찰적 진단이라는 설계 선택 때문에 발생하는 트레이드오프이며, 의도적으로
  받아들인 한계다.**
- **사후 검증 시 제약**: 군집화 결과를 분석할 때 군집 × division crosstab을 **참고용으로만**
  찍어보는 것은 괜찮다. 하지만 그 결과(예: "군집과 티어의 상관계수가 높다")를 근거로 정규화
  로직을 다시 넣거나 feature를 수정하면 안 된다. 이는 이미 끝난 논의다.

## 현재 상태
`data/style/match_team_data.csv`(팀원이 팀 단위로 평탄화해 수집한 실 데이터)로 파이프라인이
동작함을 확인했다. 이 CSV는 정제 전 상태이며(팀원이 이후 여러 수집분을 하나로 합쳐 정제할
예정), 파일이 바뀌어도 `preprocess_style.py`가 기대하는 컬럼 구성만 유지되면 그대로
동작한다. 파이프라인 로직 자체는 `tests/fake_data_style.py` +
`tests/test_style_pipeline_smoke.py` 합성 데이터로 검증됨:
```
./venv/Scripts/python.exe tests/test_style_pipeline_smoke.py
```

## 확정된 결정 (2026-09-07 추가)
- 스타일 진단용 유저의 최소 표본 경기 수 하한선: 팀원이 `match_team_data.csv` 자체를
  **70경기 이상 유저로만 정제**해서 제공하기로 함. 코드의 `MIN_MATCHES_PER_USER=50`은
  그 정제를 신뢰하되 다른 파일이 들어오는 경우를 대비한 안전장치일 뿐, 실제 하한은
  70이다.
- 이 70경기 기준은 **학습(K-means fit)용 데이터 수집** 기준이고, 대시보드에서 "진단하기"를
  누를 때 **실시간으로 그 유저 본인의 매치를 조회하는 경우**는 완전히 다른 숫자를 쓴다 —
  **최근 30경기**(CLAUDE.md 핵심 기능 2번 참고). 진단은 이미 학습된 군집 중심에 가장 가까운
  곳을 찾는 것뿐이라 유저 표본이 학습 표본만큼 클 필요가 없고, 매치 상세 조회가 경기 수만큼
  API 호출을 쓰기 때문에(진단 1건당 대략 호출수+2회) 100경기 그대로 쓰면 하루 API 한도
  기준 진단 가능 인원이 너무 적어져(9~10명) 30으로 줄였다.

## 미해결
- `dribble_intensity` 정규화 방식(현재 "경기당 평균 야드") — 실제 데이터 분포를 본 뒤
  재검토.
- 군집이 나온 뒤 각 군집에 붙일 스타일 이름(예: "스루패스 위주형") — `style_type`은
  아직 숫자 군집 id일 뿐, `train_style.py`가 저장하는 `style_diagnosis_baseline_summary.txt`의
  군집별 feature 평균을 보고 팀이 사후에 정한다.
