# 파일 역할

- app.py : 앱 시작 / 화면 이동
- search.py : 검색 화면
- result.py : 분석 결과 화면
- api.py : live_diagnosis 연결
- util.py : 공통 함수
- board.py : 축구 전술보드
- main.css : 디자인

## 같이 두어야 하는 기존 파일

프로젝트 상황에 맞게 아래 파일도 같은 프로젝트에 유지하세요.

- .env
- spid.json
- live_diagnosis.py 또는 style/live_diagnosis.py
- player_stats_final.csv
- 모델 파일들

## 실행

```bash
streamlit run app.py
```
