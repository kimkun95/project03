"""모델 저장(joblib) 및 파일 저장용 공통 함수.

winrate/utils.py와 내용이 동일하다. 두 트랙(승률/스타일)은 담당자가 달라 폴더를
독립적으로 유지하기로 했으므로, 공용 모듈로 묶지 않고 그대로 복제해서 쓴다.
"""

import json
import os

import joblib


def ensure_dir(path):
    """path가 디렉터리 경로면 그대로, 파일 경로면 부모 디렉터리를 만든다."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_model(model, path):
    """sklearn Pipeline/추정기를 joblib으로 저장한다."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)


def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
