"""모델 저장(joblib) 및 파일 저장용 공통 함수."""

import json
import os

import joblib


def ensure_dir(path):
    """path가 디렉터리 경로면 그대로, 파일 경로면 부모 디렉터리를 만든다."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_model(model, path):
    """sklearn Pipeline 등 학습된 모델을 joblib으로 저장한다."""
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
