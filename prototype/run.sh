#!/usr/bin/env bash
# EmoZid — запуск основного пайплайна (идентификация + тренировка эмоций)
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv"
PIPELINE_DIR="$PROJECT_DIR/pipeline"

if [ ! -d "$VENV" ]; then
    echo "[ERROR] venv не найден. Запусти сначала: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

source "$VENV/bin/activate"
cd "$PROJECT_DIR"

echo "=== EmoZid v4: идентификация + тренировка эмоций ==="
echo "Нажми Q в окне камеры для выхода в меню сценариев"
echo "===================================================="

exec python3 "$PROJECT_DIR/pipeline/rlm.py"
