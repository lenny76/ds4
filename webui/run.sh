#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${DS4_MODEL:-$ROOT/ds4flash.gguf}
THREADS=${DS4_THREADS:-40}
CTX=${DS4_CTX:-4096}
MODEL_HOST=${DS4_SERVER_HOST:-127.0.0.1}
MODEL_PORT=${DS4_SERVER_PORT:-8000}
UI_HOST=${DS4_UI_HOST:-0.0.0.0}
UI_PORT=${DS4_UI_PORT:-8080}
WARM=${DS4_WARM_WEIGHTS:-1}

if [ ! -x "$ROOT/ds4-server" ]; then
    echo "run.sh: missing $ROOT/ds4-server; run 'make cpu' first" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "run.sh: model not found: $MODEL" >&2
    echo "Set DS4_MODEL to the GGUF path." >&2
    exit 1
fi

check_port() {
    if ! python3 -c 'import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((sys.argv[1], int(sys.argv[2])))
s.close()' "$1" "$2" 2>/dev/null; then
        echo "run.sh: $3 port $1:$2 is already in use" >&2
        exit 1
    fi
}

# Fail before loading the model if either listener cannot be created.
check_port "$MODEL_HOST" "$MODEL_PORT" "model server"
check_port "$UI_HOST" "$UI_PORT" "Web UI"

server_pid=
cleanup() {
    trap - INT TERM EXIT
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

set -- "$ROOT/ds4-server" --cpu -t "$THREADS" -m "$MODEL" --ctx "$CTX" \
    --host "$MODEL_HOST" --port "$MODEL_PORT"
if [ "$WARM" = 1 ]; then
    set -- "$@" --warm-weights
fi

echo "Starting DwarfStar model server on $MODEL_HOST:$MODEL_PORT..."
DS4_CPU_V41_EXPERIMENTAL=${DS4_CPU_V41_EXPERIMENTAL:-1} "$@" &
server_pid=$!

echo "Waiting for model startup and weight warmup..."
while ! python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1).read()' \
    "http://$MODEL_HOST:$MODEL_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" || true
        echo "run.sh: DwarfStar model server stopped during startup" >&2
        exit 1
    fi
    sleep 1
done

echo "Starting Web UI on http://$UI_HOST:$UI_PORT"
python3 "$ROOT/webui/server.py" --host "$UI_HOST" --port "$UI_PORT" \
    --upstream "http://$MODEL_HOST:$MODEL_PORT"
