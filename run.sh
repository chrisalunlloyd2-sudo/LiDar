#!/data/data/com.termux/files/usr/bin/bash
# LiDar launcher — start the emulative LiDAR backend and print the phone URL.
set -u
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"

PORT="${1:-8790}"
LOG="${TMPDIR:-$HOME}/lidar.log"

echo "=== LiDar — emulative LiDAR backend ==="
pkill -f "$HERE/server.py" 2>/dev/null
sleep 1

nohup python3 -u server.py --port "$PORT" > "$LOG" 2>&1 &
PID=$!
sleep 2

CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/health" 2>/dev/null)
if [ "$CODE" = "200" ]; then
  echo "backend UP (HTTP $CODE), pid $PID"
else
  echo "backend FAILED (HTTP $CODE) — log:"
  tail -20 "$LOG"
  exit 1
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$IP" ]; then
  IP=$(python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('1.1.1.1',80));print(s.getsockname()[0]);s.close()" 2>/dev/null)
fi
echo "Open in Chrome: http://${IP:-<phone-ip>}:$PORT"
echo "Stop: pkill -f \"$HERE/server.py\""
