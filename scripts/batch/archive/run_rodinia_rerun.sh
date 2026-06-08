#!/bin/bash
set -euo pipefail
cd /home/samyak/Desktop/parbench_sam
source env_parbench/bin/activate

SPECS=(
  rodinia-cfd-opencl
  rodinia-huffman-cuda
  rodinia-lavamd-cuda
  rodinia-streamcluster-opencl
  rodinia-cfd-cuda
  rodinia-hybridsort-cuda
  rodinia-kmeans-cuda
  rodinia-mummergpu-cuda
  rodinia-mummergpu-omp
  rodinia-pathfinder-opencl
)

mkdir -p results/rodinia/logs

for SPEC in "${SPECS[@]}"; do
  echo "=== Running $SPEC ==="
  LOGFILE="results/rodinia/logs/${SPEC}.log"
  JSONFILE="results/rodinia/logs/${SPEC}.json"

  set +e
  python3 -m harness --json -v verify "specs/${SPEC}.json" --config correctness > "${LOGFILE}" 2>&1
  EXIT_CODE=$?
  set -e

  # Extract JSON using python (handles multi-brace content properly)
  python3 -c "
import sys, re, json

with open('${LOGFILE}') as f:
    content = f.read()

# Find first '{' that starts a JSON object at start of line
matches = list(re.finditer(r'^\{', content, re.MULTILINE))
if not matches:
    print('NO_JSON_FOUND', file=sys.stderr)
    sys.exit(1)

# Try each match position to find valid JSON
for m in matches:
    start = m.start()
    # Try to parse from here
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(content, start)
        print(json.dumps(obj, indent=2))
        sys.exit(0)
    except Exception:
        continue

print('NO_VALID_JSON', file=sys.stderr)
sys.exit(1)
" > "${JSONFILE}" 2>/dev/null || echo "  (no JSON output - build/run failed)"

  if python3 -c "import json; json.load(open('${JSONFILE}'))" 2>/dev/null; then
    STATUS=$(python3 -c "import json; d=json.load(open('${JSONFILE}')); b=d.get('build',{}).get('status','?'); r=list(d.get('runs',{}).values()); rs=r[0].get('status','?') if r else '-'; v=d.get('verification',{}).get('status','?'); print(f'BUILD:{b} RUN:{rs} VERIFY:{v}')")
    echo "  OK: $STATUS"
  else
    echo "  FAIL: no valid JSON produced"
    # Create a minimal fail JSON
    python3 -c "
import json, sys
with open('${LOGFILE}') as f:
    log = f.read()
# Check for build fail indicators
if 'BUILD: FAIL' in log or 'Error' in log:
    # Create minimal fail record
    data = {
        'spec_id': '${SPEC}',
        'build': {'status': 'fail', 'duration_seconds': 0, 'stdout': '', 'stderr': log[:500]},
        'runs': {},
        'verification': {'status': 'fail', 'strategy_used': 'exit_code', 'details': 'build failed'},
        'metrics': []
    }
    print(json.dumps(data, indent=2))
" > "${JSONFILE}" 2>/dev/null || true
  fi
done

echo ""
echo "=== Done ==="
