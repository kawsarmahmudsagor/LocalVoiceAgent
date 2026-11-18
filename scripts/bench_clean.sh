#!/usr/bin/env bash
# After Phase B is installed: delete bench venvs and model folders of families with no shortlisted
# entry (v2 spec §3.3). Prints what it would remove and asks first.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f server/voicebot/catalog.toml ] || { echo "No server/voicebot/catalog.toml: run voicebot.bench.resolve first." >&2; exit 1; }

mapfile -t targets < <(.venv/bin/python - <<'PY'
from pathlib import Path
from voicebot.catalog import load_catalog
pool = load_catalog(Path("bench/pool.toml"))
keep = {e.family for e in load_catalog(Path("server/voicebot/catalog.toml"))}
model_dirs = {"sherpa": "models/sherpa-onnx", "kroko": "models/kroko-onnx",
              "transcribe-cpp": "models/transcribe-cpp", "piper": "models/piper"}
for fam in sorted({e.family for e in pool} - keep):
    for p in (f"venvs/bench-{fam}", model_dirs.get(fam)):
        if p and Path(p).exists():
            print(p)
PY
)

if [ ${#targets[@]} -eq 0 ]; then echo "Nothing to remove."; exit 0; fi
echo "This will permanently delete:"
for t in "${targets[@]}"; do echo "  $t ($(du -sh "$t" | cut -f1))"; done
read -r -p "Delete these? [y/N] " answer
case "$answer" in
  y|Y) for t in "${targets[@]}"; do rm -rf -- "$t"; echo "removed $t"; done; df -h ~ | tail -1 ;;
  *) echo "Nothing removed." ;;
esac
