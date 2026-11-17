#!/usr/bin/env bash
# Download the PUBLIC external datasets and reference repos used by research/metodos/.
# Everything lands in data_external/ (gitignored). Safe to re-run: finished files are skipped
# and partial downloads resume. Nothing here touches the restricted BAR-ATH bundle.
#
#   bash scripts/download_public_datasets.sh            # everything (~3 GB + SkillCorner LFS)
#   bash scripts/download_public_datasets.sh sportec    # one source: sportec|metrica|skillcorner|pathcrf|ballradar
#
# Sources and licences:
#   sportec      Sportec Open DFL (Bassek et al. 2025), figshare 10.6084/m9.figshare.28196177, CC-BY 4.0
#   metrica      github.com/metrica-sports/sample-data (attribution requested)
#   skillcorner  github.com/SkillCorner/opendata, MIT (tracking via git-lfs)
#   pathcrf      github.com/hyunsungkim-ds/pathcrf, MPL-2.0 (code + 5 trained checkpoints)
#   ballradar    github.com/hyunsungkim-ds/ballradar (code, no checkpoints)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${MTAH_EXTERNAL_DIR:-$ROOT/data_external}"
FIGSHARE_ARTICLE=28196177
mkdir -p "$OUT"

clone() {  # clone <url> <dir> : shallow clone, or fast-forward if already there
    if [ -d "$2/.git" ]; then
        git -C "$2" pull --ff-only
    else
        git clone --depth 1 "$1" "$2"
    fi
}

sportec() {
    local dir="$OUT/sportec_dfl"
    mkdir -p "$dir"
    curl -fsS "https://api.figshare.com/v2/articles/$FIGSHARE_ARTICLE/files?page_size=100" \
        > "$dir/_figshare_files.json"
    python3 - "$dir" <<'EOF'
import hashlib, json, os, subprocess, sys
d = sys.argv[1]
files = json.load(open(os.path.join(d, "_figshare_files.json")))
for f in files:
    path = os.path.join(d, f["name"])
    if os.path.exists(path) and os.path.getsize(path) == f["size"]:
        print("ok  ", f["name"]); continue
    print("get ", f["name"], round(f["size"] / 1e6, 1), "MB", flush=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-sS", "-o", path,
                    f["download_url"]], check=True)
bad = []
for f in files:
    path = os.path.join(d, f["name"])
    md5 = hashlib.md5()
    with open(path, "rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest() != f.get("computed_md5", md5.hexdigest()):
        bad.append(f["name"])
print("sportec: %d files, md5 mismatches: %s" % (len(files), bad or "none"))
sys.exit(1 if bad else 0)
EOF
}

metrica()     { clone https://github.com/metrica-sports/sample-data.git "$OUT/metrica_sample_data"; }
skillcorner() { clone https://github.com/SkillCorner/opendata.git "$OUT/skillcorner_opendata"
                git -C "$OUT/skillcorner_opendata" lfs pull; }
pathcrf()     { clone https://github.com/hyunsungkim-ds/pathcrf.git "$OUT/pathcrf_repo"; }
ballradar()   { clone https://github.com/hyunsungkim-ds/ballradar.git "$OUT/ballradar_repo"; }

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(metrica pathcrf ballradar skillcorner sportec)
for t in "${targets[@]}"; do
    echo "== $t"
    "$t"
done
du -sh "$OUT"/* 2>/dev/null
