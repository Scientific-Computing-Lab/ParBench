#!/usr/bin/env bash
# =============================================================================
# render_figure.sh  —  Regenerate a TikZ/pgfplots figure fragment to a PNG
#                      preview, and check it against a frozen "golden" image.
#
# Why this exists
#   The figure .tex files in this directory are *fragments* (no \documentclass);
#   they rely on the paper preamble (pb* colours, model-name macros, pgfplots).
#   The machine's default `texbin` is TeX Live 2023basic, which has NO pgfplots,
#   so plain `pdflatex` fails. This script finds a TeX Live that HAS pgfplots,
#   wraps the fragment, compiles, crops with Ghostscript (pdfcrop is absent),
#   rasterises with pdftocairo, then compares the result to a golden reference.
#
# What the check DOES and does NOT guarantee  (read before trusting it)
#   The "golden" (references/<name>.png) is a frozen PRIOR render that you
#   approved with --set-golden. It is NOT the original screenshot.
#   The comparison reports TWO numbers: a global similarity (catches layout /
#   spacing / scale drift) AND a per-tile content delta (catches even a single
#   status cell changing colour — global MAE alone is blind to that because the
#   figure is ~80% white/grey background). The gate therefore verifies "this
#   render still matches the approved golden render".
#   It does NOT prove the preview matches the PAPER-compiled figure: for tuned
#   profiles (e.g. f3) the preview is deliberately enlarged (bigger fonts/gaps,
#   wider page) and differs from the paper's column-sized figure. It does NOT
#   prove the DATA is correct — a golden frozen from stale data keeps passing.
#   Re-freeze with --set-golden whenever the figure legitimately changes.
#
# Usage
#   ./render_figure.sh <figure-name-or-path> [options]
#     <figure-name>      e.g. f3_kernel_model_heatmap_unified  (".tex" optional)
#   Options
#     --dpi N            output resolution (default 600)
#     --tune             apply the spacious "preview" tuning (bigger fonts +
#                        wider inter-panel gaps). Auto-ON for known profiles.
#     --raw              force faithful render (disable any auto-tune profile)
#     --reference IMG    compare against IMG instead of references/<name>.png
#     --threshold P      similarity %% below which the check FAILS (default 95)
#     --set-golden       after rendering, (re)freeze result as references/<name>.png
#     --no-compare       skip the reference comparison entirely
#     -h | --help        show this help
#
# Outputs (all under preview_output/, which is git-ignored by project policy)
#     preview_<name>.png              the rendered preview
#     <name>_vs_reference.png         side-by-side montage (when comparing)
#
# First-time setup for a new figure:
#     ./render_figure.sh <name> --set-golden     # approve & freeze the look
#   Thereafter just:
#     ./render_figure.sh <name>                  # renders + checks vs golden
# =============================================================================
set -euo pipefail

# --- locate ourselves so the script works from any cwd -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Tooling moved OUT of the paper tree (2026-08-19 consolidation): the paper
# folder holds only files the manuscript compiles; fragments live there, the
# render tooling lives here in docs/figures/figure_tooling/.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FIG_DIR="$REPO_ROOT/docs/paper/NeurIPS_ready_version/figures/figures_tek_version"
OUT_DIR="$SCRIPT_DIR/preview_output"
REF_DIR="$SCRIPT_DIR/references"
# (D5) single source of truth for colours/macros: the paper's own files. We
# grep colours out of main and \input the macros file directly, so the
# preview can't silently drift from the paper. Fallbacks if absent.
PAPER_ROOT="$REPO_ROOT/docs/paper/NeurIPS_ready_version"
MAIN_TEX="$PAPER_ROOT/main_neurips.tex"
MACROS_TEX="$PAPER_ROOT/sections/macros.tex"
BUILD_DIR=""                       # set per-run via mktemp (see below)
PY="$(command -v python3 || true)" # assignment masks command-v failure under set -e
[ -n "$PY" ] || { echo "ERROR: python3 not found on PATH (activate env_parbench?)." >&2; exit 1; }

# --- defaults ----------------------------------------------------------------
DPI=600
TUNE=auto          # auto | on | off
REFERENCE=""
THRESHOLD=95
SET_GOLDEN=0
COMPARE=1

# Print the leading comment block (everything after the shebang up to the first
# non-comment line). Robust to header edits — no hardcoded line range.
usage() {
  awk 'NR>1 { if ($0 ~ /^#/) { sub(/^# ?/,""); print } else exit }' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0;; esac
NAME_ARG="$1"; shift
# require a value for the next option (avoids '$2: unbound' under set -u)
need_val() { [ "$2" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; exit 1; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dpi)        need_val "$1" "$#"; DPI="$2"; shift 2;;
    --tune)       TUNE=on; shift;;
    --raw)        TUNE=off; shift;;
    --reference)  need_val "$1" "$#"; REFERENCE="$2"; shift 2;;
    --threshold)  need_val "$1" "$#"; THRESHOLD="$2"; shift 2;;
    --set-golden) SET_GOLDEN=1; shift;;
    --no-compare) COMPARE=0; shift;;
    -h|--help)    usage 0;;
    *) echo "ERROR: unknown option '$1'" >&2; usage 1;;
  esac
done

# validate numeric options (they flow straight into pdftocairo / python)
[[ "$DPI" =~ ^[0-9]+$ ]] && [ "$DPI" -gt 0 ] \
  || { echo "ERROR: --dpi must be a positive integer (got '$DPI')." >&2; exit 1; }
[[ "$THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || { echo "ERROR: --threshold must be a number 0-100 (got '$THRESHOLD')." >&2; exit 1; }

# --- normalise the figure name / path ----------------------------------------
NAME="$(basename "$NAME_ARG")"; NAME="${NAME%.tex}"
SRC="$FIG_DIR/$NAME.tex"
[ -f "$SRC" ] || { echo "ERROR: figure source not found: $SRC" >&2; exit 1; }

# --- per-figure profiles: which figures need the spacious "preview" tuning ----
# Add new entries here as you approve more figures.
profile_wants_tune() {
  case "$1" in
    f3_kernel_model_heatmap_unified) return 0;;
    *) return 1;;
  esac
}
if [ "$TUNE" = auto ]; then
  if profile_wants_tune "$NAME"; then TUNE=on; else TUNE=off; fi
fi

# --- find a pdflatex that actually has pgfplots ------------------------------
find_pdflatex() {
  local cand dir found=""
  # the glob expands ascending (2020 < 2022 < 2023); keep the LAST pgfplots-
  # capable match => newest TeX Live, without parsing `ls` output.
  for cand in /usr/local/texlive/20*/bin/*/pdflatex; do
    [ -x "$cand" ] || continue
    dir="$(dirname "$cand")"
    "$dir/kpsewhich" pgfplots.sty >/dev/null 2>&1 && found="$cand"
  done
  [ -n "$found" ] && { echo "$found"; return 0; }
  if command -v pdflatex >/dev/null 2>&1 && kpsewhich pgfplots.sty >/dev/null 2>&1; then
    command -v pdflatex; return 0
  fi
  return 1
}
PDFLATEX="$(find_pdflatex || true)"
[ -n "$PDFLATEX" ] || { echo "ERROR: no pdflatex with pgfplots found (need e.g. TL2022)." >&2; exit 1; }
command -v gs >/dev/null         || { echo "ERROR: ghostscript (gs) not found." >&2; exit 1; }
command -v pdftocairo >/dev/null || { echo "ERROR: pdftocairo not found (brew install poppler)." >&2; exit 1; }

echo "figure       : $NAME"
echo "pdflatex     : $PDFLATEX"
echo "tuning       : $TUNE   dpi: $DPI"

mkdir -p "$OUT_DIR" "$REF_DIR"
# per-run build dir (mktemp -> no concurrent-run collisions); removed on success,
# kept on failure so the pdflatex log survives for debugging.
BUILD_DIR="$(mktemp -d "$OUT_DIR/.build.XXXXXX")"
cleanup() {
  local rc=$?
  [ -n "${BUILD_DIR:-}" ] && [ -d "$BUILD_DIR" ] || return 0
  if [ "$rc" -eq 0 ]; then rm -rf "$BUILD_DIR"
  else echo "build kept for debugging: $BUILD_DIR" >&2; fi
}
trap cleanup EXIT

# --- build the figure body (tuned copy or the source verbatim) ---------------
BODY="$BUILD_DIR/${NAME}_body.tex"
if [ "$TUNE" = on ]; then
  "$PY" - "$SRC" "$BODY" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
# Spacious-preview tuning (presentation only; the paper source is untouched).
# Calibrated on the f3 three-panel heatmaps; each replace is a safe no-op when
# its pattern is absent, so the same pass is harmless on other figures.
hits = 0
for a, b in [
    ("{0.30\\textwidth}",            "{4.7cm}"),                       # panel widths
    ("\\fontsize{2.7pt}{3.2pt}",     "\\fontsize{4.2pt}{5pt}"),        # x-tick font
    ("\\fontsize{3.5pt}{4.5pt}",     "\\fontsize{4.6pt}{5.6pt}"),      # y-tick font
    ("title={\\scriptsize\\bfseries","title={\\normalsize\\bfseries"), # panel titles
]:
    hits += t.count(a)
    t = t.replace(a, b)
# Standalone \hfill between panels -> explicit wide gap.
hits += len(re.findall(r"(?m)^\\hfill\s*$", t))
t = re.sub(r"(?m)^\\hfill\s*$", r"\\hspace{2.0cm}", t)
open(dst, "w").write(t)
if hits == 0:
    sys.stderr.write("WARNING: --tune matched 0 patterns in the source .tex; it may have "
                     "changed since the profile was calibrated. The preview will likely "
                     "diverge from the golden (the compare step should flag it).\n")
PY
else
  cp "$SRC" "$BODY"
fi

# --- geometry: wide page for tuned multipanel; NeurIPS column for faithful ----
if [ "$TUNE" = on ]; then
  GEOM="paperwidth=9.5in,paperheight=7.5in,margin=0.4in"
else
  GEOM="paperwidth=6.5in,paperheight=11in,margin=0.5in"   # textwidth=5.5in (NeurIPS)
fi

# Baked-in fallbacks — used ONLY when the paper files can't be found (e.g. a
# worktree checkout). The live paper files are the source of truth (D5).
fallback_colors() {
  cat <<'EOF'
\definecolor{pbPass}{RGB}{44,160,44}
\definecolor{pbBuildFail}{RGB}{214,39,40}
\definecolor{pbRunFail}{RGB}{255,127,14}
\definecolor{pbVerifyFail}{RGB}{148,103,189}
\definecolor{pbExtractionFail}{RGB}{140,86,75}
\definecolor{pbNA}{RGB}{200,200,200}
\definecolor{pbTeal}{RGB}{23,190,207}
\definecolor{pbTealDark}{RGB}{15,120,140}
\definecolor{pbTealLight}{RGB}{158,218,229}
\definecolor{pbRose}{RGB}{227,119,194}
\definecolor{pbOrange}{RGB}{255,127,14}
\definecolor{pbYellow}{RGB}{255,215,0}
\definecolor{pbGray}{RGB}{150,150,150}
\definecolor{pbRodinia}{RGB}{31,119,180}
\definecolor{pbXSBench}{RGB}{255,127,14}
\definecolor{pbRSBench}{RGB}{44,160,44}
\definecolor{pbMixbench}{RGB}{214,39,40}
\definecolor{pbHeCBench}{RGB}{148,103,189}
EOF
}
fallback_macros() {
  cat <<'EOF'
\newcommand{\qwen}{Qwen~3.5 397B-A17B}
\newcommand{\qwenshort}{Qwen~3.5}
\newcommand{\gptnew}{GPT-5.4}
\newcommand{\codex}{GPT-5.3-codex}
EOF
}
fallback_pgfstyles() {
  cat <<'EOF'
\pgfplotsset{
  parbench compact/.style={
    width=\columnwidth, height=5.5cm,
    enlarge x limits=0.08,
    tick pos=left,
    legend style={font=\footnotesize, draw=gray!50},
    tick label style={font=\footnotesize},
    label style={font=\footnotesize},
    title style={font=\small\bfseries, align=center},
    ymajorgrids=true,
    grid style={draw=gray!20},
    clip=false,
  },
  parbench heatmap/.style={
    width=\columnwidth, height=\columnwidth,
    enlargelimits=false,
    tick label style={font=\tiny, rotate=45, anchor=east},
    title style={font=\small\bfseries},
    colorbar style={font=\tiny},
    axis on top,
  },
}
EOF
}

# --- the standalone wrapper: reuse the paper's REAL colours + macros (D5) -----
WRAP="$BUILD_DIR/${NAME}_wrap.tex"
cat > "$WRAP" <<EOF
\\documentclass[11pt]{article}
\\usepackage[$GEOM]{geometry}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\usepackage{etoolbox}
\\usepackage{array}
\\usepackage{amssymb}
EOF
# colours: pulled live from main_neurips.tex (so they can't drift from the paper)
if [ -f "$MAIN_TEX" ] && grep -q 'pbPass' "$MAIN_TEX"; then
  { echo "% colours sourced live from $MAIN_TEX"
    grep -E '^\\definecolor|^\\colorlet' "$MAIN_TEX"; } >> "$WRAP"
else
  echo "WARNING: $MAIN_TEX not found/usable; using baked-in colour fallback." >&2
  fallback_colors >> "$WRAP"
fi
# pgfplots styles (parbench compact / heatmap): pulled live from main_neurips.tex,
# same D5 single-source rule as the colours above (f5/f6/f7 use `parbench compact`).
if [ -f "$MAIN_TEX" ] && grep -q 'parbench compact' "$MAIN_TEX"; then
  { echo "% pgfplots styles sourced live from $MAIN_TEX"
    awk '/^\\pgfplotsset\{[[:space:]]*$/{cap=1} cap{print} cap&&/^\}[[:space:]]*$/{exit}' "$MAIN_TEX"; } >> "$WRAP"
else
  echo "WARNING: $MAIN_TEX not usable for pgfplots styles; using baked-in fallback." >&2
  fallback_pgfstyles >> "$WRAP"
fi
# macros: \input the real macros.tex (so model names can't drift from the paper)
if [ -f "$MACROS_TEX" ]; then
  echo "\\input{$MACROS_TEX}" >> "$WRAP"
else
  echo "WARNING: $MACROS_TEX not found; using baked-in macro fallback." >&2
  fallback_macros >> "$WRAP"
fi
cat >> "$WRAP" <<EOF
\\pagestyle{empty}
\\begin{document}
\\centering
\\input{$BODY}
\\end{document}
EOF

# --- compile -----------------------------------------------------------------
echo "compiling ..."
if ! "$PDFLATEX" -interaction=nonstopmode -halt-on-error \
       -output-directory="$BUILD_DIR" "$WRAP" > "$BUILD_DIR/${NAME}.log" 2>&1; then
  echo "ERROR: pdflatex failed. Last lines:" >&2
  tail -n 20 "$BUILD_DIR/${NAME}.log" >&2
  exit 1
fi
PDF="$BUILD_DIR/${NAME}_wrap.pdf"

# --- tight crop via Ghostscript bbox (pdfcrop is not installed) --------------
# Capture (don't read from a process-substitution: that hides gs/awk failures
# from set -e/pipefail), then validate we got four numeric coordinates.
BBOX="$(gs -q -dBATCH -dNOPAUSE -sDEVICE=bbox "$PDF" 2>&1 \
          | awk '/HiResBoundingBox/ {print $2, $3, $4, $5; exit}')" \
  || { echo "ERROR: ghostscript bbox pass failed for $PDF" >&2; exit 1; }
read -r LLX LLY URX URY <<< "$BBOX"
for v in "${LLX:-}" "${LLY:-}" "${URX:-}" "${URY:-}"; do
  [[ "$v" =~ ^-?[0-9]+([.][0-9]+)?$ ]] \
    || { echo "ERROR: ghostscript produced no usable bounding box for $PDF" >&2; exit 1; }
done
PAD=4
W=$(awk "BEGIN{print ($URX-$LLX)+2*$PAD}")
H=$(awk "BEGIN{print ($URY-$LLY)+2*$PAD}")
OX=$(awk "BEGIN{print -($LLX-$PAD)}")
OY=$(awk "BEGIN{print -($LLY-$PAD)}")
CROP="$BUILD_DIR/${NAME}_crop.pdf"
gs -q -o "$CROP" -sDEVICE=pdfwrite \
   -dDEVICEWIDTHPOINTS="$W" -dDEVICEHEIGHTPOINTS="$H" -dFIXEDMEDIA \
   -c "<</PageOffset [$OX $OY]>> setpagedevice" -f "$PDF"

# --- rasterise ---------------------------------------------------------------
OUT_PNG="$OUT_DIR/preview_${NAME}.png"
pdftocairo -png -r "$DPI" -singlefile "$CROP" "${OUT_PNG%.png}"
echo "rendered     : $OUT_PNG"
rm -rf "$BUILD_DIR"        # render done; build no longer needed (trap then no-ops)

# --- freeze as golden if requested -------------------------------------------
GOLDEN="$REF_DIR/${NAME}.png"
if [ "$SET_GOLDEN" -eq 1 ]; then
  cp "$OUT_PNG" "$GOLDEN"
  echo "golden set   : $GOLDEN"
fi

# --- compare against the reference -------------------------------------------
# Skip on a --set-golden run with no explicit --reference: comparing to the copy
# we just froze is tautological (always ~100%).
RC=0
if [ "$COMPARE" -eq 1 ] && { [ "$SET_GOLDEN" -ne 1 ] || [ -n "$REFERENCE" ]; }; then
  REF="${REFERENCE:-$GOLDEN}"
  # golden = a same-recipe prior render -> content-aware (per-tile) check is valid.
  # external --reference (e.g. a screenshot) -> global similarity only (per-tile
  # would be dominated by DPI/anti-alias jitter and give false FAILs).
  if [ -n "$REFERENCE" ]; then CMP_MODE=external; else CMP_MODE=golden; fi
  # (D3) the content check only catches changes that actually reach the pixels;
  # warn proactively if the figure source or preamble files are newer than the
  # golden, so a legitimately-changed figure isn't silently checked against a
  # stale reference.
  if [ "$CMP_MODE" = golden ] && [ -f "$GOLDEN" ]; then
    for s in "$SRC" "$MAIN_TEX" "$MACROS_TEX"; do
      if [ -f "$s" ] && [ "$s" -nt "$GOLDEN" ]; then
        echo "WARNING      : $(basename "$s") is newer than the golden; re-freeze with --set-golden if the figure changed." >&2
      fi
    done
  fi
  if [ -f "$REF" ]; then
    MONTAGE="$OUT_DIR/${NAME}_vs_reference.png"
    set +e
    "$PY" - "$REF" "$OUT_PNG" "$MONTAGE" "$THRESHOLD" "$CMP_MODE" <<'PY'
import sys
from PIL import Image, ImageDraw
import numpy as np

ref_p, new_p, montage_p = sys.argv[1], sys.argv[2], sys.argv[3]
thr  = float(sys.argv[4])
mode = sys.argv[5]            # "golden" (content-aware) | "external" (global only)
CONTENT_DELTA = 28.0          # max worst-tile mean-colour shift (0-255) in golden mode
TILE = 24                     # tile size (px) on the width-normalised canvas

def load(p): return Image.open(p).convert("RGB")
ref, new = load(ref_p), load(new_p)

# aspect sanity (large mismatch => comparison is apples-to-oranges)
ar_ref, ar_new = ref.size[0]/ref.size[1], new.size[0]/new.size[1]
aspect_warn = abs(ar_ref - ar_new) / ar_ref > 0.02

# normalise to a common width, pad the shorter to common height with white
T = 1000
def fit(im):
    w, h = im.size; nh = max(1, round(T * h / w)); return im.resize((T, nh), Image.LANCZOS)
rf, nf = fit(ref), fit(new)
CH = max(rf.size[1], nf.size[1])
def pad(im):
    c = Image.new("RGB", (T, CH), (255, 255, 255)); c.paste(im, (0, 0)); return c
A = np.asarray(pad(rf)).astype(np.float64)
B = np.asarray(pad(nf)).astype(np.float64)

# (1) global similarity — catches layout / spacing / scale drift
gsim = 100.0 * (1.0 - np.abs(A - B).mean() / 255.0)

# (2) per-tile content delta — catches a single status cell changing colour
#     (a global mean is blind to it because the figure is ~80% white/grey).
worst, changed, ntiles = 0.0, 0, 0
if mode == "golden":
    for y in range(0, CH, TILE):
        for x in range(0, T, TILE):
            d = float(np.abs(A[y:y+TILE, x:x+TILE].reshape(-1,3).mean(0)
                             - B[y:y+TILE, x:x+TILE].reshape(-1,3).mean(0)).max())
            ntiles += 1
            worst = max(worst, d)
            if d > CONTENT_DELTA: changed += 1

# montage: reference | new | amplified diff (dark = changed)
diff = np.clip(np.abs(A - B) * 4.0, 0, 255).astype(np.uint8)
dimg = Image.fromarray(255 - diff)
gap, head = 16, 30
m = Image.new("RGB", (T*3 + gap*2, CH + head), (255, 255, 255))
for i, p in enumerate([pad(rf), pad(nf), dimg]): m.paste(p, (i*(T+gap), head))
dr = ImageDraw.Draw(m)
dr.text((4, 8), f"REFERENCE: {ref_p.split('/')[-1]}", fill=(0,0,0))
dr.text((T+gap+4, 8), f"NEW: {new_p.split('/')[-1]}", fill=(0,0,0))
dr.text((2*(T+gap)+4, 8), "DIFF (4x; dark = changed)", fill=(0,0,0))
m.save(montage_p)

ok = gsim >= thr
print(f"global sim   : {gsim:.2f}%  (threshold {thr:.1f}%)")
if mode == "golden":
    ok = ok and worst <= CONTENT_DELTA
    print(f"content      : worst-tile {worst:.1f}/255 (limit {CONTENT_DELTA:.0f}); {changed}/{ntiles} tiles differ")
else:
    print("content      : SKIPPED (external reference; global similarity only)")
if aspect_warn:
    print(f"WARNING      : aspect differs (ref {ar_ref:.3f} vs new {ar_new:.3f}); comparison may be unreliable")
print(f"montage      : {montage_p}")
print(f"verdict      : {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 2)
PY
    RC=$?
    set -e
  else
    echo "compare      : SKIPPED (no reference at $REF)"
    echo "               run once with --set-golden to freeze the approved look."
  fi
elif [ "$COMPARE" -eq 1 ] && [ "$SET_GOLDEN" -eq 1 ]; then
  echo "compare      : SKIPPED (golden just frozen this run)"
fi

if [ "$RC" -ne 0 ]; then
  echo "RESULT       : preview differs from reference beyond threshold — inspect the montage." >&2
  echo "               if the change is intended, re-freeze: ./render_figure.sh $NAME --set-golden" >&2
  exit "$RC"
fi
echo "RESULT       : OK"
