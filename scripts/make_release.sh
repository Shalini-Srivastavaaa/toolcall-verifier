#!/usr/bin/env bash
# Export a clean copy of this repository for publication, with a NEW git history (one initial commit) (D-032).
#
#   scripts/make_release.sh [DEST] [--force]        # default DEST: ../toolcall-verifier-release
#
# What it does
#   1. Takes the files of the last COMMIT (git archive HEAD): uncommitted changes and untracked files are never copied. It warns if the working tree is dirty.
#   2. Deletes what must not be published (the DENY list below): raw benchmark data, model files, training data and everything derived from it (data/derived,
#      the Kaggle upload folders), induced run dumps, review-decision files, raw judge outputs, caches, secrets, and ANY file over 5 MB.
#   3. Creates a new repository in DEST with a single commit. The author is generic unless you set RELEASE_AUTHOR_NAME and RELEASE_AUTHOR_EMAIL.
#   4. Runs gitleaks over the release copy (files and the new history) if it is installed (GITLEAKS=/path/to/gitleaks, or on PATH); findings are printed with the values redacted.
#   5. Prints what was removed, and what is left.
# Nothing is pushed or uploaded.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$SRC/../toolcall-verifier-release}"
FORCE=0
for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done
[ "$DEST" = "--force" ] && DEST="$SRC/../toolcall-verifier-release"
MAX_BYTES=$((5 * 1024 * 1024))

# Paths removed from the export (relative to the repository root; a trailing / means a directory).
DENY=(
  "data/raw/"                      # downloaded benchmarks and the induced run dumps (data/raw/induced/induced_runs.jsonl)
  "data/derived/"                  # training pools, negatives, controls, CriticTool conversion, review queues and decisions, localization caches, induced plans and candidates, final splits
  "data/kaggle_upload/" "data/kaggle_upload_v3a/" "data/kaggle_upload_v3b/"   # packaged training files
  "data/labels_manual.jsonl"       # the first hand labels with reviewer notes (the labels themselves are in release/test_set)
  "models/" "checkpoints/"         # weights and run folders (the weights go to Hugging Face, see release/hf_model)
  ".env" ".env.local" ".release-scrub"   # secrets and the local scrub patterns
  "results/judge_new_test/openai-mini.jsonl" "results/judge_new_test/openai-nano.jsonl"   # raw judge outputs (the *_predictions.jsonl files stay)
  "results/phase0_raw/" "results/speed_cache/"
  "docs/kaggle_v2_run.ipynb"       # an executed notebook with printed training rows
  "release/hf_model/model.safetensors"
)

[ -d "$SRC/.git" ] || { echo "not a git repository: $SRC" >&2; exit 1; }
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: the working tree has uncommitted changes to tracked files; they are NOT in the release (only HEAD is exported)." >&2
fi
if [ -e "$DEST" ]; then
  if [ "$FORCE" = 1 ]; then rm -rf "$DEST"; else echo "$DEST exists; use --force to replace it" >&2; exit 1; fi
fi
mkdir -p "$DEST"
git -C "$SRC" archive HEAD | tar -x -C "$DEST"
BEFORE=$(find "$DEST" -type f | wc -l | tr -d ' ')

REMOVED="$(mktemp)"
for p in "${DENY[@]}"; do
  if [ -e "$DEST/$p" ]; then
    find "$DEST/$p" -type f | sed "s#^$DEST/##" >> "$REMOVED"
    rm -rf "$DEST/$p"
  fi
done
# anything over the size limit
find "$DEST" -type f -size +"${MAX_BYTES}c" | while read -r f; do echo "${f#$DEST/}  (over 5 MB)" >> "$REMOVED"; rm -f "$f"; done
# empty directories left behind
find "$DEST" -type d -empty -delete 2>/dev/null || true

# Personal identifiers (user names, home paths) are replaced in the COPY only. The patterns live in the untracked file .release-scrub of the source repository, one per line as
# "pattern=>replacement" (never committed, never exported: it would publish what it hides).
if [ -f "$SRC/.release-scrub" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ""|\#*) continue;; esac
    PAT="${line%%=>*}" REP="${line#*=>}"
    grep -rIl --exclude-dir=.git -F -- "$PAT" "$DEST" 2>/dev/null | while read -r f; do PAT="$PAT" REP="$REP" perl -pi -e 's/\Q$ENV{PAT}\E/$ENV{REP}/g' "$f"; done
  done < "$SRC/.release-scrub"
  echo "scrubbed identifiers from the copy using .release-scrub ($(grep -cv '^\(#\|$\)' "$SRC/.release-scrub") patterns)"
else
  echo "no .release-scrub file: personal identifiers (user names, home paths) are NOT removed from the copy" >&2
fi

cd "$DEST"
git init -q -b main
git add -A
GIT_AUTHOR_NAME="${RELEASE_AUTHOR_NAME:-toolcall-verifier}" GIT_AUTHOR_EMAIL="${RELEASE_AUTHOR_EMAIL:-noreply@example.invalid}" \
GIT_COMMITTER_NAME="${RELEASE_AUTHOR_NAME:-toolcall-verifier}" GIT_COMMITTER_EMAIL="${RELEASE_AUTHOR_EMAIL:-noreply@example.invalid}" \
  git commit -q -m "Initial release"
echo "release copy: $DEST"
echo "files exported from HEAD: $BEFORE; removed: $(wc -l < "$REMOVED" | tr -d ' '); kept: $(git ls-files | wc -l | tr -d ' ')  (list of removed files: $REMOVED)"
du -sh "$DEST/.git" "$DEST" | sed 's/^/size: /'

GL="${GITLEAKS:-$(command -v gitleaks || true)}"
if [ -n "$GL" ] && [ -x "$GL" ]; then
  RD="$(mktemp -d)"
  echo "--- gitleaks (files, then the new history; values redacted; .gitleaks.toml allowlists the BFCL placeholder tokens of release/test_set/test_synth.jsonl) ---"
  "$GL" dir . --redact=100 --no-banner --log-level warn --report-format json --report-path "$RD/dir.json" || true
  "$GL" git . --redact=100 --no-banner --log-level warn --report-format json --report-path "$RD/git.json" || true
  echo "reports (redacted JSON, outside the release copy): $RD"
else
  echo "gitleaks not found: install it and re-run, or run it yourself over $DEST before publishing" >&2
fi
