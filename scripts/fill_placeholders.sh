#!/usr/bin/env bash
# Fill in the publishing placeholders (D-032). Run it in the working repository, review `git diff`, commit, THEN run scripts/make_release.sh.
#
#   scripts/fill_placeholders.sh --holder "Your Name or Organization" --hf-user YOUR_HF_NAME --gh-owner YOUR_GITHUB_NAME [--repo toolcall-verifier]
#
# Replaces: the copyright holder in NOTICE (and its copies in release/hf_model/ and release/space/); HF_USERNAME in release/hf_model/README.md and release/space/{app.py,README.md};
# https://github.com/OWNER/REPO in release/hf_model/README.md. It does NOT touch the TODO fields for authors, contact and citation in docs/model_card.md, release/test_set/README.md and
# DATA_LICENSE.md: edit those by hand (they need your words), then run `python3 scripts/prepare_hf_model.py --repo-url https://github.com/OWNER/REPO` to regenerate the Hub card from docs/model_card.md.
set -euo pipefail
HOLDER=""; HFUSER=""; OWNER=""; REPO="toolcall-verifier"
while [ $# -gt 0 ]; do
  case "$1" in
    --holder) HOLDER="$2"; shift 2;; --hf-user) HFUSER="$2"; shift 2;; --gh-owner) OWNER="$2"; shift 2;; --repo) REPO="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
[ -n "$HOLDER" ] && [ -n "$HFUSER" ] && [ -n "$OWNER" ] || { echo "usage: $0 --holder NAME --hf-user HF_NAME --gh-owner GITHUB_NAME [--repo NAME]" >&2; exit 1; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rep() { # file, literal pattern, replacement
  [ -f "$1" ] || return 0
  grep -qF -- "$2" "$1" || return 0
  PAT="$2" REP="$3" perl -pi -e 's/\Q$ENV{PAT}\E/$ENV{REP}/g' "$1"
  echo "  $1: $2 -> $3"
}
for f in NOTICE release/hf_model/NOTICE release/space/NOTICE; do rep "$f" "<COPYRIGHT HOLDER: fill in before publishing>" "$HOLDER"; done
for f in release/hf_model/README.md release/space/app.py release/space/README.md; do rep "$f" "HF_USERNAME" "$HFUSER"; done
rep release/hf_model/README.md "https://github.com/OWNER/REPO" "https://github.com/$OWNER/$REPO"
echo "done. Review with: git diff --stat; remaining placeholders:"
git grep -n -E 'OWNER/REPO|HF_USERNAME|COPYRIGHT HOLDER: fill' -- . ':!docs/decision_log.md' ':!docs/archive/' ':!docs/doc_audit.md' ':!docs/release_steps.md' ':!release/UPLOAD_COMMANDS.md' ':!scripts/' | cut -c1-120 || true
