#!/usr/bin/env bash
#
# Fetch the third-party baseline agents this repository does NOT redistribute.
#
# SakkirinaSolo (2025 competition winner) and BestMCTS3 are other people's
# competition entries. They are not ours to republish, so they are not in this
# repository. This script clones the official competition archive at a pinned
# commit and copies them into Bots/src/ on your machine, where the build will
# pick them up. They remain the work of their respective authors and are not
# covered by this repository's licence -- see README.md, "Third-party agents".
#
# It then derives two of our own agents from the fetched SakkirinaSolo.cs by
# applying the patches in this directory (see their headers for exactly what
# each one changes and why they ship as patches rather than source files).
#
# Running this is required to reproduce any baseline comparison. It is NOT
# required to build or run DeepSetsBot / DeepSetsBlendBot themselves.
#
# Idempotent: safe to re-run. Use --force to overwrite existing files.

set -euo pipefail

ARCHIVE_URL="https://github.com/ScriptsOfTribute/ScriptsOfTribute-CompetitionsArchive.git"
# Pinned so this script keeps producing the same baselines as the paper used.
ARCHIVE_COMMIT="105aaa72b2cdc4544873f75cd0356d47929dadea"
ARCHIVE_SUBDIR="competition-2025-08-COG/cs_agents"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/Bots/src"
SCRIPTS="$REPO_ROOT/scripts"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# BestMCTS3 needs six support files; the closure is BestMCTS3 -> GameStrategy,
# InstantPlayCards, and GameStrategy -> GPCardTierList (TierList.cs), AgentTier
# (AgentTierList.cs), HandTierList, PatronTierList. Nothing else.
FILES=(
  "SakkirinaSolo/SakkirinaSolo.cs"
  "BestMCTS3/BestMCTS3.cs"
  "BestMCTS3/GameStrategy.cs"
  "BestMCTS3/InstantPlayCards.cs"
  "BestMCTS3/TierList.cs"
  "BestMCTS3/AgentTierList.cs"
  "BestMCTS3/HandTierList.cs"
  "BestMCTS3/PatronTierList.cs"
)

# SHA-256 of each file as it stands at the pinned commit, recorded when this
# repository was assembled. A mismatch means the archive moved under the pin or
# the checkout is corrupt -- either way, stop, rather than silently benchmark
# against something other than what the paper measured.
EXPECTED_SHA256="\
ef3f6e08a5d54e86092fe79fd8e46abf6701ab8a6f5f3bf3c2e7b4ffeba9ebf7  SakkirinaSolo.cs
513053e5fd002cba9130f65c0aa63ef2c706e8341863c72b80c48134d93cb3d6  BestMCTS3.cs
2bd5a613e7520286c96078e59ef4bbbf7cb90331ef6bf29c97b610e41c0af679  GameStrategy.cs
60935cbdd69cd1a5eff90a46fe79883a187a81ba8da63a1446ad1583aad1f54c  InstantPlayCards.cs
b7f779a01e40ed0f19ee6cf9b511ce9d1cc06937f3fab93971729a1fe42f0cad  TierList.cs
9947208bd0da9efa618e8564301b87b788b2e38663f342ea84a6c4bc27f79831  AgentTierList.cs
a8309eee986cde63ad6dda9a2f9295d60619a0bc0873c49356e90074cfb991b1  HandTierList.cs
b09acea6203a13c390f60114dc590fc7c1fd6196e948aba83c59e3a68d0732b3  PatronTierList.cs"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' is required but not installed." >&2; exit 1; }; }
need git
need patch

if command -v sha256sum >/dev/null 2>&1; then SHA() { sha256sum "$@"; }
elif command -v shasum >/dev/null 2>&1;    then SHA() { shasum -a 256 "$@"; }
else echo "ERROR: need sha256sum or shasum." >&2; exit 1; fi

TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "Cloning competition archive at pinned commit ${ARCHIVE_COMMIT:0:12}..."
git clone --quiet --filter=blob:none --no-checkout "$ARCHIVE_URL" "$TMP/archive"
git -C "$TMP/archive" checkout --quiet "$ARCHIVE_COMMIT"

SRC="$TMP/archive/$ARCHIVE_SUBDIR"
[[ -d "$SRC" ]] || { echo "ERROR: $ARCHIVE_SUBDIR not found at the pinned commit." >&2; exit 1; }

echo
echo "Verifying archive contents against recorded checksums ..."
(
  cd "$SRC"
  for rel in "${FILES[@]}"; do
    base="$(basename "$rel")"
    want="$(printf '%s\n' "$EXPECTED_SHA256" | awk -v f="$base" '$2==f {print $1}')"
    got="$(SHA "$rel" | cut -d' ' -f1)"
    if [[ "$want" != "$got" ]]; then
      echo "ERROR: checksum mismatch for $base" >&2
      echo "  expected $want" >&2
      echo "  got      $got" >&2
      echo "The archive has changed under the pinned commit. Stopping." >&2
      exit 1
    fi
  done
)
echo "  all ${#FILES[@]} files match."

echo
echo "Copying baseline agents into Bots/src/ ..."
for rel in "${FILES[@]}"; do
  base="$(basename "$rel")"
  if [[ -e "$DEST/$base" && $FORCE -eq 0 ]]; then
    echo "  skip    $base (already present; --force to overwrite)"
    continue
  fi
  cp "$SRC/$rel" "$DEST/$base"
  echo "  copied  $base"
done

echo
echo "Deriving our agents from the fetched SakkirinaSolo.cs ..."
apply_patch() {
  local out="$1" patchfile="$2"
  if [[ -e "$DEST/$out" && $FORCE -eq 0 ]]; then
    echo "  skip    $out (already present; --force to overwrite)"
    return
  fi
  patch --quiet -o "$DEST/$out" "$DEST/SakkirinaSolo.cs" < "$SCRIPTS/$patchfile"
  echo "  derived $out"
}
apply_patch SakkirinaGen.cs    sakkirina_gen.patch
apply_patch SakkirinaHalf.cs   sakkirina_half.patch
apply_patch SakkirinaScaled.cs sakkirina_scaled.patch

cat <<'EOF'

Done. Bots/src/ now contains the baseline agents and the two derived agents.

  Baselines (not ours -- see README.md, "Third-party agents"):
    SakkirinaSolo.cs, BestMCTS3.cs and BestMCTS3's six support files

  Derived from SakkirinaSolo.cs by the patches in scripts/:
    SakkirinaGen.cs     self-play data generation
    SakkirinaHalf.cs    search-volume control condition
    SakkirinaScaled.cs  time-budget-scalable baseline (SOT_BASELINE_TIME_SCALE),
                        with an evaluations-per-turn counter

Next:
    dotnet build TalesOfTribute.sln -c Release

These files are deliberately listed in .gitignore. Do not commit them.
EOF
