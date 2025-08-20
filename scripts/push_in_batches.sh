#!/usr/bin/env bash
set -euo pipefail
cd /home/kronos/mushroooom
MAX=$((100*1024*1024))
# 1) Untrack and ignore files >90MB
mapfile -d "" -t big < <(find . -type f -not -path "./.git/*" -size +90M -print0)
if ((${#big[@]})); then
  for f in "${big[@]}"; do
    f="${f#./}"
    git rm --cached --ignore-unmatch "$f" || true
    grep -qxF "$f" .gitignore 2>/dev/null || echo "$f" >> .gitignore
  done
  git add .gitignore || true
  git diff --cached --quiet || git commit -m "chore: untrack >90MB assets and update .gitignore"
  git push --progress origin HEAD:main | cat || true
fi
# 2) Collect pending files (modified + untracked), exclude deletions; recurse into dirs
declare -a list=()
add_path() {
  local p="$1"
  # skip excluded roots
  case "$p" in
    data/*|data|./data|.venv/*|.venv|./.venv|__pycache__/*|__pycache__|./__pycache__|cache/*|cache|./cache)
      return;;
  esac
  if [[ -d "$p" ]]; then
    while IFS= read -r -d "" f; do
      add_path "${f#./}"
    done < <(find "$p" -type f -not -path "./.git/*" -print0)
  elif [[ -f "$p" ]]; then
    list+=("$p")
  fi
}
while IFS= read -r -d "" entry; do
  status="${entry:0:2}"
  path="${entry:3}"
  if [[ "$status" == " D" || "$status" == "D " || "$status" == "DD" ]]; then
    continue
  fi
  add_path "$path"
done < <(git status --porcelain -z)
# 3) Batch and push
batch_bytes=0
declare -a batch=()
commit_and_push() {
  if ((${#batch[@]}==0)); then return; fi
  git add -f -- "${batch[@]}" || true
  if git diff --cached --quiet; then
    batch=()
    batch_bytes=0
    return
  fi
  msg="batch($(date -Iseconds)): ${#batch[@]} files, <=100MB"
  git commit -m "$msg"
  git push --progress origin HEAD:main | cat
  batch=()
  batch_bytes=0
}
for f in "${list[@]}"; do
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if (( size > MAX )); then
    git rm --cached --ignore-unmatch "$f" || true
    grep -qxF "$f" .gitignore 2>/dev/null || echo "$f" >> .gitignore
    continue
  fi
  if (( batch_bytes + size > MAX )); then
    commit_and_push
  fi
  batch+=("$f")
  batch_bytes=$((batch_bytes + size))
done
commit_and_push
# 4) Push any lingering .gitignore update
git add .gitignore 2>/dev/null || true
git diff --cached --quiet || { git commit -m "update .gitignore for big files"; git push --progress origin HEAD:main | cat; }
