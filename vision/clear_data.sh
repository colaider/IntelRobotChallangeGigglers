#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ -L dataset ]]; then
    echo "Cannot clear a symlinked dataset directory." >&2
    exit 1
fi

folders=()
for folder in dataset/raw dataset/processed; do
    if [[ -d "$folder" && ! -L "$folder" ]]; then
        folders+=("$folder")
    fi
done

if [[ ${#folders[@]} -eq 0 ]]; then
    echo "No data folders to clear."
    exit 0
fi

printf 'Move these folders to Trash?\n'
printf '  %s\n' "${folders[@]}"
read -r -p "Type yes to continue: " answer
[[ "$answer" == "yes" ]] || exit 0

trash_dir=$(mktemp -d "$HOME/.Trash/brick-data-XXXXXX")
for folder in "${folders[@]}"; do
    mv -- "$folder" "$trash_dir/"
done
printf 'Moved to %s\n' "$trash_dir"
