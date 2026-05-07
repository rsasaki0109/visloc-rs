#!/usr/bin/env sh
set -eu

matches=$(mktemp)
trap 'rm -f "$matches"' EXIT
rg -n -o '\[[^]]+\]\([^)]+\)' README.md docs -g '*.md' > "$matches"

status=0
while IFS= read -r entry; do
    source_file=${entry%%:*}
    rest=${entry#*:}
    line_number=${rest%%:*}
    markdown=${rest#*:}
    target=$(printf '%s\n' "$markdown" | sed 's/.*](\([^)]*\)).*/\1/')

    case "$target" in
        http://*|https://*|mailto:*|"")
            continue
            ;;
    esac

    target_file=${target%%#*}
    if [ -z "$target_file" ]; then
        continue
    fi

    case "$target_file" in
        /*)
            path=".$target_file"
            ;;
        *)
            source_dir=$(dirname "$source_file")
            path="$source_dir/$target_file"
            ;;
    esac

    if [ ! -e "$path" ]; then
        printf 'broken markdown link: %s:%s -> %s\n' "$source_file" "$line_number" "$target" >&2
        status=1
    fi
done < "$matches"

exit "$status"
