#!/usr/bin/env sh
set -eu

markdown_anchor_exists() {
    file=$1
    anchor=$2

    headings=$(mktemp)
    (grep -E '^#+[[:space:]]+' "$file" || true) | while IFS= read -r heading; do
        printf '%s\n' "$heading" \
            | sed -E 's/^#+[[:space:]]*//; s/<[^>]*>//g; s/`//g; s/[^[:alnum:] _-]//g; s/[[:space:]]+/-/g; s/-+/-/g; s/^-//; s/-$//' \
            | tr '[:upper:]' '[:lower:]'
    done > "$headings"

    if grep -Fxq "$anchor" "$headings"; then
        rm -f "$headings"
        return 0
    fi

    rm -f "$headings"
    return 1
}

matches=$(mktemp)
trap 'rm -f "$matches"' EXIT
{
    printf '%s\n' README.md
    find docs -type f -name '*.md' | sort
} | while IFS= read -r file; do
    [ -f "$file" ] || continue
    (grep -n -o '\[[^][]*\]([^)]*)' "$file" || true) | sed "s|^|$file:|"
done > "$matches"

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
    anchor=""
    case "$target" in
        *#*)
            anchor=${target#*#}
            ;;
    esac

    if [ -z "$target_file" ]; then
        path="$source_file"
    else
        case "$target_file" in
            /*)
                path=".$target_file"
                ;;
            *)
                source_dir=$(dirname "$source_file")
                path="$source_dir/$target_file"
                ;;
        esac
    fi

    if [ ! -e "$path" ]; then
        printf 'broken markdown link: %s:%s -> %s\n' "$source_file" "$line_number" "$target" >&2
        status=1
        continue
    fi

    if [ -n "$anchor" ] && ! markdown_anchor_exists "$path" "$anchor"; then
        printf 'broken markdown anchor: %s:%s -> %s\n' "$source_file" "$line_number" "$target" >&2
        status=1
    fi
done < "$matches"

exit "$status"
