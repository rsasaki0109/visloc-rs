pub(super) fn label_slug(label: &str) -> String {
    let mut slug = String::new();
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('_') {
            slug.push('_');
        }
    }
    slug.trim_matches('_').to_string()
}

pub(super) fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub(super) fn csv_cell(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_slug_normalizes_report_labels() {
        assert_eq!(
            label_slug("deep-style (HogLike + MutualSoftmax)"),
            "deep_style_hoglike_mutualsoftmax"
        );
        assert_eq!(label_slug(" classical  "), "classical");
    }

    #[test]
    fn html_escape_escapes_attribute_sensitive_characters() {
        assert_eq!(
            html_escape(r#"A&B <frame> "49""#),
            "A&amp;B &lt;frame&gt; &quot;49&quot;"
        );
    }

    #[test]
    fn csv_cell_quotes_only_when_needed() {
        assert_eq!(csv_cell("deep"), "deep");
        assert_eq!(csv_cell("deep,style"), "\"deep,style\"");
        assert_eq!(csv_cell("quote\"here"), "\"quote\"\"here\"");
    }
}
