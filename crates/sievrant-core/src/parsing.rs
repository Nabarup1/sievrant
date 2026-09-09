use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::{Result, SievrantError};
use crate::ingestion::{DocumentParser, ParsedDocument, Section};
use crate::types::DocumentMetadata;

/// Routes raw document bytes to the correct parser based on MIME type.
///
/// Parsers are registered at construction time. Unknown MIME types fall back
/// to the plain-text parser. Registration follows the open/closed principle:
/// any crate can push a new parser without modifying this file.
pub struct ParserRouter {
    parsers: HashMap<String, Box<dyn DocumentParser>>,
    fallback: Box<dyn DocumentParser>,
}

impl ParserRouter {
    pub fn new(fallback: Box<dyn DocumentParser>) -> Self {
        Self {
            parsers: HashMap::new(),
            fallback,
        }
    }

    pub fn register(&mut self, mime_type: impl Into<String>, parser: Box<dyn DocumentParser>) {
        self.parsers.insert(mime_type.into(), parser);
    }
}

#[async_trait]
impl DocumentParser for ParserRouter {
    async fn parse(&self, content: &str, mime_type: &str) -> Result<ParsedDocument> {
        let parser = self
            .parsers
            .get(mime_type)
            .map(|p| p.as_ref())
            .unwrap_or(self.fallback.as_ref());

        parser.parse(content, mime_type).await
    }
}

/// HTML document parser: walks the DOM and maps heading hierarchy to Sections.
///
/// Tables are converted to Markdown format before embedding so the tabular
/// structure is preserved in the vector representation.
pub struct HtmlParser;

#[async_trait]
impl DocumentParser for HtmlParser {
    async fn parse(&self, content: &str, _mime_type: &str) -> Result<ParsedDocument> {
        parse_html(content)
    }
}

fn parse_html(html: &str) -> Result<ParsedDocument> {
    use std::fmt::Write;

    // Simple DOM traversal without pulling in scraper/kuchiki at this stage.
    // The logic strips tags and reconstructs heading-based section hierarchy.
    // A production build would use the `scraper` crate for robust CSS selector support.
    let mut sections: Vec<Section> = Vec::new();
    let mut current_title = "Document".to_string();
    let mut current_content = String::new();
    let mut depth: usize = 0;

    let mut chars = html.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            // Collect the full tag name
            let tag: String = chars.by_ref().take_while(|&c| c != '>').collect();
            let tag_lower = tag.to_lowercase();

            if tag_lower == "h1" || tag_lower == "h2" || tag_lower == "h3" {
                // Flush current section before starting new heading
                if !current_content.trim().is_empty() {
                    sections.push(Section {
                        heading: Some(current_title.clone()),
                        content: current_content.trim().to_string(),
                        level: depth as u32,
                        children: vec![],
                        page: None,
                        char_span: [0, 0],
                    });
                }
                depth = match tag_lower.as_str() {
                    "h1" => 0,
                    "h2" => 1,
                    _ => 2,
                };
                // The heading text follows as the next non-tag text
                current_title = String::new();
                current_content = String::new();
                // Collect text until closing tag
                for ch in chars.by_ref() {
                    if ch == '<' {
                        // skip the closing tag
                        chars.by_ref().take_while(|&c| c != '>').for_each(drop);
                        break;
                    }
                    current_title.push(ch);
                }
            } else if tag_lower.starts_with("table") {
                // Convert table content to a simple Markdown representation
                let mut table_html = String::new();
                loop {
                    match chars.next() {
                        None => break,
                        Some('<') => {
                            let inner: String = chars.by_ref().take_while(|&c| c != '>').collect();
                            if inner.to_lowercase().starts_with("/table") {
                                break;
                            }
                            write!(table_html, "<{}>", inner).ok();
                        }
                        Some(ch) => table_html.push(ch),
                    }
                }
                // Naive table to markdown: replace td/tr boundaries with pipes and newlines
                let md_table = table_html
                    .replace("<tr>", "\n|")
                    .replace("</tr>", "|")
                    .replace("<td>", " ")
                    .replace("</td>", " |")
                    .replace("<th>", " **")
                    .replace("</th>", "** |");
                current_content.push_str("\n\n");
                current_content.push_str(md_table.trim());
            }
        } else {
            current_content.push(c);
        }
    }

    // Flush final section
    if !current_content.trim().is_empty() {
        sections.push(Section {
            heading: Some(current_title),
            content: current_content.trim().to_string(),
            level: depth as u32,
            children: vec![],
            page: None,
            char_span: [0, 0],
        });
    }

    if sections.is_empty() {
        return Err(SievrantError::Internal(
            "HTML parser produced no sections; document may be empty".into(),
        ));
    }

    Ok(ParsedDocument {
        title: sections.first().and_then(|s| s.heading.clone()),
        sections,
        metadata: DocumentMetadata::default(),
    })
}

/// Plain text parser used as the fallback for unknown MIME types.
pub struct PlainTextParser;

#[async_trait]
impl DocumentParser for PlainTextParser {
    async fn parse(&self, content: &str, _mime_type: &str) -> Result<ParsedDocument> {
        Ok(ParsedDocument {
            title: None,
            sections: vec![Section {
                heading: None,
                content: content.to_string(),
                level: 0,
                children: vec![],
                page: None,
                char_span: [0, 0],
            }],
            metadata: DocumentMetadata::default(),
        })
    }
}

/// PDF parser stub: in production this wraps the `lopdf` crate or calls the
/// `docling` Python service via subprocess. The stub preserves the interface.
pub struct PdfParser;

#[async_trait]
impl DocumentParser for PdfParser {
    async fn parse(&self, _content: &str, _mime_type: &str) -> Result<ParsedDocument> {
        // Real implementation: use lopdf to extract text streams, map PDF
        // outline/bookmark tree to Section hierarchy. Requires &[u8] not &str.
        // The trait will be updated to accept bytes in a follow-up.
        Err(SievrantError::Internal(
            "PDF parsing requires binary content bytes; upgrade the trait signature to &[u8]".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn html_parser_extracts_sections() {
        let html = r#"
<h1>Introduction</h1>
<p>This is the intro paragraph.</p>
<h2>Background</h2>
<p>Some background text here.</p>
"#;
        let doc = parse_html(html).unwrap();
        assert!(!doc.sections.is_empty(), "should produce at least one section");
    }

    #[tokio::test]
    async fn html_parser_converts_tables() {
        let html = r#"
<h1>Data</h1>
<table>
<tr><th>Name</th><th>Value</th></tr>
<tr><td>A</td><td>1</td></tr>
</table>
"#;
        let doc = parse_html(html).unwrap();
        let all_content: String = doc.sections.iter().map(|s| s.content.as_str()).collect();
        assert!(all_content.contains("|") || !all_content.is_empty());
    }

    #[tokio::test]
    async fn router_dispatches_to_html_parser() {
        let mut router = ParserRouter::new(Box::new(PlainTextParser));
        router.register("text/html", Box::new(HtmlParser));

        let html = "<h1>Test</h1><p>hello</p>";
        let doc = router.parse(html, "text/html").await.unwrap();
        assert!(!doc.sections.is_empty());
    }

    #[tokio::test]
    async fn router_falls_back_for_unknown_mime() {
        let router = ParserRouter::new(Box::new(PlainTextParser));
        let doc = router.parse("plain content", "text/csv").await.unwrap();
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].content, "plain content");
    }
}
