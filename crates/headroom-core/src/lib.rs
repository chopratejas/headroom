//! headroom-core: foundation crate for the Rust port of Headroom.

pub mod ccr;
pub mod relevance;
pub mod tokenizer;
pub mod transforms;

/// Identity stub used by downstream crates and the Python binding to verify
/// linkage end-to-end.
pub fn hello() -> &'static str {
    "headroom-core"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_crate_name() {
        assert_eq!(hello(), "headroom-core");
    }
}
