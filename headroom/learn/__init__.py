"""Headroom Learn — offline session learning for coding agents.

Analyzes conversation logs using Sonnet 4.6 to extract actionable patterns
and generates context (CLAUDE.md, MEMORY.md, AGENTS.md, etc.) that prevents
future token waste.

Architecture:
    Scanner (adapter)  →  Analyzer (LLM)  →  Writer (adapter)
    ├── ClaudeCodeScanner   SessionAnalyzer     ├── ClaudeCodeWriter
    └── CodexScanner        (Sonnet 4.6)        ├── CodexWriter
                                                └── GeminiWriter
"""
