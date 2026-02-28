"""Headroom Learn — offline failure learning for coding agents.

Analyzes conversation logs to find tool call failure patterns and generates
context (CLAUDE.md, MEMORY.md, .cursorrules, etc.) that prevents future failures.

Architecture:
    Scanner (adapter)  →  Analyzer (generic)  →  Writer (adapter)
    ├── ClaudeCodeScanner   ├── EnvironmentAnalyzer   ├── ClaudeCodeWriter
    ├── CursorScanner       ├── StructureAnalyzer     ├── CursorWriter
    └── GenericScanner      ├── RetryAnalyzer         └── GenericWriter
                            ├── PermissionAnalyzer
                            └── CrossSessionAnalyzer

Scanners read tool-specific log formats and produce normalized ToolCall sequences.
Analyzers work on ToolCall — same analysis for any agent system.
Writers output to tool-specific context injection mechanisms.
"""
