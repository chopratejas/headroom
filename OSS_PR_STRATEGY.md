# Headroom OSS PR Strategy

## Goal
Contribute to popular LangChain ecosystem repos to demonstrate Headroom's value and drive adoption.

## Target Repos (Ranked by Priority)

### Priority 1: `langchain-ai/how_to_fix_your_context`
- **What**: LangChain's official repo of context management techniques
- **PR**: Add `06-context-compression.ipynb` notebook showing Headroom as technique #6
- **Why accept**: They're curating techniques, not competing. Compression is genuinely different from pruning/summarization.
- **Status**: IN PROGRESS

### Priority 2: `langchain-ai/langgraph` docs/cookbook
- **What**: Core LangGraph framework
- **PR**: Add `compress_tool_messages` pre-model hook example
- **Issues it addresses**: #3717 (ToolMessage overflow), #11405 (agent token limit), #2140 (127K tokens from plugin)
- **Status**: DONE — `compress_tool_messages()` and `create_compress_tool_messages_node()` in `headroom/integrations/langchain/langgraph.py`

### Priority 3: `langchain-ai/deepagents` (~17K stars)
- **What**: LangChain's coding agent (like Claude Code but OSS)
- **PR**: Integrate Headroom as compression backend (they already claim "automatic compression")
- **Status**: TODO

### Priority 4: `langchain-ai/open-swe`
- **What**: Async coding agent that resolves GitHub issues
- **PR**: Add optional Headroom compression for long-running tasks
- **Status**: TODO

### Priority 5: `assafelovic/gpt-researcher`
- **What**: Autonomous research agent (explicitly cites token limits as motivation)
- **PR**: Add Headroom to compress scraped web content before synthesis
- **Status**: TODO

### Priority 6: `langchain-core` — `compress_messages` utility
- **What**: Core LangChain library
- **PR**: Add `compress_messages()` alongside `trim_messages()`
- **Status**: TODO (hardest to land, highest impact)

## Key Links
- [how_to_fix_your_context](https://github.com/langchain-ai/how_to_fix_your_context)
- [LangGraph Issue #3717](https://github.com/langchain-ai/langgraph/issues/3717)
- [LangChain Issue #11405](https://github.com/langchain-ai/langchain/issues/11405)
- [deepagents](https://github.com/langchain-ai/deepagents)
- [open-swe](https://github.com/langchain-ai/open-swe)
- [gpt-researcher](https://github.com/assafelovic/gpt-researcher)
