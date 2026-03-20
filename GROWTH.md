# GROWTH.md — Headroom Growth Playbook

A community-contributed guide to help Headroom reach the developers who need it.

---

## 📍 Positioning

**One-liner:** The context compression layer that makes AI agents 70-95% cheaper without code changes.

**The problem you solve:**
- AI agents inject huge amounts of boilerplate into every prompt (tool outputs, logs, RAG chunks, file reads)
- Developers are paying for tokens they don't need
- Context windows fill up before tasks complete

**Why Headroom wins:**
| vs RTK | vs Cloud APIs (Compresr, Token Company) |
|--------|------------------------------------------|
| Compresses *all* context, not just CLI | Data stays local — privacy by default |
| Framework integrations built-in | Reversible compression (CCR) |
| Works alongside RTK, not instead | Open source, no vendor lock-in |

---

## 👁️ Visibility Checklist

### Awesome Lists (submit PRs here)

| List | Status | Link to Submit |
|------|--------|----------------|
| [awesome-llm](https://github.com/Hannibal046/Awesome-LLM) | ⬜ Submit | Add under "LLM Tools" |
| [awesome-langchain](https://github.com/kyrolabs/awesome-langchain) | ⬜ Submit | Add under "Tools & Utilities" |
| [awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) | ⬜ Submit | Add under "Developer Tools" |
| [awesome-chatgpt](https://github.com/Kamigami55/awesome-chatgpt) | ⬜ Submit | Add under "Development Tools" |
| [awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents) | ⬜ Submit | Add under "Infrastructure & Tools" |
| [awesome-mcp](https://github.com/punkpeye/awesome-mcp-servers) | ⬜ Submit | MCP integration |
| [awesome-claude](https://github.com/anthropics/anthropic-cookbook) | ⬜ Submit | Add example to cookbook |

**Submission template:**
```markdown
- [Headroom](https://github.com/chopratejas/headroom) - Context compression layer for LLM applications. Reduces tokens 70-95% without code changes. Supports proxy mode, LangChain, LiteLLM, MCP.
```

### Directories & Aggregators

| Platform | Action |
|----------|--------|
| [LibHunt](https://www.libhunt.com/) | Submit for indexing |
| [OSS Insight](https://ossinsight.io/) | Auto-indexed, boost with GitHub activity |
| [Futurepedia](https://www.futurepedia.io/) | Submit as AI tool |
| [There's An AI For That](https://theresanaiforthat.com/) | Submit under "Developer Tools" |
| [Product Hunt](https://www.producthunt.com/) | Consider launch when hitting 1k stars |

---

## 🎯 Community Distribution

### Where Headroom users hang out

| Channel | Content Type | Why |
|---------|--------------|-----|
| **r/LocalLLaMA** | "I reduced my Claude API bill by 80% with this proxy" | Cost-conscious local AI devs |
| **r/ChatGPTCoding** | "How I made Cursor 3x cheaper" | Cursor/Claude Code users |
| **r/ClaudeAI** | MCP integration guide + real savings numbers | Claude power users |
| **HackerNews** | "Show HN: I compressed AI agent context by 90%" | Technical audience, loves benchmarks |
| **LangChain Discord** | Integration announcement in #announcements | LangChain users |
| **LiteLLM Discord** | Callback integration demo | LiteLLM users |

### Content ideas (high-leverage)

1. **"The hidden cost of AI agents: 90% of your tokens are boilerplate"**
   - Blog post with real numbers
   - Visual breakdown of what gets compressed

2. **"How I cut my Claude Code bill by $200/month"**
   - First-person case study
   - Step-by-step proxy setup

3. **Video demo: "Zero-code context compression for any LLM app"**
   - 3-5 min walkthrough
   - Before/after token counts

4. **Benchmark deep-dive post**
   - Your existing benchmarks are great — package for HN
   - "We tested compression on GSM8K, TruthfulQA, SQuAD — here's what we learned"

---

## 📅 Sustainable Rhythm

### Weekly
- [ ] Respond to GitHub issues/discussions within 24h
- [ ] Share 1-2 user wins on Twitter/X (with permission)
- [ ] Monitor r/LocalLLaMA and r/ClaudeAI for context window complaints → helpful comments

### Monthly
- [ ] Publish 1 tutorial or benchmark post
- [ ] Submit to 1-2 awesome-lists
- [ ] Check competitor repos for new users who might benefit from Headroom

### Quarterly
- [ ] Major version announcement on HN/Reddit
- [ ] Review and update benchmark numbers
- [ ] Reach out to AI tool roundup authors for inclusion

---

## 🤝 Contributor Community

### Easy first issues
Tag issues with `good-first-issue`:
- New compressor for specific content types (YAML, TOML, CSV)
- Documentation improvements
- Integration examples (new frameworks)
- Test coverage expansion

### Recognition
- Shout out contributors in release notes
- Consider a CONTRIBUTORS.md hall of fame
- Feature community integrations in README

---

## 📊 Success Metrics

| Metric | Current | 6-mo Target | How to Track |
|--------|---------|-------------|--------------|
| GitHub Stars | ~725 | 2,000+ | GitHub |
| PyPI Downloads/month | ? | 5,000+ | pypistats.org |
| Discord members | ? | 500+ | Discord |
| Awesome-list inclusions | 0 | 5+ | Manual check |

---

## 📚 Additional Resources

If you're scaling open source growth, these playbooks might help:
- [Gingiris Open Source Launch Playbook](https://github.com/Gingiris/opensource-launch) — GitHub star growth tactics
- [Gingiris Product Launch Guide](https://github.com/Gingiris/product-launch) — Product Hunt, KOL, community distribution

---

*This is a community contribution. Feel free to adapt, expand, or remove sections as needed.*

*Last updated: 2026-03-13*
