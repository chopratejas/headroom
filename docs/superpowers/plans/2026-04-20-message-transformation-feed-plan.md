# Message Transformation Live Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-updating right sidebar drawer to the dashboard that shows a streaming feed of message transformations with side-by-side before/after diffs, virtual scrolling, and auto-scroll that pauses when the user scrolls to review.

**Architecture:** Backend adds a new `/transformations/feed` endpoint returning recent log entries with full request/response messages. Frontend adds a drawer component with virtual scrolling and synchronized side-by-side diff panels.

**Tech Stack:** FastAPI (existing), Alpine.js (existing), vanilla JS virtual scroll, Tailwind (existing)

---

## Security Note

**XSS Prevention:** Backend messages contain user content that will be rendered as HTML. All message content MUST be escaped using `textContent` or a sanitizer before innerHTML insertion. The `escapeHtml()` helper uses `document.createElement('div').textContent` which is safe.

---

## File Structure

### Backend
- Modify: `headroom/proxy/request_logger.py` — add `get_recent_with_messages()` method
- Modify: `headroom/proxy/server.py` — add new `/transformations/feed` endpoint

### Frontend
- Modify: `headroom/dashboard/templates/dashboard.html` — add drawer HTML, Alpine.js drawer component, virtual scroll, diff panel

### Tests
- Create: `tests/test_proxy/test_transformations_feed.py` — backend endpoint tests
- Create: `tests/test_dashboard/test_live_feed.py` — E2E test

---

## Task 1: Backend — `/transformations/feed` Endpoint

**Files:**
- Modify: `headroom/proxy/request_logger.py:68-79` — add `get_recent_with_messages()` method
- Modify: `headroom/proxy/server.py` — add `@app.get("/transformations/feed")` endpoint
- Create: `tests/test_proxy/test_transformations_feed.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_proxy/test_transformations_feed.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from headroom.proxy.server import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_transformations_feed_endpoint_returns_list(app):
    """The endpoint should return a list of recent transformations."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/transformations/feed")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "transformations" in data
    assert isinstance(data["transformations"], list)


@pytest.mark.asyncio
async def test_transformations_feed_returns_messages(app):
    """Each transformation should include request_messages and response_content."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/transformations/feed")
    
    data = response.json()
    transformations = data["transformations"]
    for t in transformations:
        assert "request_messages" in t or t.get("request_messages") is None
        assert "response_content" in t or t.get("response_content") is None


@pytest.mark.asyncio
async def test_transformations_feed_respects_limit(app):
    """The endpoint should respect a ?limit= query parameter."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/transformations/feed?limit=5")
    
    data = response.json()
    assert len(data["transformations"]) <= 5
```

Run: `pytest tests/test_proxy/test_transformations_feed.py -v`
Expected: FAIL — endpoint doesn't exist

- [ ] **Step 2: Add `get_recent_with_messages()` to RequestLogger**

Modify `headroom/proxy/request_logger.py` — add after `get_recent()`:

```python
def get_recent_with_messages(self, n: int = 20) -> list[dict]:
    """Get recent log entries including full request/response messages."""
    entries = list(self._logs)[-n:]
    return [asdict(e) for e in entries]
```

- [ ] **Step 3: Add `/transformations/feed` endpoint to server.py**

After the `/stats-history` endpoint (around line 1624 in server.py), add:

```python
@app.get("/transformations/feed")
async def transformations_feed(limit: int = 20):
    """Get recent message transformations for the live feed."""
    if limit > 100:
        limit = 100  # Cap at 100 for performance

    transformations = []
    if proxy and proxy.logger:
        logs = proxy.logger.get_recent_with_messages(limit)
        for log in logs:
            transformations.append({
                "request_id": log.get("request_id"),
                "timestamp": log.get("timestamp"),
                "provider": log.get("provider"),
                "model": log.get("model"),
                "input_tokens_original": log.get("input_tokens_original"),
                "input_tokens_optimized": log.get("input_tokens_optimized"),
                "tokens_saved": log.get("tokens_saved"),
                "savings_percent": log.get("savings_percent"),
                "transforms_applied": log.get("transforms_applied", []),
                "request_messages": log.get("request_messages"),
                "response_content": log.get("response_content"),
            })

    return {"transformations": transformations}
```

Note: The existing `/stats` and `/stats-history` endpoints use the module-level `proxy` object (set at line 1081) directly — not `request.state.proxy`. Follow the same pattern.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proxy/test_transformations_feed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add headroom/proxy/request_logger.py headroom/proxy/server.py tests/test_proxy/test_transformations_feed.py
git commit -m "feat(dashboard): add /transformations/feed backend endpoint

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Frontend — Sidebar Drawer Shell

**Files:**
- Modify: `headroom/dashboard/templates/dashboard.html` — add drawer HTML, CSS, trigger button

- [ ] **Step 1: Add trigger button to header**

In `dashboard.html`, find the header section (around line 36-78). After the "Updated" timestamp div (around line 74-76), add before the closing `</div>` of the header's right-side div:

```html
<button id="feed-toggle"
        class="px-3 py-1.5 text-sm rounded-md border border-border bg-surface text-gray-300 hover:text-white transition-colors"
        @click="feedOpen = !feedOpen"
        :class="feedOpen ? 'bg-accent text-black' : ''">
    Live Feed
</button>
```

- [ ] **Step 2: Add drawer HTML (before `</main>`)**

Find `</main>` around line 1173. Add before it:

```html
<!-- Live Feed Sidebar Drawer -->
<div x-show="feedOpen"
     x-transition:enter="transition ease-out duration-300"
     x-transition:enter-start="translate-x-full"
     x-transition:enter-end="translate-x-0"
     x-transition:leave="transition ease-in duration-200"
     x-transition:leave-start="translate-x-0"
     x-transition:leave-end="translate-x-full"
     @click.away="feedOpen = false"
     class="fixed top-0 right-0 h-full w-[520px] bg-surface border-l border-border z-50 flex flex-col"
     style="display: none;">

    <!-- Drawer Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-border">
        <div class="flex items-center gap-3">
            <span class="text-sm font-medium text-gray-200">Message Transformations</span>
            <span class="text-xs text-gray-500 font-mono" x-text="transformations.length + ' msgs'"></span>
        </div>
        <button @click="feedOpen = false" class="text-gray-500 hover:text-gray-200 transition-colors p-1">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 16 16" fill="currentColor">
                <path d="M2.146 2.854a.5.5 0 1 1 .708-.708L8 7.293l5.146-5.147a.5.5 0 0 1 .708.708L8.707 8l5.147 5.146a.5.5 0 0 1-.708.708L8 8.707l-5.146 5.147a.5.5 0 0 1-.708-.708L7.293 8 2.146 2.854z"/>
            </svg>
        </button>
    </div>

    <!-- New messages indicator -->
    <div x-show="feedScrolled_ && feedNewCount > 0"
         @click="scrollToFeedTop()"
         class="absolute top-16 right-4 px-3 py-2 bg-accent text-black text-sm rounded-lg shadow-lg cursor-pointer z-50 font-medium">
        <span x-text="feedNewCount"></span> new message<span x-show="feedNewCount > 1">s</span>
    </div>

    <!-- Drawer Content (virtual scroll container) -->
    <div class="flex-1 overflow-y-auto" id="feed-container"
         @scroll="handleFeedScroll()">
        <div id="feed-virtual-list" class="relative"></div>
    </div>
</div>
```

- [ ] **Step 3: Add CSS for drawer**

In the `<style>` block (around line 24-32), add:

```css
#feed-container {
    height: calc(100vh - 57px);
    scroll-behavior: smooth;
}
#feed-virtual-list {
    position: relative;
}
.transformation-card {
    background: #141414;
}
.transformation-card:hover {
    background: #1a1a1a;
}
```

- [ ] **Step 4: Initialize `feedOpen`, `transformations`, and scroll state in Alpine data**

In the `dashboard()` function (around line 1188-1201), add to the returned object:

```javascript
feedOpen: false,
transformations: [],
feedScrolled_: false,
feedNewCount: 0,
feedScrollY: 0,
feedItemHeight: 160,
feedBuffer: 5,
```

- [ ] **Step 5: Add `fetchTransformations()` method**

Add to the `dashboard()` function, after `fetchStats()`:

```javascript
async fetchTransformations() {
    try {
        const prevLen = this.transformations.length;
        const response = await fetch('/transformations/feed?limit=50');
        if (response.ok) {
            const data = await response.json();
            const newLen = (data.transformations || []).length;
            if (this.feedScrolled_ && newLen > prevLen) {
                this.feedNewCount = newLen - prevLen;
            }
            this.transformations = data.transformations || [];
            this.renderTransformations();
        }
    } catch (e) {
        console.error('Failed to fetch transformations:', e);
    }
},

scrollToFeedTop() {
    const container = document.getElementById('feed-container');
    if (container) {
        container.scrollTop = 0;
        this.feedScrolled_ = false;
        this.feedNewCount = 0;
        this.renderTransformations();
    }
},

handleFeedScroll() {
    const container = document.getElementById('feed-container');
    if (!container) return;

    this.feedScrollY = container.scrollTop;
    this.feedScrolled_ = container.scrollTop > 50;
    this.renderTransformations();
},
```

- [ ] **Step 6: Call `fetchTransformations()` on init and on poll interval**

In `init()` (around line 1202-1211), add `await this.fetchTransformations();` and update the poll interval:

```javascript
async init() {
    await this.fetchStats();
    await this.fetchTransformations();

    this.pollInterval = setInterval(() => {
        this.fetchStats();
        this.fetchTransformations();
    }, 3000);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') {
            this.fetchStats();
        }
    });
},
```

- [ ] **Step 7: Verify in browser**

Open dashboard, click "Live Feed" button — sidebar drawer should appear with empty state or feed content.

- [ ] **Step 8: Commit**

```bash
git add headroom/dashboard/templates/dashboard.html
git commit -m "feat(dashboard): add sidebar drawer shell for live feed

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Virtual Scrolling + Side-by-Side Diff

**Files:**
- Modify: `headroom/dashboard/templates/dashboard.html` — replace placeholder with virtual scroll + diff panels

- [ ] **Step 1: Add virtual scroll rendering methods**

Add to the `dashboard()` function — these replace the placeholder `renderTransformations()`:

```javascript
renderTransformations() {
    const container = document.getElementById('feed-virtual-list');
    if (!container) return;

    const scrollTop = this.feedScrollY;
    const viewportHeight = (document.getElementById('feed-container')?.clientHeight || 600);

    const totalHeight = this.transformations.length * this.feedItemHeight;
    container.style.height = totalHeight + 'px';

    // Calculate visible range with buffer
    const startIdx = Math.max(0, Math.floor(scrollTop / this.feedItemHeight) - this.feedBuffer);
    const endIdx = Math.min(
        this.transformations.length,
        Math.ceil((scrollTop + viewportHeight) / this.feedItemHeight) + this.feedBuffer
    );

    const visible = this.transformations.slice(startIdx, endIdx);
    const offsetTop = startIdx * this.feedItemHeight;

    let html = `<div style="position: absolute; top: ${offsetTop}px; width: 100%;">`;
    html += visible.map((t, i) => this.renderTransformationCard(t, startIdx + i)).join('');
    html += '</div>';

    container.innerHTML = html;
},

renderTransformationCard(t, idx) {
    const msgs = (t.request_messages || []).map(m => m.content || '').join('');
    const response = t.response_content || '';
    const before = msgs.substring(0, 2000) + (msgs.length > 2000 ? '\n\n[truncated]' : '');
    const after = response.substring(0, 2000) + (response.length > 2000 ? '\n\n[truncated]' : '');

    const time = t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '--:--:--';
    const model = (t.model || 'unknown').replace(/^(anthropic\.|openai\.)/, '').substring(0, 25);
    const tokensSaved = t.tokens_saved || 0;
    const savingsPct = ((t.savings_percent || 0)).toFixed(0);

    return `
        <div class="transformation-card border-b border-border p-3" data-idx="${idx}" style="height: ${this.feedItemHeight}px; box-sizing: border-box;">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2 min-w-0">
                    <span class="text-xs font-mono text-gray-400 truncate">${this.escapeHtml(model)}</span>
                    <span class="text-xs text-gray-600 shrink-0">·</span>
                    <span class="text-xs text-emerald-400 shrink-0">${tokensSaved} tok</span>
                    <span class="text-xs text-gray-600 shrink-0">(${savingsPct}%)</span>
                </div>
                <span class="text-xs text-gray-600 shrink-0">${time}</span>
            </div>
            <div class="grid grid-cols-2 gap-2" style="height: 115px;">
                <div class="rounded border border-red-900/30 overflow-hidden flex flex-col">
                    <div class="bg-[#1a0808] px-2 py-1 border-b border-red-900/30 shrink-0">
                        <span class="text-[10px] text-red-400 uppercase tracking-wide font-semibold">Before</span>
                    </div>
                    <div class="diff-before bg-[#120606] p-2 font-mono text-[11px] text-gray-300 overflow-auto flex-1"
                         >${this.escapeHtml(before)}</div>
                </div>
                <div class="rounded border border-emerald-900/30 overflow-hidden flex flex-col">
                    <div class="bg-[#081a08] px-2 py-1 border-b border-emerald-900/30 shrink-0">
                        <span class="text-[10px] text-emerald-400 uppercase tracking-wide font-semibold">After</span>
                    </div>
                    <div class="diff-after bg-[#0a120a] p-2 font-mono text-[11px] text-gray-300 overflow-auto flex-1"
                         >${this.escapeHtml(after)}</div>
                </div>
            </div>
            ${(t.transforms_applied || []).length > 0 ? `
                <div class="flex flex-wrap gap-1 mt-1.5">
                    ${t.transforms_applied.slice(0, 4).map(tr => `<span class="text-[9px] px-1.5 py-0.5 bg-border rounded font-mono text-gray-500">${this.escapeHtml(tr)}</span>`).join('')}
                    ${t.transforms_applied.length > 4 ? `<span class="text-[9px] text-gray-600">+${t.transforms_applied.length - 4}</span>` : ''}
                </div>
            ` : ''}
        </div>
    `;
},

escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
},
```

- [ ] **Step 2: Verify virtual scroll works**

Open dashboard, open live feed, make requests through the proxy — feed should populate with transformations and scroll smoothly with virtual rendering.

- [ ] **Step 3: Commit**

```bash
git add headroom/dashboard/templates/dashboard.html
git commit -m "feat(dashboard): add virtual scrolling with side-by-side diff panels

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Dashboard E2E Tests

**Files:**
- Create: `tests/test_dashboard/test_live_feed.py`

- [ ] **Step 1: Write E2E tests**

```python
import pytest


@pytest.fixture
def dashboard_url():
    return "http://localhost:8787/dashboard"


def test_live_feed_button_exists(page, dashboard_url):
    """The Live Feed button should be visible in the dashboard header."""
    page.goto(dashboard_url)
    feed_button = page.locator("#feed-toggle")
    assert feed_button.is_visible(), "Live Feed button not visible in header"


def test_live_feed_drawer_opens(page, dashboard_url):
    """Clicking Live Feed should open the sidebar drawer."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    page.wait_for_timeout(400)
    # Check drawer is displayed (x-show becomes visible)
    drawer = page.locator('[x-show="feedOpen"]')
    assert drawer.count() > 0


def test_live_feed_shows_empty_state(page, dashboard_url):
    """Feed should show empty state when no transformations available."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    page.wait_for_timeout(1000)
    # Check for empty state or feed container
    feed_container = page.locator("#feed-virtual-list")
    assert feed_container.is_visible()


def test_live_feed_fetches_and_displays(page, dashboard_url):
    """Feed should display transformation data after polling."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    # Wait for at least one poll cycle
    page.wait_for_timeout(4000)
    feed_container = page.locator("#feed-virtual-list")
    content = feed_container.inner_html()
    # Should have at least empty state
    assert len(content) >= 0
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_dashboard/test_live_feed.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_dashboard/test_live_feed.py
git commit -m "test(dashboard): add E2E tests for live feed feature

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Playwright Screenshot Verification

- [ ] **Step 1: Start the proxy and navigate to dashboard**

Open `http://localhost:8787/dashboard` in browser.

- [ ] **Step 2: Click "Live Feed" to open the drawer**

- [ ] **Step 3: Take screenshot**

Use `mcp__plugin_playwright_playwright__browser_take_screenshot` to capture the dashboard with the live feed open.

---

## Spec Coverage Check

1. Right sidebar drawer — Task 2 (Step 1-2)
2. Side-by-side diff — Task 3 (Step 1, `renderTransformationCard`)
3. Auto-stream with fixed-view detection — Task 2 (Step 5: `feedScrolled_`, `feedNewCount`, `scrollToFeedTop`)
4. Virtual scrolling — Task 3 (Step 1, `renderTransformations` with `startIdx/endIdx` windowing)
5. Batch polling every ~3s — Task 2 (Step 6: `fetchTransformations` in poll interval)
6. New dedicated `/transformations/feed` endpoint — Task 1

All spec requirements are covered.
