# Why DELTA? — Transformer vs Graph Neural Net vs DELTA

<style>
.delta-visual {
  background: #0d1117;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  padding: 32px 24px;
  max-width: 780px;
  margin: 24px auto;
  border-radius: 12px;
  border: 1px solid #30363d;
}
.delta-visual *, .delta-visual *::before, .delta-visual *::after { box-sizing: border-box; }
.delta-visual h2 { font-size: 20px; font-weight: 600; color: #e6edf3; margin: 0 0 6px 0; border: none; }
.delta-visual .dv-subtitle { font-size: 13px; color: #8b949e; margin-bottom: 24px; line-height: 1.5; }
.delta-visual .dv-tab-row { display: flex; gap: 6px; margin-bottom: 20px; flex-wrap: wrap; }
.delta-visual .dv-tab {
  padding: 6px 16px; border-radius: 20px; font-size: 13px; cursor: pointer;
  border: 1px solid #30363d; background: transparent; color: #8b949e;
  transition: all .15s; font-family: inherit;
}
.delta-visual .dv-tab:hover { border-color: #6e7681; color: #c9d1d9; }
.delta-visual .dv-tab.active { background: #161b22; color: #e6edf3; border-color: #8b949e; font-weight: 500; }
.delta-visual .dv-panel { display: none; }
.delta-visual .dv-panel.active { display: block; }
.delta-visual .dv-caption { font-size: 13px; color: #8b949e; margin-bottom: 16px; line-height: 1.6; }
.delta-visual .dv-caption strong { color: #e6edf3; }
.delta-visual .dv-caption em { color: #a5d6ff; font-style: normal; }
.delta-visual .dv-legend { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
.delta-visual .dv-legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }
.delta-visual .dv-legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.delta-visual svg { width: 100%; display: block; }
.delta-visual .c-purple rect, .delta-visual .c-purple circle { fill: rgba(127,119,221,0.12); stroke: #7F77DD; }
.delta-visual .c-teal rect, .delta-visual .c-teal circle { fill: rgba(29,158,117,0.12); stroke: #1D9E75; }
.delta-visual .c-gray rect, .delta-visual .c-gray circle { fill: rgba(136,135,128,0.12); stroke: #888780; }
.delta-visual .c-amber rect, .delta-visual .c-amber circle { fill: rgba(239,159,39,0.12); stroke: #EF9F27; }
.delta-visual text.ts { font-size: 11px; fill: #8b949e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.delta-visual text.th { font-size: 12px; fill: #e6edf3; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.delta-visual .dv-key-insight {
  background: rgba(229,74,74,0.08); border: 1px solid rgba(229,74,74,0.3);
  border-radius: 8px; padding: 12px 16px; margin-top: 24px;
  font-size: 13px; color: #c9d1d9; line-height: 1.6;
}
.delta-visual .dv-key-insight strong { color: #ff7b72; }
</style>

<div class="delta-visual" markdown="0">
<h2>Transformer vs Graph Neural Net vs DELTA</h2>
<p class="dv-subtitle">Three paradigms for relational reasoning — click through the tabs to see the core architectural difference.</p>

<div class="dv-tab-row">
  <button class="dv-tab active" onclick="showDvPanel('transformer', this)">Transformer</button>
  <button class="dv-tab" onclick="showDvPanel('graph', this)">Graph Neural Net</button>
  <button class="dv-tab" onclick="showDvPanel('delta', this)">DELTA</button>
</div>

<div id="dv-panel-transformer" class="dv-panel active">
  <p class="dv-caption">Tokens in a <strong>flat sequence</strong>. Every token attends to every other — attention is the only way relationships are discovered. The model must <em>reconstruct</em> relational structure from sequential position alone.</p>
  <svg viewBox="0 0 680 310">
    <defs>
      <marker id="dv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="#8b949e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    <rect x="40" y="20" width="600" height="36" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.8"/>
    <text class="ts" x="340" y="38" text-anchor="middle" dominant-baseline="central">Token embeddings</text>
    <g class="c-purple"><rect x="56" y="76" width="72" height="38" rx="6" stroke-width="0.8"/><text class="th" x="92" y="95" text-anchor="middle" dominant-baseline="central">Paris</text></g>
    <g class="c-gray"><rect x="148" y="76" width="72" height="38" rx="6" stroke-width="0.5"/><text class="ts" x="184" y="95" text-anchor="middle" dominant-baseline="central">is</text></g>
    <g class="c-gray"><rect x="240" y="76" width="72" height="38" rx="6" stroke-width="0.5"/><text class="ts" x="276" y="95" text-anchor="middle" dominant-baseline="central">capital</text></g>
    <g class="c-gray"><rect x="332" y="76" width="72" height="38" rx="6" stroke-width="0.5"/><text class="ts" x="368" y="95" text-anchor="middle" dominant-baseline="central">of</text></g>
    <g class="c-teal"><rect x="424" y="76" width="72" height="38" rx="6" stroke-width="0.8"/><text class="th" x="460" y="95" text-anchor="middle" dominant-baseline="central">France</text></g>
    <g class="c-gray"><rect x="516" y="76" width="100" height="38" rx="6" stroke-width="0.5"/><text class="ts" x="566" y="95" text-anchor="middle" dominant-baseline="central">relationship</text></g>
    <line x1="92" y1="114" x2="184" y2="150" stroke="#30363d" stroke-width="0.8"/>
    <line x1="92" y1="114" x2="276" y2="150" stroke="#30363d" stroke-width="0.8"/>
    <line x1="92" y1="114" x2="368" y2="150" stroke="#30363d" stroke-width="0.8"/>
    <line x1="92" y1="114" x2="460" y2="150" stroke="#1D9E75" stroke-width="3.5" opacity="0.7"/>
    <line x1="92" y1="114" x2="566" y2="150" stroke="#1D9E75" stroke-width="1.8" opacity="0.35"/>
    <rect x="40" y="150" width="600" height="38" rx="6" fill="rgba(127,119,221,0.12)" stroke="#7F77DD" stroke-width="0.8"/>
    <text class="th" x="300" y="169" text-anchor="middle" dominant-baseline="central">Multi-head self-attention</text>
    <text class="ts" x="600" y="169" text-anchor="end" dominant-baseline="central" fill="#c62626">O(N²)</text>
    <rect x="40" y="208" width="600" height="36" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.5"/>
    <text class="ts" x="340" y="226" text-anchor="middle" dominant-baseline="central">Feed-forward layer</text>
    <line x1="340" y1="244" x2="340" y2="266" stroke="#30363d" stroke-width="1" marker-end="url(#dv-arr)"/>
    <rect x="200" y="268" width="280" height="30" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.5"/>
    <text class="ts" x="340" y="283" text-anchor="middle" dominant-baseline="central">Output (reconstructed relationships)</text>
  </svg>
  <div class="dv-legend">
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#7F77DD"></div>Query token</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#1D9E75"></div>High attention weight</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#888780"></div>Low attention / neutral</div>
  </div>
</div>

<div id="dv-panel-graph" class="dv-panel">
  <p class="dv-caption">Nodes pass messages to neighbors. Relationships are edges with <strong>scalar weights</strong>. Edges are <em>passive conduits</em> — they carry signal from node to node, but can't reason about what kind of relationship they represent.</p>
  <svg viewBox="0 0 680 310">
    <line x1="200" y1="120" x2="460" y2="120" stroke="#1D9E75" stroke-width="3" opacity="0.6"/>
    <rect x="293" y="101" width="94" height="20" rx="4" fill="#0d1117"/>
    <text class="ts" x="340" y="113" text-anchor="middle" dominant-baseline="central">0.91 weight</text>
    <line x1="200" y1="120" x2="340" y2="230" stroke="#30363d" stroke-width="1"/>
    <rect x="232" y="166" width="80" height="18" rx="4" fill="#0d1117"/>
    <text class="ts" x="272" y="177" text-anchor="middle" dominant-baseline="central">0.12 weight</text>
    <line x1="460" y1="120" x2="340" y2="230" stroke="#21262d" stroke-width="1.8" opacity="0.8"/>
    <rect x="396" y="166" width="80" height="18" rx="4" fill="#0d1117"/>
    <text class="ts" x="436" y="177" text-anchor="middle" dominant-baseline="central">0.44 weight</text>
    <line x1="460" y1="120" x2="580" y2="230" stroke="#21262d" stroke-width="2.5" opacity="0.5"/>
    <g class="c-purple"><circle cx="200" cy="120" r="46" stroke-width="1.5"/></g>
    <text class="th" x="200" y="115" text-anchor="middle" dominant-baseline="central">Paris</text>
    <text class="ts" x="200" y="132" text-anchor="middle" dominant-baseline="central">node</text>
    <g class="c-teal"><circle cx="460" cy="120" r="46" stroke-width="1.5"/></g>
    <text class="th" x="460" y="115" text-anchor="middle" dominant-baseline="central">France</text>
    <text class="ts" x="460" y="132" text-anchor="middle" dominant-baseline="central">node</text>
    <g class="c-gray"><circle cx="340" cy="230" r="40" stroke-width="1"/></g>
    <text class="th" x="340" y="225" text-anchor="middle" dominant-baseline="central">Berlin</text>
    <text class="ts" x="340" y="242" text-anchor="middle" dominant-baseline="central">node</text>
    <g class="c-amber"><circle cx="580" cy="230" r="42" stroke-width="1"/></g>
    <text class="th" x="580" y="225" text-anchor="middle" dominant-baseline="central">Germany</text>
    <text class="ts" x="580" y="242" text-anchor="middle" dominant-baseline="central">node</text>
    <rect x="40" y="274" width="340" height="26" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.5"/>
    <text class="ts" x="210" y="287" text-anchor="middle" dominant-baseline="central">Edges are passive — scalar weights only, no reasoning</text>
  </svg>
  <div class="dv-legend">
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#7F77DD"></div>Concept node</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#1D9E75"></div>Strong edge</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#888780"></div>Weak edge</div>
  </div>
</div>

<div id="dv-panel-delta" class="dv-panel">
  <p class="dv-caption">DELTA promotes edges to <strong>first-class citizens</strong>. Nodes and edges both carry rich representations and attend to each other in <strong>parallel dual streams</strong>. Edge-to-edge attention enables reasoning about <em>relationships between relationships</em> — the key to compositional inference.</p>
  <svg viewBox="0 0 680 430">
    <defs>
      <marker id="dv-arr3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="#E24B4A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    <rect x="40" y="14" width="600" height="24" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.5"/>
    <text class="ts" x="340" y="26" text-anchor="middle" dominant-baseline="central" font-weight="500" fill="#c9d1d9">Parallel dual attention — nodes + edges simultaneously</text>
    <rect x="40" y="50" width="280" height="24" rx="4" fill="none" stroke="#534AB7" stroke-width="0.8" stroke-dasharray="4 3"/>
    <text class="ts" x="180" y="62" text-anchor="middle" dominant-baseline="central" fill="#7F77DD">Node attention stream</text>
    <rect x="360" y="50" width="280" height="24" rx="4" fill="none" stroke="#0F6E56" stroke-width="0.8" stroke-dasharray="4 3"/>
    <text class="ts" x="500" y="62" text-anchor="middle" dominant-baseline="central" fill="#1D9E75">Edge attention stream</text>
    <line x1="200" y1="175" x2="390" y2="175" stroke="#1D9E75" stroke-width="2.5" opacity="0.5"/>
    <line x1="200" y1="175" x2="300" y2="285" stroke="#21262d" stroke-width="1.5" opacity="0.6"/>
    <line x1="390" y1="175" x2="300" y2="285" stroke="#EF9F27" stroke-width="2" opacity="0.5"/>
    <line x1="390" y1="175" x2="490" y2="285" stroke="#EF9F27" stroke-width="2.5" opacity="0.6"/>
    <g class="c-teal"><rect x="255" y="155" width="80" height="40" rx="8" stroke-width="1"/></g>
    <text class="th" x="295" y="170" text-anchor="middle" dominant-baseline="central">capital of</text>
    <text class="ts" x="295" y="185" text-anchor="middle" dominant-baseline="central">edge node</text>
    <g class="c-amber"><rect x="355" y="248" width="80" height="40" rx="8" stroke-width="1"/></g>
    <text class="th" x="395" y="263" text-anchor="middle" dominant-baseline="central">located in</text>
    <text class="ts" x="395" y="278" text-anchor="middle" dominant-baseline="central">edge node</text>
    <path d="M295 195 Q295 232 355 258" fill="none" stroke="#E24B4A" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#dv-arr3)" opacity="0.9"/>
    <rect x="196" y="218" width="102" height="18" rx="4" fill="#0d1117"/>
    <text class="ts" x="247" y="227" text-anchor="middle" dominant-baseline="central" fill="#c62626">edge→edge attn</text>
    <g class="c-purple"><circle cx="200" cy="175" r="42" stroke-width="1.5"/></g>
    <text class="th" x="200" y="170" text-anchor="middle" dominant-baseline="central">Paris</text>
    <text class="ts" x="200" y="186" text-anchor="middle" dominant-baseline="central">+ memory</text>
    <g class="c-teal"><circle cx="390" cy="175" r="42" stroke-width="1.5"/></g>
    <text class="th" x="390" y="170" text-anchor="middle" dominant-baseline="central">France</text>
    <text class="ts" x="390" y="186" text-anchor="middle" dominant-baseline="central">+ memory</text>
    <g class="c-gray"><circle cx="300" cy="285" r="38" stroke-width="1"/></g>
    <text class="th" x="300" y="280" text-anchor="middle" dominant-baseline="central">Berlin</text>
    <text class="ts" x="300" y="296" text-anchor="middle" dominant-baseline="central">+ memory</text>
    <g class="c-amber"><circle cx="490" cy="285" r="38" stroke-width="1"/></g>
    <text class="th" x="490" y="280" text-anchor="middle" dominant-baseline="central">Germany</text>
    <text class="ts" x="490" y="296" text-anchor="middle" dominant-baseline="central">+ memory</text>
    <rect x="40" y="342" width="600" height="30" rx="6" fill="rgba(136,135,128,0.08)" stroke="#30363d" stroke-width="0.5"/>
    <text class="ts" x="340" y="357" text-anchor="middle" dominant-baseline="central">Importance router — scores nodes + edges, guides sparse attention</text>
    <rect x="40" y="384" width="185" height="26" rx="4" fill="none" stroke="rgba(229,74,74,0.5)" stroke-width="0.8"/>
    <text class="ts" x="132" y="397" text-anchor="middle" dominant-baseline="central" fill="#c62626">Hot tier (active)</text>
    <rect x="235" y="384" width="185" height="26" rx="4" fill="none" stroke="rgba(186,117,23,0.5)" stroke-width="0.8"/>
    <text class="ts" x="327" y="397" text-anchor="middle" dominant-baseline="central" fill="#9a6010">Warm tier (compressed)</text>
    <rect x="430" y="384" width="210" height="26" rx="4" fill="none" stroke="#30363d" stroke-width="0.6"/>
    <text class="ts" x="535" y="397" text-anchor="middle" dominant-baseline="central">Cold tier (archived)</text>
  </svg>
  <div class="dv-legend">
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#7F77DD"></div>Concept node (+ memory)</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#1D9E75"></div>Rich edge node (novel)</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#E24B4A"></div>Edge-to-edge attention (novel)</div>
    <div class="dv-legend-item"><div class="dv-legend-dot" style="background:#EF9F27"></div>Typed relationship</div>
  </div>
</div>

<div class="dv-key-insight">
  <strong>Why this matters:</strong> The "capital of" edge between Paris→France can attend to the "capital of" edge between Berlin→Germany and recognize they encode the same relationship type. That's structural analogy — impossible in a GNN where edges are just scalars. At 80% feature noise, this mechanism gives DELTA a <strong>+24% accuracy advantage</strong> over standard GNNs (Phase 28).
</div>
</div>

<script>
function showDvPanel(id, btn) {
  document.querySelectorAll('.dv-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.dv-tab').forEach(function(t) { t.classList.remove('active'); });
  document.getElementById('dv-panel-' + id).classList.add('active');
  btn.classList.add('active');
}
</script>

---

## The Three Paradigms

### 1. Transformer

Tokens in a flat sequence, every token attending to every other. The model has no native concept of relationships — it has to discover that "Paris" and "France" are related by reading the words in order and learning the pattern. That's the O(N²) problem: as sequences get longer, the attention cost explodes.

**What it does well:** Flexible, learns arbitrary patterns from data.
**What it lacks:** No structural inductive bias. Relationships must be reconstructed from position alone.

### 2. Graph Neural Network

Relationships are explicit as edges, which is better. But edges are just scalar weights — they're passive wires, not thinkers. The edge connecting Paris→France just says "0.91 strong" and moves signal along. It can't reason about *what kind* of relationship that is, or how it relates to the Berlin→Germany edge.

**What it does well:** Exploits relational structure, message-passing is efficient.
**What it lacks:** Edges are passive conduits. No mechanism for reasoning about relationships between relationships.

### 3. DELTA (Dual Edge-Level Transformer Architecture)

DELTA promotes edges to **first-class computational citizens**. Nodes and edges both carry rich representations and attend to each other simultaneously in parallel streams.

The key mechanism: **edge-to-edge attention**. The "capital of" edge between Paris and France can *attend* to the "capital of" edge between Berlin and Germany — and recognize they're the same relationship type. That's structural analogy, compositional reasoning, and relational inference all in one mechanism.

On top of that:
- **Tiered memory** on every node (hot / warm / cold) — the router manages what's active vs archived
- **Importance router** — scores nodes and edges to guide sparse attention
- **Parallel dual streams** — node attention and edge attention run simultaneously

**What it does well:** Everything above, plus it degrades gracefully under noise.
**What it lacks:** Higher per-layer cost than vanilla GNN (offset by fewer layers needed).

---

## Where the Gap Shows Up

The gap between GNN and DELTA is where the **Phase 28 result** lives. At extreme noise levels (80% corrupted features), DELTA's edge-aware attention maintains **+24% accuracy** over standard GNN approaches.

Why? Because nodes can reason about their neighbors' *relationships*, not just their neighbors' *values*. When node features are noisy, the relational structure (edge-to-edge patterns) is still intact — and DELTA can leverage it.

| Noise Level | Standard GNN | DELTA | Gap |
|------------|-------------|-------|-----|
| 0% (clean) | ~95% | ~97% | +2% |
| 20% | ~88% | ~94% | +6% |
| 50% | ~72% | ~86% | +14% |
| 80% | ~54% | ~78% | **+24%** |

*Results from Phase 28 (noise robustness), synthetic benchmark.*

---

## The Visual Explained

Click through all three tabs in the interactive diagram above:

1. **Transformer tab** — A flat sequence of tokens with O(N²) self-attention. "Paris" must discover its relationship to "France" purely through attention weights. The green lines show where attention concentrates, but every token must attend to every other token to find these patterns.

2. **Graph Neural Net tab** — Nodes connected by edges with scalar weights. Paris→France has weight 0.91, but that edge is just a wire — it carries signal, it doesn't compute. The edge can't look at the Berlin→Germany edge and recognize they encode the same relationship.

3. **DELTA tab** — The red dashed arrow is the key. Edge nodes (green rounded rectangles) represent "capital of" and "located in" as rich learned representations. The edge-to-edge attention arrow means the "capital of" edge can attend to other edges and discover structural analogies. Below that, the importance router and tiered memory complete the architecture.

---

## Connection to DELTA's Experiment Phases

| Phase | What it validates | Paradigm gap addressed |
|-------|-------------------|----------------------|
| Phase 28 | Noise robustness (+24% at 80% noise) | Edge-to-edge attention preserves relational structure when node features degrade |
| Phase 27b | Attention-topology interaction | Shows how edge attention interacts with graph structure |
| Phase 33 | Task-aware graph construction | Hybrid constructor preserves base topology while learning long-range edges |
| Phase 34 | DELTA vs GraphGPS vs GRIT | Head-to-head comparison with state-of-the-art graph transformers |
| Phase 25 | Full-scale FB15k-237 | Validates edge-aware reasoning at knowledge graph scale (14,505 entities) |

---

*Visual and writeup created March 2026.*
