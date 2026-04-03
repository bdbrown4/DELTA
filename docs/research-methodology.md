# Research Methodology and AI Assistance

This project was developed by a solo software engineer using LLMs and AI agents as force-multipliers throughout the research process. All architectural decisions, experimental design, and scientific direction were driven by the human researcher. AI tools (primarily Claude, Gemini, and GitHub Copilot) were used extensively for:

- **Code generation** — boilerplate PyTorch, training loops, and data loading utilities
- **Debugging** — diagnosing gradient issues, tracking down shape mismatches, fixing convergence failures
- **Sounding board** — stress-testing architectural hypotheses, reviewing experimental designs
- **Documentation** — drafting and refining explanatory text

---

## What the AI Tools Did *Not* Do

- Identify the three-paradigm gap
- Formulate the edge-as-first-class-citizen thesis
- Design the 37-phase ablation structure
- Recognize when results were confounded (e.g., Phase 27 batch-1 bug)
- Decide which failures were scientifically meaningful vs. implementation artifacts

Those required the human researcher to look at the numbers and reason about what they meant.

---

## Disclosure

This is consistent with how most modern ML research is conducted. It is disclosed here unapologetically and will be noted in any resulting publication.
