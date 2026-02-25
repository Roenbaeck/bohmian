# Copilot instructions (diffusion)

## Build / PDF generation
- This repo does **not** have a local LaTeX toolchain installed (no `latexmk`, `pdflatex`, etc.).
- The author uses **OpenAI Prism** to generate PDFs from LaTeX sources.

## Figures workflow
- For figures, set up a Python virtual environment (`venv`) for reproducible plotting.
- Create:
  - `scripts/` for Python code that generates figures
  - `figures/` for resulting **PDF** figure outputs

## Validation preference
- If you need to run validations (e.g., quick numerical checks, data sanity checks, figure generation), prefer using the **local venv Python** rather than MCP-based execution.
