# Quick Start Guide

## Portfolio Folder Structure

```
llamatelemetry-pdf-portfolio/
├── compile.sh                      # Automated compilation script
├── .gitignore                      # Git ignore rules for build artifacts
├── llamatelemetry_v2_2_0_portfolio.pdf     # Compiled portfolio (506 KB, 20 pages)
├── llamatelemetry_v2_2_0_portfolio.tex     # LaTeX source code (47 KB)
├── Makefile                        # Build automation
├── README.md                       # Comprehensive documentation
└── QUICK_START.md                  # This file
```

## Quick Commands

### Compile PDF

```bash
# Option 1: Using Makefile (recommended)
make

# Option 2: Using compile script
./compile.sh

# Option 3: Manual compilation
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex
```

### Clean Build Artifacts

```bash
# Remove auxiliary files only
make clean

# Remove everything including PDF
make cleanall
```

### View Information

```bash
# Show Makefile help
make help

# Show PDF information
make info

# View README
cat README.md
```

## Git Workflow

### Add to Git Repository

```bash
# Navigate to llamatelemetry project root
cd /media/waqasm86/External1/Project-Nvidia-Office/llamatelemetry

# Add portfolio folder
git add llamatelemetry-pdf-portfolio/

# Commit
git commit -m "Add llamatelemetry v2.2.0 professional portfolio with LaTeX source"

# Push to GitHub (when ready)
git push origin main
```

### What Gets Committed

The `.gitignore` file ensures only source files are tracked:

**Tracked:**
- `llamatelemetry_v2_2_0_portfolio.pdf` (compiled portfolio)
- `llamatelemetry_v2_2_0_portfolio.tex` (LaTeX source)
- `Makefile` (build automation)
- `compile.sh` (compilation script)
- `README.md` (documentation)
- `QUICK_START.md` (this file)
- `.gitignore` (ignore rules)

**Ignored:**
- `*.aux`, `*.log`, `*.out`, `*.toc` (LaTeX build artifacts)
- Editor temporary files

## File Descriptions

### `llamatelemetry_v2_2_0_portfolio.pdf`
- **Size:** 506 KB
- **Pages:** 20
- **Content:** Professional portfolio showcasing llamatelemetry v2.2.0 project
- **Highlights:** 5 TikZ diagrams, 14 tables, Notebook 11 flagship section

### `llamatelemetry_v2_2_0_portfolio.tex`
- **Size:** 47 KB
- **Lines:** ~1,350
- **Sections:** 10 major sections
- **Diagrams:** 5 custom TikZ architectural diagrams
- **Tables:** 14 comprehensive specification tables

### `Makefile`
- Automated build system for PDF compilation
- Includes `clean`, `cleanall`, `info`, `help` targets
- Runs 3 compilation passes for proper references

### `compile.sh`
- Interactive bash script for PDF compilation
- Checks for dependencies
- Shows compilation progress
- Offers to clean auxiliary files

### `README.md`
- Comprehensive documentation (5.3 KB)
- Portfolio overview and structure
- Compilation instructions
- LaTeX dependencies
- Design philosophy
- Version history

### `.gitignore`
- Git ignore rules for build artifacts
- Prevents committing temporary files
- Keeps repository clean

## Portfolio Contents

### Section Breakdown

1. **Executive Summary** (1 page) - Project overview
2. **Architecture** (2 pages) - Split-GPU design with diagrams
3. **Notebooks 01-10** (3 pages) - Core functionality demos
4. **Notebook 11** (7 pages) - FLAGSHIP: GGUF visualization
5. **Performance** (2 pages) - Benchmarks and metrics
6. **Production** (1 page) - API compatibility
7. **Innovations** (1 page) - Technical breakthroughs
8. **Skills** (1 page) - Demonstrated competencies
9. **Links** (1 page) - Resources and repositories
10. **About** (1 page) - Professional summary

### Key Metrics

- **Model:** Llama-3.2-3B-Instruct, Q4_K_M quantization
- **Size:** 1.88 GB (5.6× compression from 10.6 GB)
- **Architecture:** 28 layers, 896 attention heads, 3,072 hidden dim
- **Graph:** 929 nodes, 981 edges
- **Visualizations:** 8 Graphistry dashboards
- **Performance:** 48 tok/s on Tesla T4
- **GPUs:** Dual T4 (15GB each) with split workload

## Support

For issues or questions about the portfolio:
1. Read [README.md](README.md) for detailed documentation
2. Check llamatelemetry documentation: https://llamatelemetry.github.io
3. View source repository: https://github.com/llamatelemetry/llamatelemetry

## Version

**Portfolio Version:** 2.2.0
**Created:** January 25, 2026
**Last Updated:** January 25, 2026
