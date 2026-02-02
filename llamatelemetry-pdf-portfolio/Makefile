# Makefile for llamatelemetry v2.2.0 Portfolio PDF Compilation

# LaTeX compiler
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode

# Source and output files
SOURCE = llamatelemetry_v2_2_0_portfolio.tex
PDF = llamatelemetry_v2_2_0_portfolio.pdf

# Auxiliary files to clean
AUX_FILES = *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all pdf clean help

# Default target
all: pdf

# Compile PDF (requires 3 passes for proper references)
pdf:
	@echo "Compiling portfolio PDF (pass 1/3)..."
	@$(LATEX) $(LATEX_FLAGS) $(SOURCE) > /dev/null
	@echo "Compiling portfolio PDF (pass 2/3)..."
	@$(LATEX) $(LATEX_FLAGS) $(SOURCE) > /dev/null
	@echo "Compiling portfolio PDF (pass 3/3)..."
	@$(LATEX) $(LATEX_FLAGS) $(SOURCE) > /dev/null
	@echo "✓ PDF compiled successfully: $(PDF)"
	@ls -lh $(PDF)

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f $(AUX_FILES)
	@echo "✓ Clean complete"

# Clean everything including PDF
cleanall: clean
	@echo "Removing PDF..."
	@rm -f $(PDF)
	@echo "✓ All files removed"

# View PDF info
info:
	@echo "Portfolio PDF Information:"
	@echo "=========================="
	@ls -lh $(PDF) 2>/dev/null || echo "PDF not found. Run 'make pdf' to compile."
	@echo ""
	@pdfinfo $(PDF) 2>/dev/null || echo "pdfinfo not available"

# Show help
help:
	@echo "llamatelemetry v2.2.0 Portfolio Makefile"
	@echo "================================"
	@echo ""
	@echo "Targets:"
	@echo "  make         - Compile PDF (default)"
	@echo "  make pdf     - Compile PDF (3 passes)"
	@echo "  make clean   - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make cleanall - Remove auxiliary files and PDF"
	@echo "  make info    - Show PDF file information"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (texlive-latex-extra)"
	@echo "  - TikZ packages (texlive-pictures)"
	@echo ""
	@echo "Usage:"
	@echo "  make         # Compile the portfolio"
	@echo "  make clean   # Clean up build artifacts"
