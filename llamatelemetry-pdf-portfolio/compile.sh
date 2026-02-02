#!/bin/bash
# Compilation script for llamatelemetry v2.2.0 portfolio PDF

set -e  # Exit on error

echo "========================================"
echo "llamatelemetry v2.2.0 Portfolio PDF Compilation"
echo "========================================"
echo ""

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found"
    echo "Install with: sudo apt-get install texlive-latex-extra texlive-pictures"
    exit 1
fi

echo "✓ pdflatex found: $(which pdflatex)"
echo ""

# Compile PDF (3 passes)
echo "Compiling pass 1/3..."
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex > /dev/null 2>&1
echo "✓ Pass 1 complete"

echo "Compiling pass 2/3..."
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex > /dev/null 2>&1
echo "✓ Pass 2 complete"

echo "Compiling pass 3/3..."
pdflatex -interaction=nonstopmode llamatelemetry_v2_2_0_portfolio.tex > /dev/null 2>&1
echo "✓ Pass 3 complete"

echo ""
echo "========================================"
echo "Compilation successful!"
echo "========================================"
echo ""

# Show PDF info
if [ -f "llamatelemetry_v2_2_0_portfolio.pdf" ]; then
    echo "Output file: llamatelemetry_v2_2_0_portfolio.pdf"
    ls -lh llamatelemetry_v2_2_0_portfolio.pdf
    echo ""

    # Show page count if pdfinfo is available
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo llamatelemetry_v2_2_0_portfolio.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ -n "$pages" ]; then
            echo "Total pages: $pages"
        fi
    fi
else
    echo "Error: PDF file not created"
    exit 1
fi

echo ""
echo "Clean up auxiliary files? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Cleaning..."
    rm -f *.aux *.log *.out *.toc *.lof *.lot
    echo "✓ Auxiliary files removed"
fi

echo ""
echo "Done!"
