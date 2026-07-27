#!/usr/bin/env python3
"""Convert the HTML whitepaper demo to PDF using WeasyPrint."""
import sys
from pathlib import Path

HTML_FILE = Path(__file__).resolve().parent / "whitepaper.html"
PDF_FILE = Path(__file__).resolve().parent / "MYTHOS_Whitepaper_v0.1.02.pdf"

def main():
    from weasyprint import HTML
    
    print(f"Reading HTML: {HTML_FILE}")
    html_content = HTML_FILE.read_text(encoding='utf-8')
    print(f"HTML size: {len(html_content):,} bytes, {html_content.count(chr(10)):,} lines")
    
    print("Converting to PDF with WeasyPrint...")
    doc = HTML(filename=str(HTML_FILE))
    doc.write_pdf(str(PDF_FILE))
    
    sz = PDF_FILE.stat().st_size
    print(f"PDF generated: {PDF_FILE}")
    print(f"PDF size: {sz:,} bytes ({sz/1024:.1f} KB)")
    
    # Verify page count
    try:
        import fitz
        pdf = fitz.open(str(PDF_FILE))
        pages = pdf.page_count
        total_lines = 0
        thin = 0
        dense = 0
        for i in range(pages):
            text = pdf[i].get_text()
            lines = [l for l in text.split('\n') if l.strip()]
            total_lines += len(lines)
            if len(lines) < 30:
                thin += 1
            else:
                dense += 1
        pdf.close()
        print(f"Pages: {pages}")
        print(f"Avg lines/page: {total_lines/pages:.1f}")
        print(f"Dense (30+): {dense}, Thin (<30): {thin}")
    except ImportError:
        print("PyMuPDF not available for verification")

if __name__ == "__main__":
    main()
