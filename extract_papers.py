import pdfplumber
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

paper_dir = 'D:/Projects/1111-soh/paper'
files = [
    's41467-024-48779-z (1).pdf',
    's41598-025-30602-4.pdf',
    's41598-026-37850-y.pdf',
]

for f in files:
    fpath = os.path.join(paper_dir, f)
    print(f'=== FILE: {f} ===')
    try:
        with pdfplumber.open(fpath) as pdf:
            print(f'Total pages: {len(pdf.pages)}')
            for i, page in enumerate(pdf.pages[:6]):
                text = page.extract_text()
                if text:
                    print(f'--- PAGE {i+1} ---')
                    print(text[:2000])
                    print()
    except Exception as e:
        print(f'ERROR: {e}')
    print()

print("DONE")
