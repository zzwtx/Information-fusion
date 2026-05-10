import fitz
import os

base = r"D:\FL\智能信息融合"
pdfs = [
    (r"NeurIPS_2025\FedMGP.pdf", "_fedmgp_text.txt"),
    (r"ICLR_2026\pFedMMA.pdf", "_pfedmma_text.txt"),
    (r"ICML_2025\FedDDA.pdf", "_feddda_text.txt"),
]

for pdf_path, out_name in pdfs:
    full_path = os.path.join(base, pdf_path)
    out_path = os.path.join(base, out_name)
    doc = fitz.open(full_path)
    text_parts = []
    for i, page in enumerate(doc):
        t = page.get_text()
        text_parts.append(f"=== PAGE {i+1} ===\n{t}")
    full_text = "\n".join(text_parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"{pdf_path}: {len(full_text)} chars, {doc.page_count} pages")
    doc.close()

print("Done.")
