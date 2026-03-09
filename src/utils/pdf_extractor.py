"""
pdf_extractor.py
────────────────
Standalone script that extracts structured invoice data from a PDF using
Azure OpenAI GPT-4.1 (vision).

Pipeline:
  1. Render every PDF page to a high-res JPEG via pypdfium2
  2. Base64-encode each page image
  3. Send images + a structured prompt to Azure OpenAI GPT-4.1 vision
  4. Parse the model's JSON response and save to disk

Usage:
  python src/utils/pdf_extractor.py --pdf source_data_files/Invoice_1.pdf
  python utils/pdf_extractor.py --pdf invoice.pdf --output result.json
  python utils/pdf_extractor.py --help
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from openai import AzureOpenAI

# Handle both `python -m utils.pdf_extractor` and `python utils/pdf_extractor.py`
try:
    from utils.llm import get_llm_client
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.llm import get_llm_client

# Model deployment name — keep in sync with llm.py
LLM_MODEL = "gpt-4.1"

# ──────────────────────────────────────────────────────────────────────
# Schema & Prompts
# ──────────────────────────────────────────────────────────────────────

INVOICE_SCHEMA = {
    "InvoiceNumber": "",
    "PurchaseOrderNumber": "",
    "CustomerName": "",
    "CustomerAddress": "",
    "DeliveryDate": "",
    "PayableBy": "",
    "Products": [
        {"Id": "", "Description": "", "UnitPrice": 0.0, "Quantity": 0, "Total": 0.0}
    ],
    "Returns": [
        {"Id": "", "Description": "", "Quantity": 0, "Reason": ""}
    ],
    "TotalQuantity": 0,
    "TotalPrice": 0.0,
    "ProductsSignatures": [
        {"Type": "", "Name": "", "IsSigned": False}
    ],
    "ReturnsSignatures": [
        {"Type": "", "Name": "", "IsSigned": False}
    ],
}

SYSTEM_PROMPT = (
    "You are an expert invoice data extraction assistant. "
    "Extract all structured information from the invoice image(s) and return ONLY valid JSON "
    "that exactly matches the schema provided — no markdown fences, no extra keys. "
    "For handwritten notes include them in the most relevant fields. "
    "For signatures detect: type (e.g. 'Customer', 'Supplier'), name (if legible), "
    "and whether the field is actually signed (IsSigned: true/false)."
)

USER_PROMPT = (
    "Extract all invoice data from the provided page image(s) and return JSON matching "
    "this schema exactly:\n\n"
    + json.dumps(INVOICE_SCHEMA, indent=2)
    + "\n\nRules:\n"
    "- Empty string for missing text fields, 0 for missing numbers, false for missing booleans\n"
    "- Products / Returns must list ALL line items (including handwritten ones)\n"
    "- Detect ALL signature areas and set IsSigned accordingly"
)

# ──────────────────────────────────────────────────────────────────────
# PDF → Base64 JPEG images
# ──────────────────────────────────────────────────────────────────────

def pdf_to_base64_images(pdf_path: Path, scale: float = 2.0) -> list[str]:
    """Render every PDF page to a high-res JPEG and return base64 strings."""
    doc = pdfium.PdfDocument(str(pdf_path))
    images_b64: list[str] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images_b64.append(b64)
        print(
            f"  Page {page_index + 1}: {pil_img.size[0]}x{pil_img.size[1]} px  "
            f"| {len(b64):,} base64 chars"
        )
    doc.close()
    return images_b64

# ──────────────────────────────────────────────────────────────────────
# GPT-4.1 Vision extraction
# ──────────────────────────────────────────────────────────────────────

def extract_invoice(client: AzureOpenAI, page_images: list[str]) -> dict:
    """Send page images to GPT-4.1 vision and return parsed invoice JSON."""
    # Build multi-modal content: text prompt + one image_url block per page
    content: list[dict] = [{"type": "text", "text": USER_PROMPT}]
    for b64 in page_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            }
        )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=4096,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model ignores the instruction
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])  # drop opening fence line
        raw = raw.rsplit("```", 1)[0].strip()  # drop closing fence

    return json.loads(raw)

# ──────────────────────────────────────────────────────────────────────
# CLI & main
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured invoice data from a PDF using Azure OpenAI GPT-4.1 vision."
    )
    parser.add_argument(
        "--pdf", required=True, type=Path,
        help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for the output JSON file. Default: <pdf_path>.Extraction.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate PDF path ──
    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    output_path = (args.output or Path(str(pdf_path) + ".Extraction.json")).resolve()

    print(f"[input]  PDF  : {pdf_path}")
    print(f"[output] JSON : {output_path}")
    print()

    # ── Get Azure OpenAI client from shared llm.py ──
    print("[auth] Initializing Azure OpenAI client via get_llm_client() ...")
    client = get_llm_client()
    print(f"[auth] Client ready  (model: {LLM_MODEL})")
    print()

    # ── Convert PDF pages to images ──
    print(f"[pdf] Converting '{pdf_path.name}' to JPEG page images ...")
    page_images = pdf_to_base64_images(pdf_path)
    print(f"[pdf] Total pages converted: {len(page_images)}")
    print()

    # ── Call GPT-4.1 vision ──
    print(f"[llm] Calling {LLM_MODEL} with {len(page_images)} page image(s) ...")
    invoice_data = extract_invoice(client, page_images)
    print("[llm] Extraction complete.")
    print()

    # ── Save JSON output ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(invoice_data, f, indent=2, ensure_ascii=False)

    print(f"[done] Extraction saved -> {output_path}")
    print()
    print("── Extraction Summary ─────────────────────────────────")
    print(f"  Invoice #          : {invoice_data.get('InvoiceNumber', 'N/A')}")
    print(f"  PO #               : {invoice_data.get('PurchaseOrderNumber', 'N/A')}")
    print(f"  Customer           : {invoice_data.get('CustomerName', 'N/A')}")
    print(f"  Address            : {invoice_data.get('CustomerAddress', 'N/A')}")
    print(f"  Delivery date      : {invoice_data.get('DeliveryDate', 'N/A')}")
    print(f"  Payable by         : {invoice_data.get('PayableBy', 'N/A')}")
    print(f"  Products           : {len(invoice_data.get('Products', []))} line item(s)")
    print(f"  Returns            : {len(invoice_data.get('Returns', []))} line item(s)")
    print(f"  Total quantity     : {invoice_data.get('TotalQuantity', 0)}")
    print(f"  Total price        : {invoice_data.get('TotalPrice', 0)}")
    print(f"  Products signatures: {len(invoice_data.get('ProductsSignatures', []))}")
    print(f"  Returns signatures : {len(invoice_data.get('ReturnsSignatures', []))}")


if __name__ == "__main__":
    main()
