"""MIDAS CLI Entrypoint.

Usage:
    python src/main.py --input data/samples/invoices/INV-0047.jpg --doc-type invoice
    python src/main.py --batch data/samples/invoices/ --doc-type invoice
    python src/main.py --search "TATA METALIKS"
    python src/main.py --stats
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.runner import run_pipeline
from src.storage.database import search, get_metrics, init_db


def main():
    parser = argparse.ArgumentParser(
        description="MIDAS — Multimodal Intelligent Document Analysis System CLI"
    )
    parser.add_argument(
        "--input", "-i", type=str, help="Path to single document (PDF, PNG, JPG)"
    )
    parser.add_argument(
        "--batch", "-b", type=str, help="Directory of documents to process in batch"
    )
    parser.add_argument(
        "--doc-type", "-t", type=str, default="auto",
        choices=["auto", "invoice", "technical_spec", "inspection_report"],
        help="Target schema document type (default: auto)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="outputs/extracted",
        help="Directory to save extracted JSON files"
    )
    parser.add_argument(
        "--search", "-s", type=str,
        help="Search query across stored document database using FTS5"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print database extraction metrics and rejection rates"
    )

    args = parser.parse_args()

    # Query metrics
    if args.stats:
        metrics = get_metrics()
        print("\n=== MIDAS Pipeline Metrics ===")
        print(f"Total Documents Validated: {metrics['total_documents']}")
        print(f"Total Rejections:         {metrics['total_rejections']}")
        print(f"Overall Avg Confidence:   {metrics['overall_avg_confidence'] * 100:.1f}%")
        print(f"Rejection Rate:           {metrics['rejection_rate'] * 100:.1f}%")
        print("\nBreakdown by Type:")
        for dt, info in metrics.get("by_type", {}).items():
            print(f"  - {dt:<20}: {info['count']} docs (avg conf: {info['avg_confidence'] * 100:.1f}%)")
        print("==============================\n")
        return

    # Full-text search
    if args.search:
        print(f"\nSearching database for: '{args.search}'...")
        results = search(args.search, doc_type=args.doc_type if args.doc_type != "auto" else None)
        print(f"Found {len(results)} matching records:\n")
        for idx, r in enumerate(results, 1):
            print(f"[{idx}] ID: {r['id']} | Type: {r['doc_type']} | Source: {r['source_file']} | Conf: {r['confidence'] * 100:.1f}%")
            data = r["extracted_data"]
            summary = ""
            if r["doc_type"] == "invoice":
                summary = f"Vendor: {data.get('vendor_name')} | Inv#: {data.get('invoice_number')} | Total: {data.get('currency', 'INR')} {data.get('grand_total')}"
            elif r["doc_type"] == "technical_spec":
                summary = f"Equipment: {data.get('equipment_name')} | Model: {data.get('model_number')} | Params: {len(data.get('parameters', []))}"
            elif r["doc_type"] == "inspection_report":
                summary = f"Site: {data.get('site_name')} | Inspector: {data.get('inspector_name')} | Items: {len(data.get('checklist_items', []))}"
            print(f"    {summary}")
        return

    # Single file processing
    if args.input:
        print(f"\nProcessing document: {args.input}")
        result = run_pipeline(
            input_path=args.input,
            doc_type=args.doc_type,
            output_dir=args.output_dir
        )
        if result["status"] == "success":
            print(f"SUCCESS: Extracted to {result['output_path']}")
            print(f"Document Type: {result['doc_type']}")
            print(f"Confidence:    {result['confidence'] * 100:.1f}%")
            print("\nExtracted Data:")
            print(json.dumps(result["data"], indent=2))
        else:
            print(f"REJECTED: {result.get('reason')}")
            print(f"Confidence: {result.get('confidence', 0.0) * 100:.1f}%")
        return

    # Batch processing
    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Error: directory does not exist: {args.batch}")
            sys.exit(1)

        valid_exts = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        files = [p for p in batch_dir.iterdir() if p.suffix.lower() in valid_exts]
        print(f"\nFound {len(files)} documents in {args.batch}. Processing batch...")

        success_count = 0
        reject_count = 0

        for f in files:
            res = run_pipeline(f, doc_type=args.doc_type, output_dir=args.output_dir)
            if res["status"] == "success":
                success_count += 1
                print(f"  [OK]   {f.name:<25} ({res['doc_type']}, conf: {res['confidence'] * 100:.0f}%)")
            else:
                reject_count += 1
                print(f"  [REJ]  {f.name:<25} ({res.get('reason', 'Validation failed')[:40]}...)")

        print(f"\nBatch Complete: {success_count} validated, {reject_count} rejected.\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
