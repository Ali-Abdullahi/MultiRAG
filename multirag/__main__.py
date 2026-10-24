"""Command line entry point: `python -m multirag add <pdf-or-folder>...`.

Chains ingest -> embed -> index, which until now had to be driven from Python by
hand. Every stage is already idempotent, so the CLI adds no state of its own:
re-running `add` over a folder only does the work that is missing.
"""

import argparse
import sys
from pathlib import Path


def _collect_pdfs(paths: list[str]) -> list[Path]:
    """Expand files and folders into a sorted, de-duplicated list of PDFs.

    Sorted so a run over the same folder always processes documents in the same
    order - progress output and partial failures are then reproducible.
    """
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*") if p.suffix.lower() == ".pdf")
        elif path.is_file() and path.suffix.lower() == ".pdf":
            found.add(path.resolve())
        else:
            raise SystemExit(f"not a PDF or folder: {raw}")
    return sorted(found)


def _add_one(pdf: Path, *, client, encoder_holder: list) -> None:
    """Run one PDF through every stage, printing what each stage did."""
    from multirag.embed import embed_document, is_embedded
    from multirag.index import index_document
    from multirag.ingest import ingest_pdf

    record = ingest_pdf(pdf)
    doc_id, pages = record["doc_id"], record["page_count"]
    print(f"  ingested  {pages} pages ({doc_id})")

    if is_embedded(doc_id):
        print("  embedded  already current")
    else:
        if not encoder_holder:
            from multirag.embed.encoder import PageEncoder

            print("  loading encoder (~8s, once per run)...", flush=True)
            encoder_holder.append(PageEncoder.load())
        embed_document(
            doc_id,
            encoder=encoder_holder[0],
            on_page=lambda i, total, _pid: print(f"  embedded  page {i + 1}/{total}", flush=True),
        )

    summary = index_document(doc_id, client=client)
    print(f"  indexed   {summary['points']} points")


def cmd_add(args) -> int:
    """Add PDFs to the index. Returns a process exit code."""
    from multirag.index.client import StoreLockedError, open_client

    pdfs = _collect_pdfs(args.paths)
    if not pdfs:
        print("no PDFs found")
        return 1

    try:
        client = open_client()
    except StoreLockedError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # The encoder is loaded on first need and shared, so a folder of already-
    # embedded documents never pays for it, and a folder of new ones pays once.
    encoder_holder: list = []
    failures = 0
    try:
        for n, pdf in enumerate(pdfs, start=1):
            print(f"[{n}/{len(pdfs)}] {pdf.name}")
            try:
                _add_one(pdf, client=client, encoder_holder=encoder_holder)
            except Exception as error:  # one bad file must not sink a corpus run
                failures += 1
                print(f"  FAILED    {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        client.close()

    print(f"done: {len(pdfs) - failures} added, {failures} failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m multirag")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="ingest, embed and index PDFs")
    add.add_argument("paths", nargs="+", help="PDF files or folders to search for PDFs")
    add.set_defaults(func=cmd_add)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
