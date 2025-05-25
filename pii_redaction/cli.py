#!/usr/bin/env python
import argparse
import os
import random
from .redactor import clean_dataset, tag_pii_in_documents, PIIHandlingMode


def main():
    parser = argparse.ArgumentParser(description="PII Redaction Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add common arguments for PII handling
    def add_pii_handling_args(parser):
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--tag",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.TAG,
            help="Keep PII content between XML tags (default)",
        )
        mode_group.add_argument(
            "--redact",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.REDACT,
            help="Replace PII with just an empty tag",
        )
        mode_group.add_argument(
            "--replace",
            action="store_const",
            dest="mode",
            const=PIIHandlingMode.REPLACE,
            help="Replace PII with fake data",
        )
        parser.set_defaults(mode=PIIHandlingMode.TAG)

        parser.add_argument(
            "--locale",
            default="en_US",
            help="Locale for generating fake data (default: en_US, only used with --replace)",
        )
        parser.add_argument(
            "--engine",
            choices=["transformers", "vllm"],
            default="transformers",
            help="Generation backend to use (default: transformers)",
        )

    # Process JSONL dataset command
    jsonl_parser = subparsers.add_parser(
        "process-jsonl", help="Process a JSONL dataset to handle PII in message content"
    )
    jsonl_parser.add_argument("input", help="Input JSONL file")
    jsonl_parser.add_argument("output", help="Output JSONL file")
    jsonl_parser.add_argument(
        "--device", help="Device to use for processing (e.g., cuda, cpu)"
    )
    add_pii_handling_args(jsonl_parser)

    # Process text files command
    text_parser = subparsers.add_parser(
        "process-text", help="Process a text file with one document per line"
    )
    text_parser.add_argument("input", help="Input file with one document per line")
    text_parser.add_argument("output", help="Output file for processed documents")
    text_parser.add_argument(
        "--device", help="Device to use for processing (e.g., cuda, cpu)"
    )
    add_pii_handling_args(text_parser)

    args = parser.parse_args()

    if args.command == "process-jsonl":
        clean_dataset(
            args.input,
            args.output,
            device=args.device,
            engine=args.engine,
            mode=args.mode,
            locale=args.locale,
        )
    elif args.command == "process-text":
        with open(args.input, "r") as fin:
            doc_count = sum(1 for _ in fin)

        with open(args.input, "r") as fin, open(args.output, "w") as fout:
            documents = (line.strip() for line in fin if line.strip())
            tag_pii_in_documents(
                documents,
                device=args.device,
                engine=args.engine,
                mode=args.mode,
                locale=args.locale,
                output_file=fout,
            )

        mode_descriptions = {
            PIIHandlingMode.TAG: "Tagged",
            PIIHandlingMode.REDACT: "Redacted",
            PIIHandlingMode.REPLACE: "Replaced",
        }
        action = mode_descriptions[args.mode]
        print(f"{action} PII in {doc_count} documents and saved to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
