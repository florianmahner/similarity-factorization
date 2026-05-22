#!/usr/bin/env python3
"""Test job that prints progress every few seconds."""

import argparse
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test")
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    print(f"Starting {args.name} job with {args.total} steps")

    for i in range(1, args.total + 1):
        print(f"[{i}/{args.total}] Processing step {i}... value={random.random():.4f}")
        time.sleep(args.interval)

    print(f"Completed {args.name}!")

if __name__ == "__main__":
    main()
