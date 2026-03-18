"""
Ground Truth Runner — tests classifier and (optionally) full E2E pipeline.

Usage:
  python run_ground_truth.py              # classifier-only (fast, no server)
  python run_ground_truth.py --e2e        # full E2E against running server
  python run_ground_truth.py --e2e --url http://localhost:8001
"""

import json
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

GROUND_TRUTH_FILE = os.path.join(os.path.dirname(__file__), "test_ground_truth.json")


def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_classifier_tests(dataset):
    """Test classifier accuracy against ground truth. No server required."""
    from retrieval.query_classifier import classify_query

    passed = 0
    failed = 0
    failures = []

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        expected_type = item["tipo_atteso"]

        result = classify_query(query)
        actual_type = result.query_type

        if actual_type == expected_type:
            passed += 1
            print(f"  OK  #{qid:02d}: \"{query}\" → {actual_type}")
        else:
            failed += 1
            failures.append(item)
            print(f"  FAIL #{qid:02d}: \"{query}\" → {actual_type} (expected: {expected_type})")

    print(f"\n{'='*60}")
    print(f"CLASSIFIER: {passed}/{passed+failed} passed")
    if failures:
        print(f"FAILURES ({failed}):")
        for f in failures:
            print(f"  - #{f['id']}: \"{f['query']}\" (expected: {f['tipo_atteso']})")
    print(f"{'='*60}")
    return failed == 0


def run_e2e_tests(dataset, base_url):
    """Full E2E tests against running server. Checks response content + latency."""
    import requests

    passed = 0
    failed = 0
    failures = []

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        expected_type = item["tipo_atteso"]
        max_latency = item["latenza_max_secondi"]
        must_contain = item.get("risposta_contiene", [])
        must_not_contain = item.get("risposta_NON_contiene", [])
        needs_pages = item.get("pagine_citate", False)

        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/api/chat/sync",
                json={"query": query},
                timeout=max_latency + 10,
            )
            elapsed = time.time() - start
            data = resp.json()

            # Server returns "answer", not "response"
            response_text = data.get("answer", data.get("response", ""))
            citations = data.get("citations", [])
            backend = data.get("backend_used", "?")
            latency_server = data.get("latency_s", elapsed)

            issues = []

            # Check latency (use wall-clock time which includes network)
            if elapsed > max_latency:
                issues.append(f"latency={elapsed:.1f}s (max {max_latency}s)")

            # Check must-contain (case-insensitive)
            resp_lower = response_text.lower()
            for keyword in must_contain:
                if keyword.lower() not in resp_lower:
                    issues.append(f"missing '{keyword}'")

            # Check must-not-contain
            for keyword in must_not_contain:
                if keyword.lower() in resp_lower:
                    issues.append(f"unwanted '{keyword}'")

            # Check citations (only for focused/comprehensive)
            if needs_pages and not citations and len(response_text) > 50:
                # Check if response mentions pages inline (e.g. "pagina 5")
                import re
                has_page_ref = bool(re.search(r'pagin[ae]\s+\d+', resp_lower))
                if not has_page_ref:
                    issues.append("no citations")

            if issues:
                failed += 1
                failures.append({"item": item, "issues": issues, "latency": elapsed,
                                 "backend": backend})
                print(f"  FAIL #{qid:02d}: \"{query}\" [{elapsed:.1f}s/{backend}] — {', '.join(issues)}")
            else:
                passed += 1
                print(f"  OK   #{qid:02d}: \"{query}\" [{elapsed:.1f}s/{backend}]")

        except requests.Timeout:
            elapsed = time.time() - start
            failed += 1
            failures.append({"item": item, "issues": [f"TIMEOUT after {elapsed:.1f}s"],
                             "latency": elapsed, "backend": "?"})
            print(f"  FAIL #{qid:02d}: \"{query}\" — TIMEOUT after {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            failed += 1
            failures.append({"item": item, "issues": [str(e)],
                             "latency": elapsed, "backend": "?"})
            print(f"  FAIL #{qid:02d}: \"{query}\" — ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"E2E: {passed}/{passed+failed} passed")
    if failures:
        print(f"FAILURES ({failed}):")
        for f in failures:
            item = f["item"]
            print(f"  - #{item['id']}: \"{item['query']}\" [{f['latency']:.1f}s/{f['backend']}] — {', '.join(f['issues'])}")
    print(f"{'='*60}")

    # Latency summary by type
    print("\nLATENCY SUMMARY:")
    type_latencies = {}
    for f in failures:
        t = f["item"]["tipo_atteso"]
        type_latencies.setdefault(t, []).append(f["latency"])
    # Note: we don't track passed latencies in this simple version
    print(f"  Total: {passed} passed, {failed} failed out of {len(dataset)}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Ground Truth Runner")
    parser.add_argument("--e2e", action="store_true", help="Run full E2E tests (needs server)")
    parser.add_argument("--url", default="http://localhost:8001", help="Server base URL")
    args = parser.parse_args()

    dataset = load_ground_truth()
    print(f"Loaded {len(dataset)} ground truth entries from {GROUND_TRUTH_FILE}\n")

    # Always run classifier tests first
    print("=" * 60)
    print("PHASE 1: CLASSIFIER ACCURACY")
    print("=" * 60)
    classifier_ok = run_classifier_tests(dataset)

    if args.e2e:
        print(f"\n{'='*60}")
        print("PHASE 2: FULL E2E (server at {})".format(args.url))
        print("=" * 60)
        e2e_ok = run_e2e_tests(dataset, args.url)
        sys.exit(0 if (classifier_ok and e2e_ok) else 1)
    else:
        print("\nSkipping E2E tests. Use --e2e to run against server.")
        sys.exit(0 if classifier_ok else 1)


if __name__ == "__main__":
    main()
