"""Explicit neutral CPU entrypoint for the bounded original-reference batch.

No exception to the unchanged raw process classifier. Check-only reads actual
CIM identity and does not validate a request or launch any native application.
"""
import argparse
import json

from .reference_bridge_owner_v1 import current_owner_evidence


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True); p.add_argument('--request-sha256', required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    evidence = current_owner_evidence()
    if args.check_only:
        print(json.dumps(dict(state='actual_owner_raw_classification_passed', evidence=evidence,
            request_validated=False, native_launched=False), indent=2))
        return
    from ..native_generated_reference.batch_owner_v2 import run
    result = run(args.request, args.request_sha256)
    print(json.dumps(dict(state=result['state'], capture=result['capture'], result_sha256=result['result_sha256'],
        training_approved=False), indent=2))


if __name__ == '__main__':
    main()
