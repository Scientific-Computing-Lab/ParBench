## What this changes

<!-- One or two sentences. Link the issue this closes, if there is one. -->

## Checklist

- [ ] `bash scripts/run_public_tests.sh` passes.
- [ ] Every spec I added or changed validates:
      `python3 scripts/validate_schema.py --spec specs/<unique_id>.json` exits 0.
- [ ] Every spec I added or changed verifies against its pristine baseline:
      `python3 -m harness verify specs/<unique_id>.json`.
- [ ] `manifest.jsonl` changes are appended lines only. No existing entry was edited or
      removed.
- [ ] Run arguments were read from the source's argument parser, not from documentation
      (`python3 scripts/spec_tools/check_spec_argc.py --all`).
- [ ] No machine-specific paths, home directories, hostnames, credentials, or local
      `config/paths.json` values appear in the diff.

## Platform tested

<!-- OS, CPU/GPU, and compiler versions the checks above were run on. -->
