# Validation Checklist

Use this list after each refactor batch.

1. Syntax and diagnostics
- No new errors in touched Python files.
- Imports are valid after helper extraction.

2. Duplication removal
- Old local duplicate definitions were removed or intentionally retained with reason.
- Shared helper is used consistently by updated files.

3. Behavior safety
- Function/class signatures used by callers are unchanged, or call sites were updated safely.
- Script/notebook outputs are expected for at least one sanity run.

4. Scope discipline
- No unrelated files were modified.
- No public API changes without explicit note.

5. Documentation
- Summary includes extracted helpers, affected files, and remaining risks.
