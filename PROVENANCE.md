# Provenance path placeholders

Packaged logs, CSVs, JSON metadata, and source maps use placeholders instead of absolute paths from an author's machine:

- `${BUNDLE_ROOT}`: the root of this extracted AE bundle.
- `${SOURCE_ROOT}`: the root of the pre-packaging research checkout; the remaining relative suffix identifies the original file or run.
- `${AUTHOR_HOME}`: an author home-directory prefix for provenance that is not located under either checkout.

These strings are provenance identifiers, not environment variables that a reviewer must define. Fresh runs record paths under the reviewer's selected `BUNDLE_ROOT`, `RUN_ROOT`, and `MODEL_ROOT`.

`make validate-provenance` scans packaged text files and fails if an absolute Unix or Windows user-home path is reintroduced.
