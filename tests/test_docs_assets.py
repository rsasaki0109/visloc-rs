import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
REFERENCE_ROOTS = [
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "examples",
    ROOT / "tests",
    ROOT / "crates",
    ROOT / "pipelines",
    ROOT / "src",
    ROOT / "benchmarks",
]


def iter_reference_files():
    for root in REFERENCE_ROOTS:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if ASSET_DIR in path.parents:
                continue
            yield path


class DocsAssetTests(unittest.TestCase):
    def test_docs_assets_are_referenced_outside_asset_directory(self):
        references = []
        for path in iter_reference_files():
            try:
                references.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue

        unused = []
        for asset in sorted(ASSET_DIR.iterdir()):
            if not asset.is_file():
                continue
            if not any(asset.name in text for text in references):
                unused.append(asset.name)

        self.assertEqual(
            unused,
            [],
            "docs/assets contains files that are not referenced by README, docs, "
            "scripts, examples, tests, crates, pipelines, src, or benchmarks",
        )


if __name__ == "__main__":
    unittest.main()
