"""CPU-only: README.md must match what scripts/make_readme.py renders from the template and results/."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("make_readme", ROOT / "scripts" / "make_readme.py")
mr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mr)


def test_readme_is_in_sync_with_results():
    rendered = mr.render(mr.TEMPLATE.read_text(), mr.build())
    assert (ROOT / "README.md").read_text() == rendered, "run: python scripts/make_readme.py"


def test_template_placeholders_all_resolve():
    keys = set(re.findall(r"\{\{(\w+)\}\}", mr.TEMPLATE.read_text()))
    assert keys <= set(mr.build()), keys - set(mr.build())


def test_readme_has_required_sections_in_order():
    text = (ROOT / "README.md").read_text()
    heads = [h for h in re.findall(r"^## (.+)$", text, flags=re.M)]
    want = ["Headline results", "Architecture", "Optimisation ladder", "Key findings", "Quality", "Scaling",
            "Reproduce", "Limitations and not done"]
    idx = [heads.index(w) for w in want]
    assert idx == sorted(idx)


def test_readme_images_exist():
    text = (ROOT / "README.md").read_text()
    for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        assert (ROOT / rel).is_file(), rel
