from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [f"pages/{p.name}" for p in sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_exception(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
