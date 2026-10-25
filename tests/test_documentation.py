"""Keep repository navigation valid when documentation moves."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def test_local_markdown_links_resolve():
    documents = [ROOT / 'README.md']
    for directory in ('docs', 'audits', 'evidence'):
        documents.extend((ROOT / directory).rglob('*.md'))
    missing = []
    for document in documents:
        text = document.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            link = urlsplit(target)
            if link.scheme or link.netloc or not link.path:
                continue
            if not (document.parent / unquote(link.path)).exists():
                missing.append(f'{document.relative_to(ROOT)}: {target}')
    assert not missing, '\n'.join(missing)
