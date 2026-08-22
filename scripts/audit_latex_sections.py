"""Report evidence density for each LaTeX section and subsection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADING_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}")


def active_source(text: str) -> str:
    text = re.sub(r"\\iffalse.*?\\fi", "", text, flags=re.DOTALL)
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def prose_word_count(text: str) -> int:
    text = re.sub(r"\\begin\{(?:figure\*?|table\*?|equation\*?|align\*?)\}.*?\\end\{[^}]+\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(?:cite|ref|label|url|includegraphics|caption)\*?(?:\[[^]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$&_~^\\]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tex", type=Path)
    args = parser.parse_args()

    source = active_source(args.tex.read_text(encoding="utf-8"))
    matches = list(HEADING_RE.finditer(source))
    print("level\ttitle\twords\tparagraphs\tfigures\ttables\tequations")
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        body = source[start:end]
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if prose_word_count(p) >= 8]
        figures = len(re.findall(r"\\begin\{figure\*?\}", body))
        tables = len(re.findall(r"\\begin\{table\*?\}", body))
        equations = len(re.findall(r"\\begin\{(?:equation\*?|align\*?)\}", body))
        print(
            f"{match.group(1)}\t{match.group(2)}\t{prose_word_count(body)}\t"
            f"{len(paragraphs)}\t{figures}\t{tables}\t{equations}"
        )


if __name__ == "__main__":
    main()
