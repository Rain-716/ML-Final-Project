from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

USER_PAT = r"(?:^|\n)\s*(?:用户|来访者|求助者|患者)\s*[：:]\s*"
COUNSELOR_PAT = r"(?:^|\n)\s*(?:心理咨询师|咨询师|医生|助手|客服|回答|答复)\s*[：:]\s*"
TURN_SPLIT_RE = re.compile(f"({USER_PAT}|{COUNSELOR_PAT})", flags=re.I)

TEXT_KEYS = ["conversation", "text", "content", "instruction", "input", "prompt", "query", "question"]
RESPONSE_KEYS = ["response", "answer", "output", "target", "reply"]


def iter_files(raw_dir: Path) -> Iterator[Path]:
    for suffix in ("*.jsonl", "*.json", "*.txt", "*.csv"):
        yield from raw_dir.rglob(suffix)


def read_jsonish(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"text": line}
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            # Common dataset structures: {"train": [...]}, {"data": [...]}
            yielded = False
            for key in ["train", "data", "items", "examples"]:
                if isinstance(obj.get(key), list):
                    yield from obj[key]
                    yielded = True
            if not yielded:
                yield obj
    elif path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            buf = []
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
            if buf:
                # Each non-empty line may be one sample; if the file is one huge corpus,
                # downstream duplicate/length filters will handle it.
                for line in buf:
                    yield {"text": line}
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)


def _flatten_strings(obj: Any, max_depth: int = 4) -> List[str]:
    out: List[str] = []
    if max_depth < 0:
        return out
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, dict):
        # Prefer common fields so instruction + output can be paired later.
        for k in TEXT_KEYS + RESPONSE_KEYS:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                out.append(obj[k])
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                out.extend(_flatten_strings(v, max_depth - 1))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_strings(item, max_depth - 1))
    return out


def extract_direct_pair(obj: Any) -> Optional[Tuple[str, str, str]]:
    """Extract (context, user_text, response) from QA-like JSON objects."""
    if not isinstance(obj, dict):
        return None
    text = None
    response = None
    for k in TEXT_KEYS:
        if isinstance(obj.get(k), str) and obj[k].strip():
            text = obj[k].strip()
            break
    for k in RESPONSE_KEYS:
        if isinstance(obj.get(k), str) and obj[k].strip():
            response = obj[k].strip()
            break
    if text and response and len(text) >= 2 and len(response) >= 2:
        return text, text, response
    return None


def parse_dialogue(text: str) -> List[Tuple[str, str, str]]:
    """Parse SoulChat-style dialogue into training pairs.

    Returns a list of (context, last_user_utterance, counselor_response). The
    parser is intentionally tolerant: it handles Chinese colon variants and
    common speaker names.
    """
    text = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not text:
        return []

    # If no explicit speakers exist, no dialogue pair can be extracted.
    if not re.search(USER_PAT, text) or not re.search(COUNSELOR_PAT, text):
        return []

    parts = TURN_SPLIT_RE.split(text)
    turns: List[Tuple[str, str]] = []
    current_speaker = None
    for part in parts:
        if not part:
            continue
        if re.match(USER_PAT, part, flags=re.I):
            current_speaker = "user"
            continue
        if re.match(COUNSELOR_PAT, part, flags=re.I):
            current_speaker = "counselor"
            continue
        content = part.strip()
        if current_speaker and content:
            # Merge consecutive same-speaker fragments.
            if turns and turns[-1][0] == current_speaker:
                turns[-1] = (current_speaker, turns[-1][1] + " " + content)
            else:
                turns.append((current_speaker, content))

    pairs: List[Tuple[str, str, str]] = []
    context_turns: List[str] = []
    last_user: Optional[str] = None
    for speaker, content in turns:
        clean = clean_text(content)
        if not clean:
            continue
        if speaker == "user":
            last_user = clean
            context_turns.append(f"用户：{clean}")
        elif speaker == "counselor" and last_user:
            context = "\n".join(context_turns[-6:])
            pairs.append((context, last_user, clean))
            context_turns.append(f"心理咨询师：{clean}")
            last_user = None
    return pairs


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pairs_from_obj(obj: Any) -> List[Tuple[str, str, str]]:
    direct = extract_direct_pair(obj)
    if direct:
        return [direct]
    pairs: List[Tuple[str, str, str]] = []
    for s in _flatten_strings(obj):
        pairs.extend(parse_dialogue(s))
    return pairs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
