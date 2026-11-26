"""Small, dependency-free contracts. Importing this module never loads a model."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def messages_for(row):
    messages = row.get('messages') or [{'role': 'user', 'content': row['prompt']}]
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('messages must end with the user prompt, excluding response')
    for message in messages:
        if message.get('role') not in ('system', 'user', 'assistant') or not isinstance(message.get('content'), str):
            raise ValueError('Invalid role/content')
    return messages


def review_payload(row):
    return {key: row.get(key) for key in ('prompt', 'messages', 'response', 'context', 'source', 'split')}


def require_revision(value):
    if not re.fullmatch(r'[a-f0-9]{40}', value):
        raise ValueError('Full immutable model revision SHA (40 lowercase hex) required')


def validate_bio(row):
    vector = row.get('bio_vector20')
    provenance = row.get('bio_provenance') or {}
    if not isinstance(vector, list) or len(vector) != 20 or any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in vector):
        raise ValueError('Finite canonical bio_vector20 required; do not invent neutral values')
    if provenance.get('producer') != 'brothertrue_core' or not provenance.get('trace_sha256'):
        raise ValueError('Real isolated core trace provenance required for physiological conditioning')
    return vector


def read_rows(path, *, approved=False, limit=10000):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages_for(row)
            if approved:
                review = row.get('teacher_review') or {}
                if (review.get('status') != 'approved' or not review.get('reviewer') or
                        not review.get('evidence') or review.get('payload_sha256') != canonical_hash(review_payload(row))):
                    raise ValueError('Missing real approved review or changed reviewed content')
                if row.get('split') not in ('train', 'validation'):
                    raise ValueError('Training input only accepts preassigned train/validation rows')
                if not isinstance(row.get('response'), str) or not row['response'].strip():
                    raise ValueError('Empty supervised response')
            rows.append(row)
            if len(rows) > limit:
                raise ValueError('Pilot row limit exceeded; use bounded streaming shards')
    return rows


def assert_no_prompt_overlap(training, heldout):
    def digest(row):
        messages = messages_for(row)
        return canonical_hash([{'role': m['role'], 'content': ' '.join(m['content'].split()).casefold()} for m in messages])
    seen = {digest(row) for row in training}
    if any(digest(row) in seen for row in heldout):
        raise ValueError('Train/heldout prompt overlap')
