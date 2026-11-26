from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--report', type=Path, required=True); ap.add_argument('--adapter', type=Path, required=True)
    a = ap.parse_args(); report = json.loads(a.report.read_text(encoding='utf-8'))
    entry = {'event':'generation_validation', 'report_path':str(a.report), 'adapter_path':str(a.adapter),
             'experiment':report.get('experiment'), 'models':report.get('models'), 'production_checkpoint_untouched':True}
    journal=Path(r'D:\brothertrue\training_memory_journal.jsonl'); journal.parent.mkdir(parents=True,exist_ok=True)
    with journal.open('a',encoding='utf-8') as h: h.write(json.dumps(entry,ensure_ascii=False)+'\n')
    print(json.dumps(entry,ensure_ascii=False))
if __name__=='__main__': main()
