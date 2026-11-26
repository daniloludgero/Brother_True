from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
    report=json.loads(a.report.read_text(encoding='utf-8'))
    entry={'event':'expanded_adapter_comparison','report_path':str(a.report),'experiment':report.get('experiment'),'rows':report.get('rows'),'models':report.get('models'),'production_checkpoint_untouched':True}
    journal=Path(r'D:\brothertrue\training_memory_journal.jsonl')
    with journal.open('a',encoding='utf-8') as h: h.write(json.dumps(entry,ensure_ascii=False)+'\n')
    print(json.dumps(entry,ensure_ascii=False))
if __name__=='__main__': main()
