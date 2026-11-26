from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--report',type=Path,required=True); ap.add_argument('--kind',required=True); a=ap.parse_args(); report=json.loads(a.report.read_text(encoding='utf-8')); entry={'event':'isolated_architecture_experiment','kind':a.kind,'report_path':str(a.report),'experiment':report.get('experiment'),'production_checkpoint_untouched':True,'decision':'not_promoted'}; j=Path(r'D:\brothertrue\training_memory_journal.jsonl');
    with j.open('a',encoding='utf-8') as h: h.write(json.dumps(entry,ensure_ascii=False)+'\n')
    print(json.dumps(entry,ensure_ascii=False))
if __name__=='__main__': main()
