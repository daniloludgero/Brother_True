from __future__ import annotations
import argparse, json
from pathlib import Path
from .graph_rag_hibrido import GraphRAGHibrido

def good(row):
    q = row.get('quality_features') or {}
    return (float(row.get('quality_score_proxy', 0)) >= .75 and
            float(q.get('alignment_proxy', 0)) >= .65 and
            float(q.get('final_supported_by_reasoning', 0)) >= .55 and
            float(q.get('reasoning_structure', 0)) >= 1 and
            float(q.get('completeness', 0)) >= 1 and
            float(q.get('length_ok', 0)) >= 1 and
            all(str(row.get(k, '')).strip() for k in ('prompt','think','final')) and
            '<think>' in row.get('prompt', ''))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', type=Path, required=True); ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--semantic', type=Path, required=True); ap.add_argument('--minilm', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--keep', type=int, default=48)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in a.queue.open(encoding='utf-8') if x.strip()]
    structural = [r for r in rows if good(r)]
    rag = GraphRAGHibrido(a.graph, a.semantic, a.minilm); scored=[]
    try:
        for r in structural:
            hits = rag.search(r['prompt'], top_k=3); top = hits[0] if hits else {}
            x=dict(r); x.update(graphrag_filter=True, graphrag_top_score=float(top.get('score',0)),
                graphrag_top_semantic_score=float(top.get('semantic_score',0)), graphrag_top_record_index=top.get('record_index'),
                graphrag_graph_expansion=bool(top.get('graph_expansion',False))); scored.append(x)
    finally: rag.close()
    scored.sort(key=lambda x:(x['graphrag_top_score'], float(x.get('quality_score_proxy',0))), reverse=True)
    chosen=scored[:min(a.keep,len(scored))]; out=a.out/'filtered_dataset_graphrag.jsonl'
    out.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in chosen)+'\n', encoding='utf-8')
    report={'input_rows':len(rows),'structural_rows':len(structural),'scored_rows':len(scored),'selected_rows':len(chosen),
      'score_min':min((x['graphrag_top_score'] for x in scored),default=None),'score_max':max((x['graphrag_top_score'] for x in scored),default=None),
      'score_selected_min':min((x['graphrag_top_score'] for x in chosen),default=None),'output':str(out)}
    (a.out/'graphrag_filter_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with Path(r'D:\brothertrue\training_memory_journal.jsonl').open('a',encoding='utf-8') as h: h.write(json.dumps({'event':'graphrag_filter_v2',**report},ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
