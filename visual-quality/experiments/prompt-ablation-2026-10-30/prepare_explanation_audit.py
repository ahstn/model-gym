"""Prepare anonymized explanation checks on the preselected twelve images."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data/prompt-ablation-2026-10-30'
TASKS = ('technical_quality', 'aesthetics')

def main():
    selection = json.loads((HERE / 'explanation-audit-selection.json').read_text())
    cases = {c['case_id']: c for c in json.loads((HERE / 'manifest.json').read_text())['cases']}
    observed = {}
    for route in ('deepseek', 'gemini', 'luna', 'haiku'):
        for line in (DATA / route / 'images/observations.jsonl').read_text().splitlines():
            row = json.loads(line)
            observed[(route, row['arm_id'], row['case_id'])] = row
    packets = [[], [], []]
    mapping = {}
    for index, cid in enumerate(selection['case_ids']):
        labels = []
        for route in ('deepseek', 'gemini', 'luna', 'haiku'):
            for arm in ('baseline', 'combined'):
                row = observed[(route, arm, cid)]
                for task in TASKS:
                    alias = hashlib.sha256(f'20261030-evidence-{cid}-{route}-{arm}-{task}'.encode()).hexdigest()[:12]
                    rating = row.get(task)
                    if not rating:
                        continue
                    labels.append({'label_id': alias, 'task': task, 'evidence': rating['evidence'], 'limitations': rating['limitations']})
                    mapping[alias] = {'case_id': cid, 'route': route, 'arm_id': arm, 'task': task}
        packets[index % 3].append({'case_id': cid, 'path': cases[cid]['image_path'], 'view_sha256': cases[cid]['view_sha256'], 'labels': sorted(labels, key=lambda x: x['label_id'])})
    target = HERE / 'explanation-audit'
    target.mkdir(exist_ok=True)
    for index, packet in enumerate(packets, 1):
        (target / f'packet-{index}.json').write_text(json.dumps(packet, indent=2) + '\n')
    (target / 'mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    print(json.dumps({'images': sum(map(len,packets)), 'labels': len(mapping)}))

if __name__ == '__main__':
    main()
