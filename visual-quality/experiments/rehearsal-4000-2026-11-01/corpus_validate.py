"""Validate frozen family, image, statistics and donor identities before model calls."""
import json
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from corpus_prepare import HERE,PILOT,sha,read_lines,write


def main(benchmark_path=None):
    manifest=json.loads((HERE/'manifest.json').read_text())
    cases=manifest['cases'];by_id={r['case_id']:r for r in cases}
    benchmark_path=benchmark_path or Path(manifest['inputs'].get('benchmark_manifest_path',str(HERE/'benchmark-manifest.json')))
    assert sha(benchmark_path)==manifest['inputs']['benchmark_manifest_sha256']
    benchmark=json.loads(benchmark_path.read_text())['cases']
    assert len(cases)==len(by_id)==len({r['image_id'] for r in cases})==len({r['view_sha256'] for r in cases})==4000
    assert Counter(r['benchmark_partition'] for r in cases)=={'rehearsal':3500,'development':100,'representative':300,'challenge':100}
    groups=defaultdict(list)
    for row in cases:groups[row['family_id']].append(row)
    assert len(groups)==3700 and Counter(len(rows) for rows in groups.values())=={1:3600,4:100}
    for rows in groups.values():
        assert len({r['benchmark_partition'] for r in rows})==1
        if len(rows)==4:
            assert {r['role'] for r in rows}=={'source_parent','variant'}
            assert len({r['target_edge'] for r in rows})==1
            assert len({(r['width'],r['height']) for r in rows})==1
    benchmark_families={r['family_id'] for r in benchmark};assert len(benchmark_families)==500
    for row in benchmark:
        assert all(by_id[row['case_id']][k]==row[k] for k in('view_sha256','image_path','family_id','benchmark_partition'))
        assert len(groups[row['family_id']])==1 and row['historically_fresh']
    protected=read_lines(PILOT/'development.jsonl')+read_lines(PILOT/'test.jsonl')
    protected_families={r['split_group'] for r in protected};protected_hashes={r['sha256'] for r in protected}
    assert not set(groups)&protected_families
    assert not {r['source_sha256'] for r in cases}&protected_hashes
    assert all(not any(r['task_masks'].values()) for r in cases)
    statistics={r['case_id']:r for r in read_lines(HERE/'stats.jsonl')}
    assert set(statistics)==set(by_id)
    for case_id,row in by_id.items():
        raw=statistics[case_id]
        assert raw['sha256']==row['view_sha256'] and raw['source_sha256']==row['source_sha256']
        assert raw['analysis']['max_edge']==1536
    provenance=json.loads((HERE/'stats-provenance.json').read_text())
    calibration=set(provenance['reference_family_ids'])
    assert len(calibration)==3200 and not calibration&benchmark_families and calibration==set(groups)-benchmark_families
    donors=json.loads((HERE/'stats-donors.json').read_text())['donors']
    assert set(donors)=={r['case_id'] for r in benchmark}
    assert len(set(donors.values()))==500
    for case_id,donor_id in donors.items():
        row=by_id[donor_id]
        assert row['role']!='variant' and row['benchmark_partition']=='rehearsal'
        assert row['family_id']!=by_id[case_id]['family_id'] and row['family_id'] in calibration
    prompts=read_lines(HERE/'stats-prompts.jsonl')
    assert {r['case_id'] for r in prompts}==set(by_id) and len(prompts)==4000
    assert all('Frame area close to peak detail' not in r['text'] for r in prompts)
    percentiles=read_lines(HERE/'stats-percentiles.jsonl');reference_sha=sha(HERE/'stats-reference.json')
    assert len(percentiles)==4000 and {r['case_id'] for r in percentiles}==set(by_id)
    for row in percentiles:
        assert row['reference_sha256']==reference_sha and row['view_sha256']==by_id[row['case_id']]['view_sha256']
    review_path=Path(manifest['inputs'].get('cross_boundary_review_path',str(HERE/'cross-boundary-review.json')))
    assert sha(review_path)==manifest['inputs']['cross_boundary_review_sha256']
    review=json.loads(review_path.read_text())
    assert review['status']=='reviewed' and not review['benchmark_replacements']
    excluded=set(review['excluded_candidates'])
    revisions_path=HERE/'benchmark-revisions.json'
    if revisions_path.exists():
        revisions=json.loads(revisions_path.read_text())
        assert revisions['active_manifest']['sha256']==sha(benchmark_path)
        assert sha(revisions_path)==manifest['inputs']['benchmark_revisions_sha256']
        excluded.update(candidate for revision in revisions['revisions'] for candidate in revision['excluded_from_final_candidates'])
    assert not {r.get('candidate_id') or r.get('parent_candidate_id') for r in cases}&excluded
    artifacts=['manifest.json',benchmark_path.name,'stats.jsonl','stats-reference.json','stats-prompts.jsonl',
               'stats-donors.json','stats-provenance.json','qc-reference.json','stats-percentiles.jsonl',
               review_path.name,'candidate-embeddings.npz']
    if revisions_path.exists():artifacts.append(revisions_path.name)
    write(HERE/'corpus-audit.json',{
        'status':'passed','images':4000,'families':3700,'benchmark_families':500,'calibration_families':3200,
        'unique_background_donors':500,'controlled_families':100,'all_training_masks_false':True,
        'artifact_sha256':{name:sha(HERE/name) for name in artifacts},
        'protected_original_partitions_sha256':{name:sha(PILOT/name) for name in('development.jsonl','test.jsonl')},
        'script_sha256':sha(Path(__file__))})
    print('Corpus integrity audit passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-manifest',type=Path)
    args=parser.parse_args();main(args.benchmark_manifest)
