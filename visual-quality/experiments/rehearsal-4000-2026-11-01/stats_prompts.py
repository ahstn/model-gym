"""Freeze label-free percentile text and matched counterfactual donor assignments."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from image_stats import Reference
from image_stats.analysis import Analysis
from image_stats.describe import observations, HEADER
from corpus_prepare import HERE, SEED, sha, write


def key(value):return hashlib.sha256(f'{SEED}:donor:{value}'.encode()).hexdigest()


def text_block(record,reference):
    analysis=Analysis.from_dict(record['analysis'])
    lines=[HEADER.format(source=f", ranked against {reference.count} non-benchmark family representatives in '{reference.name}'")]
    lines.append('Use these measurements for technical-quality checking only; evaluate aesthetics from the image and its photographic intent.')
    lines.extend('- '+o.text for o in observations(analysis,reference) if o.metric!='focus_area')
    # Flags are measured pixels only. Source categories and source labels never enter the prompt.
    from image_stats.flags import FLAG_TEXT
    lines.extend('- Note: '+FLAG_TEXT[f] for f in analysis.flags)
    return '\n'.join(lines)


def donors(cases,stats,reference):
    benchmark=[r for r in cases if r['benchmark_partition']!='rehearsal']
    background=[r for r in cases if r['benchmark_partition']=='rehearsal' and r['role']!='variant']
    peaks={r['case_id']:reference.percentile('sharpness_peak',stats[r['case_id']]['analysis']['metrics']['sharpness_peak']) for r in cases}
    rank=sorted(benchmark,key=lambda r:(peaks[r['case_id']],key(r['case_id'])))
    directions={r['case_id']:(1 if i<250 else -1) for i,r in enumerate(rank)}
    used=set();mapping={};audit=[]
    # Process scarce matching strata first; then deterministic order.
    def size(row):return sum(d['category']==row['category'] and d['source_format']==row['source_format'] for d in background)
    for row in sorted(benchmark,key=lambda r:(size(r),key(r['case_id']))):
        pool=[d for d in background if d['case_id'] not in used and d['family_id']!=row['family_id']]
        strict=[d for d in pool if d['category']==row['category'] and d['source_format']==row['source_format']]
        tier='same_source_category_and_encoding'
        if not strict:
            strict=[d for d in pool if d['source_format']==row['source_format']]
            tier='same_encoding_source_relaxed'
        if not strict:
            strict=[d for d in pool if d['category']==row['category']]
            tier='same_source_category_encoding_relaxed'
        if not strict:strict=pool;tier='source_and_encoding_relaxed'
        direction=directions[row['case_id']]
        def features(d):
            dw=abs(math.log(d['width']/row['width']));dh=abs(math.log(d['height']/row['height']))
            signed=(peaks[d['case_id']]-peaks[row['case_id']])*direction
            return dw+dh,signed
        close=[d for d in strict if features(d)[0]<=.8]
        if not close:close=strict;tier+=';dimensions_relaxed'
        contradictory=[d for d in close if features(d)[1]>=25]
        if not contradictory:contradictory=[d for d in close if features(d)[1]>0]
        if not contradictory:contradictory=close
        chosen=min(contradictory,key=lambda d:(features(d)[0],key(row['case_id']+d['case_id'])))
        mapping[row['case_id']]=chosen['case_id'];used.add(chosen['case_id'])
        own=stats[row['case_id']]['analysis']['metrics'];other=stats[chosen['case_id']]['analysis']['metrics']
        audit.append({'case_id':row['case_id'],'donor_case_id':chosen['case_id'],'matching_tier':tier,
                      'intended_sharpness_direction':direction,'actual_sharpness_direction':int(np.sign(peaks[chosen['case_id']]-peaks[row['case_id']])),
                      'own_sharpness_percentile':peaks[row['case_id']],'donor_sharpness_percentile':peaks[chosen['case_id']],
                      'dimension_log_distance':features(chosen)[0],'own_source_format':row['source_format'],'donor_source_format':chosen['source_format'],
                      'own_category':row['category'],'donor_category':chosen['category'],
                      'raw_metric_deltas':{m:other[m]-v for m,v in own.items()}})
    assert len(mapping)==500 and len(set(mapping.values()))==500
    assert all(d not in mapping for d in mapping.values())
    return {'schema_version':1,'seed':SEED,'donors':mapping,
            'policy':'Unique non-benchmark, non-variant family representatives from the same4000 corpus; no labels; match source/encoding/dimensions then balance contradictory sharpness directions',
            'matching_audit':audit,'counts':{'targets':500,'unique_donors':500,'actual_directions':dict(Counter(a['actual_sharpness_direction'] for a in audit))}}


def main():
    manifest=HERE/'manifest.json';cases=json.loads(manifest.read_text())['cases']
    stats_path=HERE/'stats.jsonl';stats={r['case_id']:r for r in map(json.loads,stats_path.read_text().splitlines())}
    assert set(stats)=={r['case_id'] for r in cases}
    benchmark_groups={r['family_id'] for r in cases if r['benchmark_partition']!='rehearsal'}
    calibration=[r for r in cases if r['family_id'] not in benchmark_groups and r['role']!='variant']
    assert len(calibration)==3200 and len({r['family_id'] for r in calibration})==3200
    reference=Reference.build([stats[r['case_id']]['analysis']['metrics'] for r in calibration],name='rehearsal4000-background-families',max_edge=1536)
    reference.save(HERE/'stats-reference.json')
    reference_sha=sha(HERE/'stats-reference.json')
    write(HERE/'qc-reference.json',{
        'schema_version':1,'calibration_family_ids':sorted(r['family_id'] for r in calibration),
        'sharpness_peak_p25':reference.quantiles['sharpness_peak'][25],
        'noise_sigma_p90':reference.quantiles['noise_sigma'][90],
        'reference_sha256':reference_sha,
        'percentile_policy':'Use Reference.percentile records for tied thresholds; raw quantiles are descriptive only',
    })
    with (HERE/'stats-percentiles.jsonl').open('w') as stream:
        for row in cases:
            metrics=stats[row['case_id']]['analysis']['metrics']
            stream.write(json.dumps({'case_id':row['case_id'],'view_sha256':row['view_sha256'],
                'reference_sha256':reference_sha,
                'sharpness_peak_percentile':reference.percentile('sharpness_peak',metrics['sharpness_peak']),
                'noise_sigma_percentile':reference.percentile('noise_sigma',metrics['noise_sigma'])})+'\n')
    with (HERE/'stats-prompts.jsonl').open('w') as stream:
        for row in cases:stream.write(json.dumps({'case_id':row['case_id'],'text':text_block(stats[row['case_id']],reference)},ensure_ascii=False)+'\n')
    write(HERE/'stats-donors.json',donors(cases,stats,reference))
    write(HERE/'stats-provenance.json',{'manifest_sha256':sha(manifest),'stats_sha256':sha(stats_path),
          'reference_sha256':sha(HERE/'stats-reference.json'),'prompt_sha256':sha(HERE/'stats-prompts.jsonl'),
          'donors_sha256':sha(HERE/'stats-donors.json'),'script_sha256':sha(Path(__file__)),
          'qc_reference_sha256':sha(HERE/'qc-reference.json'),'stats_percentiles_sha256':sha(HERE/'stats-percentiles.jsonl'),
          'reference_family_ids':sorted(r['family_id'] for r in calibration),'reference_count':len(calibration),
          'focus_area':'Excluded from text, retained in raw statistics','gate':'Shadow-review flag only; no labels replaced or model calls skipped',
          'image_stats_source_hashes':{str(p):sha(p) for p in (HERE.parents[2]/'image-stats/src/image_stats').glob('*.py')}})
    print('Frozen3200-family statsReference,4000promptblocks,500matched donors',flush=True)


if __name__=='__main__':main()
