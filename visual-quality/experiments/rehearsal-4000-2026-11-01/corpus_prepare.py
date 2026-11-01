"""Prepare a fixed candidate pool from acquired training-only source families."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from visual_quality.views import _policy, _render

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/home/ahstn/git/model-gym/visual-quality')
DATA = ORIGINAL / 'data'
OUTPUT = DATA / HERE.name
PILOT = DATA / 'pilot-10k-v1'
SEED = 202611014000
LIMITS = {'megalith': 4500, 'vizwiz': 800, 'controls': 200}
BENCH_POOL = {'megalith': 800, 'vizwiz': 160, 'controls': 40}
OPERATORS = ('gaussian_blur', 'gaussian_noise', 'jpeg_compression')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(kind: str, identity: str) -> str:
    return hashlib.sha256(f'{SEED}:{kind}:{identity}'.encode()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def category(row: dict) -> str:
    return 'controls' if row['component'] in ('clevr', 'plotqa') else row['component']


def viz_stratum(row: dict) -> str | None:
    if row['component'] != 'vizwiz':
        return None
    votes = row['source_labels']['human_flaw_votes']
    return min(('BLR', 'BRT', 'DRK', 'FRM', 'OBS', 'ROT', 'OTH'), key=lambda x: (-votes[x], x))


def history(rows: list[dict]) -> tuple[set[str], set[int], dict]:
    previous = ORIGINAL / 'experiments/dual-high-2000-2026-10-31/manifest.json'
    old = json.loads(previous.read_text())
    groups = set(old['excluded_prior']['family_ids'])
    groups.update(r['family_id'] for r in old['cases'])
    # The EmbeddingGemma feasibility review and all newer comparisons reuse these families.
    paths = [previous]
    for p in (ORIGINAL / 'experiments').glob('*/manifest.json'):
        if p == previous:
            continue
        obj = json.loads(p.read_text())
        for r in obj.get('cases', []):
            if isinstance(r, dict):
                g = r.get('family_id', r.get('split_group'))
                if g:
                    groups.add(g)
        paths.append(p)
    hashes = {int(r['dhash'], 16) for r in rows if r['split_group'] in groups and r.get('dhash')}
    hashes.update(int(r['dhash'], 16) for r in old['cases'] if r.get('dhash'))
    return groups, hashes, {'inputs': {str(p): sha(p) for p in sorted(paths)},
                            'historical_family_count': len(groups),
                            'historical_family_ids': sorted(groups),
                            'history_definition': 'Prior frozen exposure-family audit plus later experiment manifests; existing 72 and embedding384 are subsets of the prior2000.'}


def metadata(row: dict) -> dict:
    path = DATA / row['image_path']
    with Image.open(path) as im:
        return {'format': im.format, 'frames': getattr(im, 'n_frames', 1),
                'icc': bool(im.info.get('icc_profile')),
                'alpha': 'A' in im.getbands() or 'transparency' in im.info}


def render(row: dict) -> dict:
    source = DATA / row['image_path']
    cache = OUTPUT / 'prepared' / (hashlib.sha256(row['id'].encode()).hexdigest() + '.json')
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get('source_sha256') != row['sha256']:
            raise ValueError('Source identity changed')
        if saved.get('error') and 'PNG exceeds common request-size cap' not in saved['error']:
            return saved
        if not saved.get('error') and Path(saved['image_path']).exists() and sha(Path(saved['image_path'])) == saved['view_sha256']:
            return {**saved, 'target_edge': saved.get('target_edge', 1536)}
    result = {'image_id': row['id'], 'family_id': row['split_group'], 'source_sha256': row['sha256']}
    try:
        if sha(source) != row['sha256']:
            raise ValueError('Source bytes differ from frozen pilot manifest')
        payload = source.read_bytes()
        edge = 1536
        while True:
            data, prep = _render(payload, edge)
            if len(data) <= 2650000:
                break
            edge = edge * 9 // 10
            if edge < 256:
                raise ValueError('Cannot satisfy common model limits')
        width, height = prep['output_dimensions']
        if min(width, height) < 256 or width * height < 78400 or width * height > 2458624:
            raise ValueError('Outside common native model dimension/area limits')
        digest = hashlib.sha256(data).hexdigest()
        image = OUTPUT / 'views' / f'{digest}.png'
        if not image.exists():
            image.write_bytes(data)
        source_metadata = metadata(row)
        result.update(image_path=str(image), source_path=str(source), view_sha256=digest,
                      delivered_bytes=len(data), width=width, height=height,
                      preprocessing=prep, source_format=source_metadata['format'],
                      source_metadata=source_metadata, target_edge=edge, mime_type='image/png')
    except (ValueError, OSError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    write(cache, result)
    return result


def clean_record(row: dict, prepared: dict, fresh: bool, group: str, parents: dict) -> dict:
    return {**prepared, 'component': row['component'], 'category': group,
            'source': row['source'], 'source_id': row['source_id'],
            'source_revision': row['source_revision'], 'source_url': row.get('source_url'),
            'license': row['license'], 'decoded_sha256': row['decoded_sha256'],
            'dhash': row.get('dhash'), 'split': 'train', 'split_group': row['split_group'],
            'source_group': row.get('source_group'), 'historically_fresh': fresh,
            'controlled_parent': row['id'] in parents,
            'operator': parents[row['id']][0]['transform']['operator'] if row['id'] in parents else None,
            'severity': None, 'vizwiz_stratum': viz_stratum(row),
            'known_nonphoto_source': row['component'] in ('clevr', 'plotqa'),
            'benchmark_partition': 'rehearsal',
            'task_masks': {'technical_quality': False, 'aesthetics': False, 'photo_domain': False}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'views').mkdir(exist_ok=True)
    train = read_lines(PILOT / 'train.jsonl')
    held = read_lines(PILOT / 'development.jsonl') + read_lines(PILOT / 'test.jsonl')
    historical, historical_hashes, audit = history(train)
    protected = {r['split_group'] for r in held}
    protected_source_groups = {r.get('source_group') for r in held} - {None}
    protected_bytes = {r['sha256'] for r in held}
    protected_pixels = {r['decoded_sha256'] for r in held}
    protected_hashes = {int(r['dhash'], 16) for r in held if r.get('dhash')}
    parents = defaultdict(list)
    for r in train:
        if r['component'] == 'distortions':
            parents[r['parent_id']].append(r)
    candidates, seen_families, seen_hashes, seen_pixels, rejection = [], set(), set(), set(), Counter()
    for row in sorted(train, key=lambda r: key('candidate', r['id'])):
        if category(row) not in LIMITS:
            continue
        if row['split_group'] in protected or row.get('source_group') in protected_source_groups or row['sha256'] in protected_bytes or row['decoded_sha256'] in protected_pixels:
            rejection['protected_identity'] += 1
            continue
        d = int(row['dhash'], 16)
        if any((d ^ other).bit_count() <= 2 for other in protected_hashes):
            rejection['protected_dhash_distance_at_most_2'] += 1
            continue
        if row['split_group'] in seen_families or row['sha256'] in seen_hashes or row['decoded_sha256'] in seen_pixels:
            rejection['duplicate_family_or_exact_pixels'] += 1
            continue
        seen_families.add(row['split_group']); seen_hashes.add(row['sha256']); seen_pixels.add(row['decoded_sha256'])
        fresh = row['split_group'] not in historical and not any((d ^ other).bit_count() <= 2 for other in historical_hashes)
        candidates.append((row, fresh))
    # Ordering ensures the fresh benchmark candidates are prepared first, before other available images.
    candidates.sort(key=lambda x: (not x[1] or x[0]['id'] in parents, key('render', x[0]['id'])))
    good = []
    with ThreadPoolExecutor(args.workers) as pool:
        for index, ((row, fresh), prepared) in enumerate(zip(candidates, pool.map(render, (r for r, _ in candidates))), 1):
            if 'error' in prepared:
                rejection[prepared['error']] += 1
            else:
                good.append(clean_record(row, prepared, fresh, category(row), parents))
            if index % 250 == 0:
                print(f'Prepared {index}/{len(candidates)}; valid={len(good)}', flush=True)
    capacity = Counter(r['category'] for r in good)
    limits = {group: min(quota, capacity[group]) for group, quota in LIMITS.items()}
    target = min(5500, sum(capacity.values()))
    for group in ('vizwiz', 'controls', 'megalith'):
        extra = min(capacity[group] - limits[group], target - sum(limits.values()))
        limits[group] += extra
    if target < 4300 or capacity['megalith'] < 3000 or capacity['vizwiz'] < 600 or capacity['controls'] < 100:
        raise ValueError(f'Insufficient candidate capacity after rendering: {dict(capacity)}')
    selected, priority, used = [], [], set()
    for group, quota in BENCH_POOL.items():
        pool = [r for r in good if r['category'] == group and r['historically_fresh'] and not r['controlled_parent']]
        pool.sort(key=lambda r: key('benchmark-pool', r['image_id']))
        if len(pool) < quota:
            raise ValueError(f'Insufficient fresh independent {group}: {len(pool)} < {quota}')
        priority.extend(pool[:quota])
    for r in priority:
        r['benchmark_candidate'] = True
        used.add(r['image_id'])
    selected.extend(sorted(priority, key=lambda r: key('embedding-priority', r['image_id'])))
    for group, quota in limits.items():
        pool = sorted([r for r in good if r['category'] == group and r['image_id'] not in used], key=lambda r: key('candidate-pool', r['image_id']))
        need = quota - sum(r['category'] == group for r in selected)
        if len(pool) < need:
            raise ValueError(f'Insufficient valid {group}: {len(pool)+quota-need} < {quota}')
        for r in pool[:need]:
            r['benchmark_candidate'] = False
            selected.append(r); used.add(r['image_id'])
    for n, row in enumerate(selected, 1):
        row['candidate_id'] = f'C{n:04d}'
        row['case_id'] = row['candidate_id']
    assert len(selected) == target and sum(r['benchmark_candidate'] for r in selected) == 1000
    audit.update(candidate_filter_rejections=dict(rejection), available_valid=Counter(r['category'] for r in good),
                 fresh_available=Counter(r['category'] for r in good if r['historically_fresh'] and not r['controlled_parent']),
                 protected_families=len(protected), original_training_rows=len(train),
                 pilot_inputs={str(PILOT/f'{s}.jsonl'): sha(PILOT/f'{s}.jsonl') for s in ('train','development','test')})
    write(HERE / 'candidate-preparation-audit.json', audit)
    write(HERE / 'candidate-manifest.json', {'schema_version': 1, 'seed': SEED, 'cases': selected,
          'selection_policy': {'candidate_quotas': limits, 'requested_candidate_quotas': LIMITS, 'benchmark_candidate_quotas': BENCH_POOL,
                               'freshness': 'Exclude all historical scored/reviewed families and dhash<=2 matches for benchmark only',
                               'quality_scores_used': False, 'source_captions_used': False},
          'preprocessing_policy': {**{k: v for k, v in _policy(1536).items() if k != 'selection'},
             'byte_budget': {'max_png_bytes': 2650000, 'method': 'Start cap1536 then floor(cap*0.9) until PNG fits; rerender original bytes each time',
                            'controlled_family': 'All four members later rerendered at one shared cap before final manifest'}},
          'inputs': {str(Path(__file__)): sha(Path(__file__))}})
    print(json.dumps({'candidate_count': len(selected), 'benchmark_candidates': len(priority), 'audit': str(HERE/'candidate-preparation-audit.json')}), flush=True)


if __name__ == '__main__':
    main()
