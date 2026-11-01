"""Propose and publish an explicit, visually approved single-case benchmark revision."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from corpus_select import HERE, DATA, SEED, vectors, sha

CASE_ID = 'R0045'
REMOVED = {'C0725', 'C5253'}
TAG = 'benchmark-replacement:R0045:v2'
FIELDS = ('candidate_id', 'image_id', 'image_path', 'view_sha256', 'family_id', 'benchmark_partition', 'benchmark_stratum')


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def seeded_key(candidate_id):
    return hashlib.sha256(f'{SEED}:{TAG}:{candidate_id}'.encode()).hexdigest()


def contract(row):
    return {key: row[key] for key in FIELDS}


def proposal():
    original_path = HERE / 'benchmark-manifest.json'
    original = json.loads(original_path.read_text())
    old = next(row for row in original['cases'] if row['case_id'] == CASE_ID)
    assert old['candidate_id'] == 'C0725'
    candidates = json.loads((HERE / 'candidate-manifest.json').read_text())['cases']
    early = json.loads((HERE / 'benchmark-candidate-review.json').read_text())
    cross = json.loads((HERE / 'cross-boundary-review.json').read_text())
    clusters = json.loads((HERE / 'benchmark-clusters.json').read_text())['clusters']
    used = {row['candidate_id'] for row in original['cases']}
    excluded = used | REMOVED | set(early['excluded_candidates']) | set(cross['excluded_candidates'])
    excluded |= set(original['selection_policy']['similarity_guard_excluded_candidates'])
    excluded |= {candidate for group in early['related_groups'] for candidate in group['candidates']}
    eligible = [row for row in candidates if row['benchmark_candidate'] and row['category'] == 'vizwiz'
                and row['historically_fresh'] and not row['controlled_parent'] and row['candidate_id'] not in excluded]
    vec = vectors()
    retained = [row for row in original['cases'] if row['case_id'] != CASE_ID]
    retained_vectors = np.asarray([vec[row['candidate_id']] for row in retained])
    similarity_exclusions = [row['candidate_id'] for row in eligible
                             if float(np.max(retained_vectors @ vec[row['candidate_id']])) >= .985]
    eligible = [row for row in eligible if row['candidate_id'] not in similarity_exclusions]
    ranked = sorted(eligible, key=lambda row: (clusters[row['candidate_id']] != old['content_cluster'], seeded_key(row['candidate_id'])))
    if not ranked:
        raise ValueError('No eligible fresh replacement remains')
    chosen = ranked[0]
    historical_ids = {row['candidate_id'] for row in candidates if not row['historically_fresh']}
    historical_ids |= {row['candidate_id'] for row in early['historical_guards']} | REMOVED
    neighbour_sets = {}
    for name, pool in [('full_candidate_pool', candidates), ('retained_benchmark', retained),
                       ('historically_exposed_candidates', [row for row in candidates if row['candidate_id'] in historical_ids])]:
        pool = [row for row in pool if row['candidate_id'] != chosen['candidate_id']]
        scores = np.asarray([vec[row['candidate_id']] for row in pool]) @ vec[chosen['candidate_id']]
        neighbour_sets[name] = [{'candidate_id': pool[i]['candidate_id'], 'image_path': pool[i]['image_path'],
                                'view_sha256': pool[i]['view_sha256'], 'family_id': pool[i]['family_id'],
                                'cosine': float(scores[i])} for i in np.argsort(-scores)[:12]]
    historical = Path('/home/ahstn/git/model-gym/visual-quality/experiments/embeddinggemma2-clustering-2026-10-31')
    old_cases = {row['case_id']: row for row in json.loads((historical / 'manifest.json').read_text())['cases']}
    archive = np.load(historical / 'embeddings.npz', allow_pickle=False)
    scores = archive['embeddings'] @ vec[chosen['candidate_id']]
    neighbour_sets['historical_384_archive'] = [{
        'candidate_id': archive['case_ids'][i].item(), 'image_path': old_cases[archive['case_ids'][i].item()]['image_path'],
        'view_sha256': old_cases[archive['case_ids'][i].item()]['view_sha256'],
        'family_id': old_cases[archive['case_ids'][i].item()]['family_id'], 'cosine': float(scores[i])}
        for i in np.argsort(-scores)[:12]]
    sheets = []
    for name, rows in neighbour_sets.items():
        page = Image.new('RGB', (1200, 1260), (241, 244, 247))
        draw = ImageDraw.Draw(page)
        for index, row in enumerate(rows):
            x, y = index % 2 * 600, index // 2 * 210
            for offset, member in ((0, chosen), (290, row)):
                assert sha(member['image_path']) == member['view_sha256']
                with Image.open(member['image_path']) as opened:
                    image = opened.convert('RGB'); image.thumbnail((280, 170))
                    page.paste(image, (x + offset + (280 - image.width) // 2, y + 25))
                draw.text((x + offset + 5, y + 5), member['candidate_id'], fill='black')
            draw.text((x + 5, y + 192), f'{name}; cosine={row["cosine"]:.4f}', fill='black')
        path = HERE / 'review' / f'replacement-r0045-{name}.jpg'
        path.parent.mkdir(exist_ok=True); page.save(path, quality=92)
        sheets.append({'path': str(path), 'sha256': sha(path)})
    candidate_ids = sorted(row['candidate_id'] for row in eligible)
    payload = {'case_id': CASE_ID, 'original_manifest_sha256': sha(original_path), 'previous': contract(old),
        'proposed_candidate': chosen, 'proposed_content_cluster': clusters[chosen['candidate_id']],
        'selection': {'seed': SEED, 'tag': TAG, 'preferred_content_cluster': old['content_cluster'],
            'eligible_candidates_sha256': hashlib.sha256(json.dumps(candidate_ids).encode()).hexdigest(),
            'eligible_candidate_ids': candidate_ids, 'ranked_candidate_ids': [row['candidate_id'] for row in ranked],
            'excluded_candidate_ids': sorted(excluded), 'similarity_exclusions': similarity_exclusions,
            'information_policy': 'Original fresh 1000-candidate frame only; source/cluster/hash selection and duplicate guards; no grader or reference labels read'},
        'neighbours': neighbour_sets, 'review_sheets': sheets,
        'inputs': {'candidate_manifest_sha256': sha(HERE / 'candidate-manifest.json'),
                  'early_review_sha256': sha(HERE / 'benchmark-candidate-review.json'),
                  'script_sha256': sha(__file__)}}
    write_new(HERE / 'benchmark-replacement-proposal.json', payload)
    print(json.dumps({'proposed_candidate': chosen['candidate_id'], 'cluster': clusters[chosen['candidate_id']],
                      'eligible_count': len(ranked), 'review_sheets': sheets}), flush=True)


def publish():
    proposal_path = HERE / 'benchmark-replacement-proposal.json'
    proposal = json.loads(proposal_path.read_text())
    review_path = HERE / 'benchmark-replacement-review.json'
    review = json.loads(review_path.read_text())
    accepted = review.get('decision') == 'accepted' or review.get('status') == 'accepted_for_blind_reference_review'
    if not accepted or review['proposal_sha256'] != sha(proposal_path):
        raise ValueError('Replacement requires accepted visual review of this exact proposal')
    chosen = proposal['proposed_candidate']
    if review.get('candidate_id', review.get('replacement_candidate_id')) != chosen['candidate_id']:
        raise ValueError('Visual approval belongs to a different candidate')
    for sheet in proposal['review_sheets']:
        assert sha(sheet['path']) == sheet['sha256'], 'Visually reviewed contact sheet changed'
    evidence_path = HERE / 'cross-boundary-historical-review.json'
    if not evidence_path.exists():
        raise ValueError('Historical duplicate adjudication artifact must exist before publication')
    original_path = HERE / 'benchmark-manifest.json'
    assert sha(original_path) == proposal['original_manifest_sha256']
    original = json.loads(original_path.read_text())
    old = next(row for row in original['cases'] if row['case_id'] == CASE_ID)
    replacement = {**chosen, 'case_id': CASE_ID, 'benchmark_partition': old['benchmark_partition'],
        'benchmark_stratum': old['benchmark_stratum'], 'content_cluster': proposal['proposed_content_cluster'],
        'reviewed_related_group': None}
    revised = {**original, 'cases': [replacement if row['case_id'] == CASE_ID else row for row in original['cases']],
        'revision': {'version': 2, 'supersedes_path': str(original_path), 'supersedes_sha256': sha(original_path),
                     'changed_case_ids': [CASE_ID], 'reason_code': 'confirmed_historical_near_duplicate'},
        'inputs': {**original['inputs'], 'replacement_proposal_sha256': sha(proposal_path),
                   'replacement_visual_review_sha256': sha(review_path), 'revision_script_sha256': sha(__file__)}}
    assert len({row['family_id'] for row in revised['cases']}) == 500
    new_path = HERE / 'benchmark-manifest-v2.json'
    write_new(new_path, revised)
    all_cases = json.loads((HERE / 'candidate-manifest.json').read_text())['cases']
    historical_peer = next(row for row in all_cases if row['candidate_id'] == 'C5253')
    revision = {'schema_version': 1,
        'original_manifest': {'path': str(original_path), 'sha256': sha(original_path)},
        'active_manifest': {'path': str(new_path), 'sha256': sha(new_path)},
        'revisions': [{'case_id': CASE_ID, 'previous': contract(old), 'replacement': contract(replacement),
            'reason_code': 'confirmed_historical_near_duplicate',
            'reason': 'Same blue carton, checkered cushion, cap and printed features as historically exposed C5253, with modest camera shift',
            'historical_peer': {key: historical_peer[key] for key in ('candidate_id', 'family_id', 'view_sha256')},
            'historical_adjudication': {'path': str(evidence_path), 'sha256': sha(evidence_path)},
            'selection': proposal['selection'],
            'proposal_artifact': {'path': str(proposal_path), 'sha256': sha(proposal_path),
                'producer_source_path': str(HERE / 'pipeline-snapshots/benchmark_revision-proposal.py'),
                'producer_source_sha256': proposal['inputs']['script_sha256']},
            'visual_review': {'path': str(review_path), 'artifact_sha256': sha(review_path), 'decision': 'accepted'},
            'excluded_from_final_candidates': sorted(REMOVED)}]}
    write_new(HERE / 'benchmark-revisions.json', revision)
    print(json.dumps({'active_manifest': str(new_path), 'sha256': sha(new_path), 'replacement': contract(replacement)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['propose', 'publish'])
    args = parser.parse_args()
    proposal() if args.stage == 'propose' else publish()
