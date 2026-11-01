"""Blind visual-review packets for the 100 controlled degradation families."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

HERE = Path(__file__).resolve().parent
DATA = Path('/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01')
CATEGORIES = {'clear_degradation', 'not_clearly_worse', 'unjudgeable'}
INSTRUCTIONS = """Inspect every family and every derivative. Compare the full frames and identical-coordinate native crops; open exact individual delivered images when the previews do not settle the comparison. Judge visible technical degradation, not aesthetic preference. The generated order does not guarantee worsening: a transform can be imperceptible, remove distracting noise, or affect detail already absent from the parent.
For each derivative, independently classify it versus the parent AND versus the preceding level as clear_degradation, not_clearly_worse, or unjudgeable. V1's preceding level is the parent, so those two entries must agree. Give brief visible evidence for each decision. Use not_clearly_worse when the available views do not establish a clear loss, including similar or improved appearance; use unjudgeable when the relevant comparison cannot be assessed reliably. Do not infer a verdict from generation order or presumed parameters. Do not discard families or assign numeric reference scores.
These packets contain no grader scores, reference scores or image-stat text. Transform parameters are held separately in provenance.json; do not inspect that file or scoring outputs before saving this review. The full-family diagnostic remains primary; verified comparisons and fully verified adjacent-level ladders are additional subsets, reported separately."""


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def parent_crop_box(parent, edge=384):
    """Select one detailed region using only the parent's pixels, never variants."""
    width, height = parent.size
    side = min(edge, width, height)
    if side < 2:
        raise ValueError('Detail crops require at least two pixels per dimension')
    luma = np.asarray(parent.convert('L'), dtype=np.float32)
    candidates = []
    for y in np.unique(np.linspace(0, height - side, 5).round().astype(int)):
        for x in np.unique(np.linspace(0, width - side, 5).round().astype(int)):
            patch = luma[y:y + side, x:x + side]
            detail = float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())
            centrality = abs(x + side / 2 - width / 2) + abs(y + side / 2 - height / 2)
            candidates.append((-detail, centrality, int(y), int(x)))
    _, _, y, x = min(candidates)
    return (x, y, x + side, y + side)


def identity(row):
    # Explicit allowlist: grader, reference, quality and statistics fields cannot leak.
    return {key: row[key] for key in ('case_id', 'image_id', 'image_path', 'view_sha256', 'source_path', 'source_sha256')}


def render_family(rows, number, assets):
    parent = next(row for row in rows if row['role'] == 'source_parent')
    variants = sorted((row for row in rows if row['role'] == 'variant'), key=lambda row: row['severity'])
    if len(rows) != 4 or [row['severity'] for row in variants] != [1, 2, 3]:
        raise ValueError('Expected one parent and three ordered derivatives')
    ordered = [parent, *variants]
    images = []
    for row in ordered:
        if sha(row['image_path']) != row['view_sha256'] or sha(row['source_path']) != row['source_sha256']:
            raise ValueError('Controlled-family source or delivered bytes changed')
        with Image.open(row['image_path']) as opened:
            images.append(opened.convert('RGB'))
    if len({image.size for image in images}) != 1:
        raise ValueError('All family members must have identical delivered dimensions')
    box = parent_crop_box(images[0])
    assets.mkdir(parents=True, exist_ok=True)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default(size=18)
    sheet = Image.new('RGB', (1760, 920), (241, 244, 247))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f'Family F{number:03d} | full-frame previews and identical native-pixel crops | no scores', font=font, fill='black')
    members = []
    for column, (row, image) in enumerate(zip(ordered, images)):
        x = column * 440
        label = 'Parent' if column == 0 else f'V{column}'
        draw.text((x + 12, 40), f'{label} | {row["case_id"]}', font=font, fill='black')
        preview = image.copy()
        preview.thumbnail((416, 330), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + (440 - preview.width) // 2, 74 + (330 - preview.height) // 2))
        draw.text((x + 12, 416), f'Exact crop: {box}', font=font, fill='black')
        crop = image.crop(box)
        crop_path = assets / f'F{number:03d}-{label.lower()}-crop.png'
        crop.save(crop_path, format='PNG', compress_level=6)
        sheet.paste(crop, (x + (440 - crop.width) // 2, 450 + (384 - crop.height) // 2))
        draw.text((x + 12, 846), f'View SHA256: {row["view_sha256"][:16]}…', font=font, fill='black')
        draw.text((x + 12, 875), f'Source SHA256: {row["source_sha256"][:16]}…', font=font, fill='black')
        members.append({**identity(row), 'label': label, 'crop_path': str(crop_path.resolve()),
                        'crop_sha256': sha(crop_path), 'crop_box_xyxy': list(box),
                        'crop_pixel_scale': 1, 'delivered_dimensions': list(image.size)})
    sheet_path = assets / f'F{number:03d}-contact-sheet.png'
    sheet.save(sheet_path, format='PNG', compress_level=6)
    return {'review_family_id': f'F{number:03d}', 'family_id': parent['family_id'],
            'contact_sheet_path': str(sheet_path.resolve()), 'contact_sheet_sha256': sha(sheet_path),
            'members': members}


def review_template(family):
    parent, *variants = family['members']
    comparisons = []
    previous = parent
    for member in variants:
        comparisons.append({'case_id': member['case_id'], 'view_sha256': member['view_sha256'],
            'versus_parent': {'comparison_case_id': parent['case_id'], 'classification': None, 'evidence': ''},
            'versus_previous': {'comparison_case_id': previous['case_id'], 'classification': None, 'evidence': ''}})
        previous = member
    return {'review_family_id': family['review_family_id'], 'family_id': family['family_id'],
            'contact_sheet_sha256': family['contact_sheet_sha256'],
            'reviewed_full_frames_and_native_crops': False, 'individual_views_opened': [], 'comparisons': comparisons}


def generate(manifest, output, assets):
    if output.exists() or assets.exists():
        raise ValueError('Refusing to overwrite review packets or assets; use a new destination')
    cases = json.loads(manifest.read_text())['cases']
    if len(cases) != 4000:
        raise ValueError('A final 4000-image manifest is required')
    grouped = defaultdict(list)
    for row in cases:
        if row.get('role') in {'source_parent', 'variant'}:
            if row['benchmark_partition'] != 'rehearsal':
                raise ValueError('Controlled diagnostics must remain outside the benchmark')
            grouped[row['family_id']].append(row)
    if len(grouped) != 100:
        raise ValueError('Expected exactly 100 controlled families')
    families = [render_family(grouped[name], number, assets) for number, name in enumerate(sorted(grouped), 1)]
    provenance = {'manifest_sha256': sha(manifest), 'generator_sha256': sha(__file__),
        'software': {'numpy': np.__version__, 'Pillow': PILLOW_VERSION},
        'crop_policy': 'Parent only; highest mean absolute luma difference on a fixed 5x5 window grid, centrality breaks ties; edge min(384,width,height)',
        'crop_scaling': 'Native pixels, no crop resampling or enhancement; coordinates shared across all four members',
        'families': {name: [{'case_id': row['case_id'], 'source_sha256': row['source_sha256'],
            'view_sha256': row['view_sha256'], 'transform': row.get('transform'),
            'family_preprocessing': row['family_preprocessing']} for row in rows] for name, rows in grouped.items()}}
    write_new(output / 'provenance.json', provenance)
    index = []
    for start in range(0, 100, 10):
        packet_id = f'controlled-{start // 10 + 1:02d}'
        packet = {'packet_id': packet_id, 'manifest_sha256': sha(manifest), 'instructions': INSTRUCTIONS,
                  'families': families[start:start + 10]}
        path = output / f'{packet_id}.json'
        write_new(path, packet)
        write_new(output / f'{packet_id}-review-template.json', {'packet_sha256': sha(path), 'reviewer': '',
            'families': [review_template(family) for family in packet['families']]})
        index.append({'packet_id': packet_id, 'path': str(path.resolve()), 'sha256': sha(path)})
    write_new(output / 'index.json', {'manifest_sha256': sha(manifest), 'packets': index,
              'families': 100, 'derivatives': 300, 'comparison_entries': 600,
              'unique_image_pairs': 500, 'numeric_reference_scores': False})
    print(f'Created 10 packets covering 100 families and 600 comparisons: {output}', flush=True)


def validate(packet_path, reviews_path):
    packet = json.loads(packet_path.read_text())
    reviews = json.loads(reviews_path.read_text())
    if reviews['packet_sha256'] != sha(packet_path) or not reviews['reviewer'].strip():
        raise ValueError('Missing reviewer or mismatched packet identity')
    expected = {row['family_id']: review_template(row) for row in packet['families']}
    for family in packet['families']:
        if sha(family['contact_sheet_path']) != family['contact_sheet_sha256']:
            raise ValueError('Contact sheet changed after packet creation')
        for member in family['members']:
            if sha(member['image_path']) != member['view_sha256'] or sha(member['crop_path']) != member['crop_sha256']:
                raise ValueError('Exact view or native crop changed after packet creation')
    if len(reviews['families']) != 10 or {row['family_id'] for row in reviews['families']} != set(expected):
        raise ValueError('Review must cover every family exactly once')
    for family in reviews['families']:
        template = expected[family['family_id']]
        if any(family[key] != template[key] for key in ('review_family_id', 'contact_sheet_sha256')):
            raise ValueError('Family identity changed')
        if family['reviewed_full_frames_and_native_crops'] is not True or len(family['comparisons']) != 3:
            raise ValueError('Every family requires full-frame/crop review and all three derivatives')
        valid_ids = {row['case_id'] for row in template['comparisons']} | {template['comparisons'][0]['versus_parent']['comparison_case_id']}
        opened = family['individual_views_opened']
        if not isinstance(opened, list) or len(opened) != len(set(opened)) or not set(opened) <= valid_ids:
            raise ValueError('Individual-view inspection list must contain unique family case IDs')
        for actual, wanted in zip(family['comparisons'], template['comparisons']):
            if actual['case_id'] != wanted['case_id'] or actual['view_sha256'] != wanted['view_sha256']:
                raise ValueError('Derivative identity or order changed')
            for relation in ('versus_parent', 'versus_previous'):
                item = actual[relation]
                if item['comparison_case_id'] != wanted[relation]['comparison_case_id'] or item['classification'] not in CATEGORIES:
                    raise ValueError('Invalid comparison identity or classification')
                if not 1 <= len(item['evidence'].split()) <= 60:
                    raise ValueError('Each comparison needs 1–60 words of visible evidence')
        first = family['comparisons'][0]
        if first['versus_parent'] != first['versus_previous']:
            raise ValueError('V1 compares with the parent in both entries; judgments must match')
    print(f'Validated {len(reviews["families"])} complete family reviews; no model labels modified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('generate')
    make.add_argument('--manifest', type=Path, default=HERE / 'manifest.json')
    make.add_argument('--output', type=Path, default=HERE / 'controlled-review')
    make.add_argument('--assets', type=Path, default=DATA / 'controlled-review-assets')
    check = sub.add_parser('validate')
    check.add_argument('--packet', type=Path, required=True)
    check.add_argument('--reviews', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args.manifest, args.output, args.assets)
    else:
        validate(args.packet, args.reviews)
