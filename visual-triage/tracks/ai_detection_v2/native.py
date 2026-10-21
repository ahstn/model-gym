"""Read pinned parquet metadata and selected native image row groups by byte range."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests


def chunk_bounds(column):
    start = column.dictionary_page_offset or column.data_page_offset
    return start, start + column.total_compressed_size - 1


def read_range(url, start, end):
    for _ in range(3):
        response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        if response.status_code == 206:
            if (
                len(response.content) != end - start + 1
                or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}"
            ):
                raise ValueError("Content range mismatch")
            return response.content
        if response.status_code == 200:
            raise ValueError("Server ignored byte range; full download stopped")
        response.raise_for_status()
    raise ValueError("Range response required")


def metadata(job):
    builder, kind, split, number, count = job
    repo, revision, _ = builder.SOURCES[kind]
    filename = f"data/{split}-{number:05d}-of-{count:05d}.parquet"
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    base = builder.CACHE / f"native-{kind}-{split}-{number:05d}"
    output = base.with_suffix(".json")
    if output.exists():
        return json.loads(output.read_text())["rows"]
    response = requests.get(url, headers={"Range": "bytes=-65536"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206:
        raise ValueError("Footer range required")
    size = int(response.headers["Content-Range"].split("/")[-1])
    sparse = base.with_suffix(".parquet")
    with sparse.open("wb") as file:
        file.write(b"PAR1")
        file.seek(size - len(response.content))
        file.write(response.content)
    parquet = pq.ParquetFile(sparse)
    rows = []
    group_records = []
    for group_index in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(group_index)
        columns = [group.column(i) for i in range(1, group.num_columns)]
        start = min(chunk_bounds(c)[0] for c in columns)
        end = max(chunk_bounds(c)[1] for c in columns)
        chunk = read_range(url, start, end)
        with sparse.open("r+b") as file:
            file.seek(start)
            file.write(chunk)
        records = parquet.read_row_group(
            group_index, columns=["label", "image_id", "ds_name", "source"]
        ).to_pylist()
        image_start, image_end = chunk_bounds(group.column(0))
        group_records.append(
            {
                "index": group_index,
                "start": image_start,
                "end": image_end,
                "size": image_end - image_start + 1,
            }
        )
        for local_index, row in enumerate(records):
            rows.append(
                {
                    "row": row,
                    "kind": kind,
                    "native_split": split,
                    "native_file": str(sparse),
                    "native_url": url,
                    "native_group": group_index,
                    "native_local_index": local_index,
                    "native_image_start": image_start,
                    "native_image_end": image_end,
                    "native_image_size": image_end - image_start + 1,
                    "row_idx": len(rows),
                }
            )
    output.write_text(
        json.dumps({"url": url, "size": size, "row_groups": group_records, "rows": rows}) + "\n"
    )
    return rows


def catalogue(builder):
    jobs = (
        [(builder, "clic", "train", n, 20) for n in range(20)]
        + [(builder, "clic", "test", n, 6) for n in range(6)]
        + [(builder, "buster", "test", n, 9) for n in range(9)]
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [row for batch in pool.map(metadata, jobs) for row in batch]
    groups = defaultdict(dict)
    for item in rows:
        key = (item["kind"], item["native_split"], item["row"]["image_id"])
        groups[key][item["row"]["source"]] = item
    return groups


def acquire_many(builder, items):
    grouped = defaultdict(list)
    for item in items:
        grouped[(item["native_file"], item["native_group"])].append(item)
    total = sum(batch[0]["native_image_size"] for batch in grouped.values())
    completed = {}
    for metadata_path in builder.CACHE.glob("native-*.json"):
        info = json.loads(metadata_path.read_text())
        for group in info["row_groups"]:
            mark = metadata_path.with_suffix(f".group-{group['index']}.complete")
            if mark.exists():
                completed[(str(metadata_path.with_suffix(".parquet")), group["index"])] = group["size"]
    cumulative = sum(completed.values()) + sum(
        batch[0]["native_image_size"] for key, batch in grouped.items() if key not in completed
    )
    if max(total, cumulative) > 3_000_000_000:
        raise ValueError(f"Native cohort exceeds 3GB budget: {total}")
    print(f"Native selected row-group download: {total / 1e9:.3f} GB in {len(grouped)} groups", flush=True)

    def acquire_group(batch):
        item = batch[0]
        mark = Path(item["native_file"]).with_suffix(f".group-{item['native_group']}.complete")
        if not mark.exists():
            chunk = read_range(item["native_url"], item["native_image_start"], item["native_image_end"])
            with open(item["native_file"], "r+b") as file:
                file.seek(item["native_image_start"])
                file.write(chunk)
            mark.write_text(builder.sha(chunk) + "\n")
        records = (
            pq.ParquetFile(item["native_file"])
            .read_row_group(item["native_group"], columns=["image"])
            .to_pylist()
        )
        for selected in batch:
            raw = records[selected["native_local_index"]]["image"]["bytes"]
            path = (
                builder.CACHE
                / f"{selected['kind']}-{selected['row']['source']}-{selected['row']['image_id']}.bin"
            )
            path.write_bytes(raw)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire_group, grouped.values()))
    return [
        (builder.CACHE / f"{item['kind']}-{item['row']['source']}-{item['row']['image_id']}.bin").read_bytes()
        for item in items
    ]
