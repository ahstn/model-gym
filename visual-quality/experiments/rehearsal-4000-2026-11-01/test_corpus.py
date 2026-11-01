"""Invariants that prevent sampling leakage and malformed model inputs."""
import json
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
from corpus_finalize import allocations,content_sample
from corpus_select import diverse,proportional,CLUSTER_SEED
from sklearn.cluster import KMeans
from corpus_prepare import HERE,PILOT,read_lines


class SamplingTests(unittest.TestCase):
    def test_allocation_preserves_capacity_and_total(self):
        groups={0:list(range(2)),1:list(range(10)),2:list(range(100))}
        q=allocations(groups,30)
        self.assertEqual(sum(q.values()),30)
        self.assertTrue(all(q[c]<=len(groups[c]) for c in groups))
        self.assertEqual(q[0],2)

    def test_diversity_handles_repeated_vectors_without_duplicate_rows(self):
        pool=[{'candidate_id':str(i)} for i in range(8)]
        vec={str(i):np.array([1.,0.]) for i in range(8)}
        result=diverse(pool,6,vec,'test')
        self.assertEqual(len({r['candidate_id'] for r in result}),6)

    def test_zero_quota_clusters_do_not_add_images(self):
        pool=[{'candidate_id':str(i)} for i in range(6)]
        vec={str(i):np.array([1.,0.]) for i in range(6)}
        clusters={str(i):i for i in range(6)}
        self.assertEqual(diverse(pool,0,vec,'empty'),[])
        self.assertEqual(len(content_sample(pool,2,clusters,vec,'small')),2)

    def test_cluster_seed_is_accepted_by_actual_sklearn(self):
        labels=KMeans(n_clusters=2,random_state=CLUSTER_SEED,n_init=2).fit_predict(np.array([[0.,1.],[0.,2.],[8.,9.]]))
        self.assertEqual(len(labels),3)

    def test_stratified_sample_preserves_requested_total(self):
        pool=[{'candidate_id':str(i)} for i in range(33)]
        clusters={str(i):i%7 for i in range(33)}
        result=proportional(pool,17,clusters,'test')
        self.assertEqual(len(result),17)
        self.assertEqual(len({r['candidate_id'] for r in result}),17)

    def test_candidate_manifest_prevents_protected_leakage(self):
        candidate=json.loads((HERE/'candidate-manifest.json').read_text())['cases']
        held=read_lines(PILOT/'development.jsonl')+read_lines(PILOT/'test.jsonl')
        groups={r['split_group'] for r in held};hashes={r['sha256'] for r in held};pixels={r['decoded_sha256'] for r in held}
        self.assertEqual(len(candidate),5500)
        self.assertEqual(len({r['family_id'] for r in candidate}),5500)
        self.assertEqual(len({r['view_sha256'] for r in candidate}),5500)
        for row in candidate:
            self.assertNotIn(row['family_id'],groups)
            self.assertNotIn(row['source_sha256'],hashes)
            self.assertNotIn(row['decoded_sha256'],pixels)
            self.assertGreaterEqual(min(row['width'],row['height']),256)
            self.assertGreaterEqual(row['width']*row['height'],78400)
            self.assertLessEqual(row['delivered_bytes'],2650000)
            self.assertFalse(any(row['task_masks'].values()))
        benchmark=[r for r in candidate if r['benchmark_candidate']]
        self.assertEqual(len(benchmark),1000)
        self.assertTrue(all(r['historically_fresh'] and not r['controlled_parent'] for r in benchmark))


if __name__=='__main__':unittest.main()
