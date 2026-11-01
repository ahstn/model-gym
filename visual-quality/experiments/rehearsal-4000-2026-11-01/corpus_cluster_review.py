"""Make cluster coverage and representative content auditable without quality labels."""
import json
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw
from corpus_select import HERE,sha,write


def main():
    candidates=json.loads((HERE/'candidate-manifest.json').read_text())['cases']
    selected=json.loads((HERE/'manifest.json').read_text())['cases']
    assignments=json.loads((HERE/'rehearsal-clusters.json').read_text())['clusters']
    vectors=np.load(HERE/'candidate-embeddings.npz',allow_pickle=False)
    embeddings=dict(zip(vectors['case_ids'].tolist(),vectors['embeddings']))
    groups=defaultdict(list)
    for row in candidates:groups[assignments[row['candidate_id']]].append(row)
    selected_ids={r['candidate_id'] for r in selected if r['role']!='variant'}
    selected_counts=Counter(r['rehearsal_content_cluster'] for r in selected if r['role']!='variant')
    selected_views=Counter(r['rehearsal_content_cluster'] for r in selected)
    audit=[];representatives={}
    for cluster,rows in sorted(groups.items()):
        matrix=np.asarray([embeddings[r['candidate_id']] for r in rows]);centroid=matrix.mean(axis=0)
        centroid/=np.linalg.norm(centroid);distance=matrix@centroid
        order=np.argsort(-distance)
        shown=list(order[:6])+list(order[-2:])
        representatives[cluster]=[rows[i] for i in shown]
        audit.append({'cluster':cluster,'candidate_families':len(rows),'selected_families':selected_counts[cluster],
            'candidate_views':len(rows),'selected_views':selected_views[cluster],
            'candidate_fraction':len(rows)/len(candidates),'selected_fraction':selected_counts[cluster]/len(selected_ids),
            'candidate_source_categories':dict(Counter(r['category'] for r in rows)),
            'candidate_components':dict(Counter(r['component'] for r in rows)),
            'review_representatives':[rows[i]['candidate_id'] for i in shown],
            'review_layout':'six nearest to centroid, followed by two furthest; text names are manual descriptions only'})
    write(HERE/'content-coverage.json',{
        'candidate_manifest_sha256':sha(HERE/'candidate-manifest.json'),'manifest_sha256':sha(HERE/'manifest.json'),
        'clusters_sha256':sha(HERE/'rehearsal-clusters.json'),'embeddings_sha256':sha(HERE/'candidate-embeddings.npz'),
        'candidate_families':5500,'selected_families':3700,'clusters':audit,
        'scope':'Derived variants do not contribute additional counts to content diversity'})
    representatives_selected=[r for r in selected if r['role']!='variant']
    representative_matrix=np.asarray([embeddings[r['candidate_id']] for r in representatives_selected])
    sim=representative_matrix@representative_matrix.T
    family_neighbours={}
    for i,row in enumerate(representatives_selected):
        sim[i,i]=-np.inf
        closest=np.argsort(-sim[i])[:5]
        family_neighbours[row['family_id']]=[
            {'case_id':representatives_selected[j]['case_id'],'cosine':float(sim[i,j])} for j in closest]
    write(HERE/'selected-neighbours.json',{
        'method':'Top five cosine neighbours among selected cross-family representatives; controlled derivatives inherit their source-parent content embedding',
        'manifest_sha256':sha(HERE/'manifest.json'),'candidate_embeddings_sha256':sha(HERE/'candidate-embeddings.npz'),
        'neighbours':{r['case_id']:family_neighbours[r['family_id']] for r in selected},
        'embedding_basis':{r['case_id']:{'candidate_id':r['candidate_id'] or r['parent_candidate_id'],
            'view_sha256':r.get('embedding_view_sha256',r['view_sha256']),
            'inherited_from_parent':r['role']=='variant',
            'same_as_delivered_view':r.get('embedding_view_sha256',r['view_sha256'])==r['view_sha256']} for r in selected},
        'warning':'Similarity is about content, not equal quality; a controlled family may use a smaller final size cap than its embedding representative'})
    for start in range(0,40,5):
        page=Image.new('RGB',(1600,1250),(241,244,247));draw=ImageDraw.Draw(page)
        for offset,cluster in enumerate(range(start,min(start+5,40))):
            y=offset*250;rows=representatives[cluster]
            draw.text((8,y+5),f'Cluster {cluster}: candidate={len(groups[cluster])}, selected families={selected_counts[cluster]}',fill=(12,22,32))
            for i,row in enumerate(rows):
                x=i*200
                with Image.open(row['image_path']) as im:
                    im=im.convert('RGB');im.thumbnail((190,190));page.paste(im,(x+(200-im.width)//2,y+27))
                mark=' selected' if row['candidate_id'] in selected_ids else ''
                draw.text((x+3,y+220),row['candidate_id']+mark,fill=(12,22,32))
        dest=HERE/'review'/f'cluster-content-{start//5+1:02d}.jpg';dest.parent.mkdir(exist_ok=True);page.save(dest,quality=92)
    print('Saved coverage and eight cluster-content review pages',flush=True)


if __name__=='__main__':main()
