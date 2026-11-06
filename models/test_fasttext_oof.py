import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import fasttext_oof as O


class OOFTest(unittest.TestCase):
    def test_grouping_excludes_duplicate_facts_and_case_ids(self):
        rows=[dict(jid=str(i),fact=str(i)) for i in range(12)]
        rows[1]['fact']=rows[0]['fact']
        rows[2]['jid']=rows[1]['jid']
        train=np.arange(10)
        fold=O.grouped_inner_folds(rows,train,3,0)
        self.assertEqual(fold[0],fold[1])
        self.assertEqual(fold[1],fold[2])
        self.assertTrue(np.all(fold[10:]==-1))
        np.testing.assert_array_equal(fold,O.grouped_inner_folds(rows,train,3,0))

    def test_complete_tiny_oof_export_has_no_fit_or_selection_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';data.mkdir()
            original=root/'original';output=root/'output'
            classes=dict(accusation=['a','b'],article=['x','y'])
            roles1=['train']*8+['valid']*2+['test']*2+['holdout']*2
            roles2=['test']*2+['train']*8+['valid']*2+['holdout']*2
            rows=[dict(fact='案件甲乙'+str(i),meta=dict(jid=str(i),year=2025 if i>=12 else 2024,
                  accusation=[classes['accusation'][i%2]],relevant_articles=['x'],
                  relevant_articles_with_suffix=[classes['article'][i%2]],
                  task_validity=dict(accusation=True,article=i!=3,imprisonment=i!=4),
                  term_of_imprisonment=dict(imprisonment=6,death_penalty=False,life_imprisonment=False))) for i in range(14)]
            (data/'shared_100k_with_article_suffix.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (data/'fixed_classes_coarse.json').write_text(json.dumps(classes))
            (data/'fixed_classes_suffix.json').write_text(json.dumps(classes))
            with (data/'shared_fold_assignments.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['sample_line_1based','jid','fold_01','fold_02'])
                writer.writeheader()
                for i in range(14):writer.writerow(dict(sample_line_1based=i+1,jid=str(i),fold_01=roles1[i],fold_02=roles2[i]))
            manifest=dict(data_sha256={n:O.sha(data/n) for n in ['shared_100k_with_article_suffix.jsonl','shared_fold_assignments.csv']},sha256={})
            for fold,roles in [('fold_01',roles1),('fold_02',roles2)]:
                (original/fold).mkdir(parents=True)
                for task in O.TASKS:
                    model=O.FastText(64,8,2)
                    torch.save(model.state_dict(),original/fold/(task+'.pt'))
                    meta=dict(fold=fold,classes=classes[task],train_source_lines=[i+1 for i,r in enumerate(roles) if r=='train' and rows[i]['meta']['task_validity'][task]])
                    np.savez(original/fold/(task+'.npz'),metadata=np.asarray(json.dumps(meta)),roles=np.asarray(roles),source_lines=np.arange(1,15))
                    manifest['sha256'][f'{fold}/{task}.pt']=O.sha(original/fold/(task+'.pt'))
            O.write(root/'manifest.json',manifest)
            args=argparse.Namespace(shared=str(data),original_vectors=str(original),checkpoint_manifest=str(root/'manifest.json'),output=str(output),inner_folds=2,epochs=2,patience=1,batch=3,buckets=64,dim=8,threads=1,seed=0,lr=.01)
            O.run(args)
            result=json.loads((output/'run_manifest.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['completed_models'],8)
            for p in (output/'checkpoints').rglob('*.json'):
                if p.name.endswith('.history.json'):continue
                m=json.loads(p.read_text())
                self.assertFalse(set(m['oof_source_lines'])&set(m['train_source_lines']))
                self.assertFalse(set(m['oof_source_lines'])&set(m['selection_source_lines']))
            for p in (output/'vectors').rglob('*.npz'):
                with np.load(p) as z, np.load(output/'in_sample_probability_control'/p.relative_to(output/'vectors')) as base:
                    train=z['roles']=='train'
                    self.assertTrue(np.all(z['producer_inner_fold'][train]>=0))
                    self.assertTrue(np.all(z['producer_inner_fold'][~train]==-1))
                    np.testing.assert_array_equal(z['vectors'][~train],base['vectors'][~train])
                    self.assertEqual(z['vectors'].shape,(14,2))
                    self.assertFalse(np.array_equal(z['vectors'][train],base['vectors'][train]))
                    self.assertTrue(np.isfinite(z['vectors']).all())
                    self.assertEqual(json.loads(z['metadata'].item())['classes'],classes[p.stem])


if __name__=='__main__':
    unittest.main()
