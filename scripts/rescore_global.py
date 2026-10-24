"""Freeze and rescore existing models on development under Global WP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort

from connectome.calibration import PredictionCache, apply_calibration, paired_bootstrap
from connectome.data import file_fingerprint, iter_sequences
from connectome.evaluation import file_hash, save_json
from connectome.global_metric import GlobalAccumulator, METRIC_ID
from connectome.gpu_training import digest
from connectome.recurrent_parity import check_parity
from connectome.runtime import session_options
from utils import add_scored_sequence

DEFAULT_OUTPUT = ROOT / 'artifacts/global_wp/rescore_v1'


def read(path):
    return json.loads(Path(path).read_text())


def candidates():
    rows = []
    for name, config, cache in (
        ('baseline', 'configs/baseline.json', 'artifacts/cache/baseline_dev'),
        ('baseline_quarter', 'configs/baseline_quarter.json', 'artifacts/cache/baseline_dev'),
        ('baseline_half', 'configs/baseline_half.json', 'artifacts/cache/baseline_dev'),
        ('affine256_raw', 'configs/affine256_raw.json', 'artifacts/training_pilot256/affine_p1e-2/dev_cache'),
        ('affine256_scaled', 'configs/affine256_scaled.json', 'artifacts/training_pilot256/affine_p1e-2/dev_cache'),
        ('affine1024_raw', 'configs/affine1024_raw.json', 'artifacts/training_pilot1024/affine_p1e-2/dev_cache'),
        ('affine1024_scaled', 'configs/affine1024_scaled.json', 'artifacts/training_pilot1024/affine_p1e-2/dev_cache'),
        ('selected', 'submission_config.json', 'artifacts/selected/dev_cache'),
        ('round2_affine_rho', 'artifacts/round2/affine_rho/config.json', None),
        ('correlation256', 'artifacts/round2/correlation_gru_lr3e-5/config.json', None),
        ('correlation1024_seed20260912', 'artifacts/round2/correlation_gru1024_seed20260912/config.json', None),
        ('correlation1024_seed20260913', 'artifacts/round2/correlation_gru1024_seed20260913/config.json', None),
    ):
        config_path = ROOT / config
        settings = read(config_path)
        model = (config_path.parent / settings['model']).resolve()
        row = {'name': name, 'config': config, 'config_sha256': file_hash(config_path),
               'model': str(model.relative_to(ROOT)), 'model_sha256': file_hash(model),
               'postprocess': settings.get('postprocess', {}), 'cache': cache}
        if cache:
            row['cache_metadata_sha256'] = file_hash(ROOT / cache / 'metadata.json')
        else:
            sequence = model.with_name('sequence.onnx')
            row.update(sequence_model=str(sequence.relative_to(ROOT)), sequence_sha256=file_hash(sequence))
        rows.append(row)
    return rows


def sources():
    names = ['scripts/rescore_global.py', 'utils.py', 'connectome/global_metric.py',
             'connectome/calibration.py', 'connectome/evaluation.py', 'connectome/data.py',
             'connectome/runtime.py', 'connectome/recurrent_parity.py']
    return {name: file_hash(ROOT / name) for name in names}


def declare(output=DEFAULT_OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve existing rescore policy and outputs')
    manifest = read(ROOT / 'artifacts/manifest.json')
    groups = manifest['splits']['dev']
    if set(groups) & set(manifest['splits']['confirmation']):
        raise ValueError('development overlaps consumed confirmation')
    dataset = ROOT / 'datasets/valid.parquet'
    fingerprint = file_fingerprint(dataset)
    if fingerprint != manifest['datasets']['valid']['fingerprint']:
        raise ValueError('validation dataset differs from the historical manifest')
    policy = {'metric_id': METRIC_ID, 'phase': 'global_metric_migration_rescore',
              'groups': groups, 'manifest_sha256': file_hash(ROOT / 'artifacts/manifest.json'),
              'dataset_fingerprint': fingerprint, 'candidates': candidates(), 'code': sources(),
              'reference_sha256': file_hash(ROOT / 'artifacts/global_wp/reference/METRIC.md'),
              'bootstrap': {'n_resamples': 10000, 'seed': 20261024},
              'selection': 'highest complete development Global WP; fixed candidate order breaks exact ties',
              'scope': '12 existing configurations only; no new calibration search or training; confirmation permanently excluded',
              'exclusions': 'unreleased failed head/student/causal experiments; no new architectures'}
    policy['policy_sha256'] = digest(policy)
    output.mkdir(parents=True)
    save_json(output / 'policy.json', policy)
    print(f"Declared {len(policy['candidates'])} existing configurations on {len(groups)} development sequences.", flush=True)
    return policy


def load_policy(output):
    policy = read(Path(output) / 'policy.json')
    if (policy.get('metric_id') != METRIC_ID or policy.get('policy_sha256') != digest({k:v for k,v in policy.items() if k!='policy_sha256'})
            or policy['code'] != sources() or policy['candidates'] != candidates()
            or policy['manifest_sha256'] != file_hash(ROOT / 'artifacts/manifest.json')
            or policy['dataset_fingerprint'] != file_fingerprint(ROOT / 'datasets/valid.parquet')):
        raise ValueError('frozen rescore inputs, code or policy changed')
    manifest = read(ROOT / 'artifacts/manifest.json')
    if policy['groups'] != manifest['splits']['dev'] or set(policy['groups']) & set(manifest['splits']['confirmation']):
        raise ValueError('rescore domain must be unchanged development only')
    return policy


def run(output=DEFAULT_OUTPUT, *, workers=4):
    output = Path(output)
    policy = load_policy(output)
    if (output / 'ranking.json').exists() or (output / 'progress.json').exists():
        raise FileExistsError('rescore already started; preserve its outputs')
    groups = policy['groups']
    accumulators = {row['name']: GlobalAccumulator() for row in policy['candidates']}
    sequences = {row['name']: [] for row in policy['candidates']}
    caches, sessions, new_caches = {}, {}, {}
    dataset = ROOT / 'datasets/valid.parquet'
    manifest = read(ROOT / 'artifacts/manifest.json')
    ids = {row['row_group']:row['seq_ix'] for row in manifest['datasets']['valid']['groups']}
    for row in policy['candidates']:
        name = row['name']
        if row['cache']:
            if row['cache'] not in caches:
                cache = PredictionCache.open(ROOT / row['cache'])
                if cache.metadata['provenance']['model_sha256'] != row['model_sha256'] or set(cache.groups) != set(groups):
                    raise ValueError('cache model or development domain mismatch')
                caches[row['cache']] = cache
        else:
            # No scoring in this check; use one fixed, complete development sequence.
            check_parity(dataset, ROOT / row['model'], groups=manifest['splits']['smoke'][:1],
                         bulk_model=ROOT / row['sequence_model'], report_path=output / f'{name}.parity.json')
            sessions[name] = ort.InferenceSession(str(ROOT / row['sequence_model']), sess_options=session_options(),
                                                 providers=['CPUExecutionProvider'])
            new_caches[name] = PredictionCache.create(output / f'{name}.cache', groups, [ids[g] for g in groups],
                {'model_sha256':row['model_sha256'], 'bulk_model_sha256':row['sequence_sha256'],
                 'dataset_path':str(dataset), 'dataset_size':dataset.stat().st_size,
                 'raw_unclipped':True, 'mode':'offline_sequence', 'policy_sha256':policy['policy_sha256']})
    started = time.monotonic()

    def infer(name, features):
        prediction = sessions[name].run(['prediction'], {'features':features[None],
            'hidden_0':np.zeros((1,1,128),np.float32), 'hidden_1':np.zeros((1,1,128),np.float32)})[0][0]
        if prediction.shape != (20000,2) or not np.isfinite(prediction).all():
            raise ValueError('invalid offline predictions')
        return prediction

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, sequence in enumerate(iter_sequences(dataset, groups)):
            futures = {name:pool.submit(infer,name,sequence.features) for name in sessions}
            raw = {cache_path:cache.read_group(sequence.row_group) for cache_path,cache in caches.items()}
            for cache in caches.values():
                if cache.seq_ids[cache._index[sequence.row_group]] != sequence.seq_ix:
                    raise ValueError('cached sequence identifier changed')
            for name, future in futures.items():
                raw[name] = future.result()
                new_caches[name].write(index, raw[name])
            for row in policy['candidates']:
                name = row['name']
                prediction = raw[row['cache'] or name]
                if row['postprocess']:
                    prediction = apply_calibration(prediction, **row['postprocess'])
                sequences[name].append(add_scored_sequence(accumulators[name], sequence.targets, prediction,
                    sequence.mask & sequence.need, row_group=sequence.row_group, seq_ix=sequence.seq_ix))
            if (index+1)%32==0 or index+1==len(groups):
                elapsed=time.monotonic()-started
                progress={'sequences':index+1, 'total':len(groups), 'elapsed_seconds':elapsed,
                          'input_rows_per_second':(index+1)*20000/elapsed}
                save_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    for cache in new_caches.values():
        cache.finalize()
    reports = {}
    for row in policy['candidates']:
        name=row['name']
        report={**accumulators[name].result(), 'per_sequence':sequences[name], 'postprocess':row['postprocess'],
                'postprocess_tracked':True, 'rows_seen':len(groups)*20000,
                'provenance':{'model_sha256':row['model_sha256'],'dataset_path':str(dataset),
                              'dataset_size':dataset.stat().st_size,'dataset_fingerprint':policy['dataset_fingerprint'],
                              'prediction_cache':row['cache'] or str((output/f'{name}.cache').relative_to(ROOT))},
                'policy_sha256':policy['policy_sha256'],'confirmation_used':False}
        save_json(output/f'{name}.json',report)
        reports[name]=report
    ranking=[]
    for row in policy['candidates']:
        report=reports[row['name']]
        ranking.append({**row,'weighted_pearson':report['weighted_pearson'],'t0':report['t0'],'t1':report['t1'],
                        'report':str((output/f"{row['name']}.json").relative_to(ROOT)),
                        'report_sha256':file_hash(output/f"{row['name']}.json"),
                        'versus_selected':paired_bootstrap(report,reports['selected'],**policy['bootstrap'])})
    ranking.sort(key=lambda row:-row['weighted_pearson'])
    result={'metric_id':METRIC_ID,'policy_sha256':policy['policy_sha256'],'ranking':ranking,
            'selected':ranking[0],'development_sequences':len(groups),'confirmation_used':False,
            'selection_caveat':'development-selected; bootstrap does not correct selection bias',
            'elapsed_seconds':time.monotonic()-started}
    save_json(output/'ranking.json',result)
    print(json.dumps({'selected':ranking[0]['name'],'global_wp':ranking[0]['weighted_pearson']}),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['declare','run'])
    parser.add_argument('--output',default=str(DEFAULT_OUTPUT))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='declare':
        declare(args.output)
    else:
        run(args.output,workers=args.workers)
