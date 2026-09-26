"""Assemble the review ZIP from an explicit, bounded set of artifacts and evidence."""
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


# The ZIP names the commit it was packaged from, so that commit must hold what is packaged.
if git('status', '--porcelain', '--untracked-files=no'):
    raise SystemExit('Commit the tracked changes first; the ZIP records the commit it came from.')
commit = git('rev-parse', 'HEAD')
NOTE = ROOT / 'supplement'
STAGE = ROOT / 'tmp/supplement-package'
STAGE.mkdir(parents=True, exist_ok=True)
selected = {}
for name in ('jaric_extreme_context_targets.pdf', 'jaric_extreme_context_targets.tex', 'README.md'):
    selected[name] = NOTE / name
for folder in ('figures', 'plot_data'):
    for p in (NOTE/folder).iterdir():
        if p.suffix in ('.pdf', '.png', '.csv'):
            selected[f'{folder}/{p.name}'] = p
selected['source/analysis/supplement_figures.py'] = ROOT/'analysis/supplement_figures.py'
for name in ('clip_context.csv', 'clip_context_real.csv', 'clip_context_real_v25v26.csv'):
    selected[f'source/results/h3_repair/{name}'] = ROOT/'results/h3_repair'/name
for name in ('clip_context_real_native.csv', 'clip_context_real_tabicl_ft.csv',
             'unit_error_real.csv'):
    selected[f'evidence/historical/{name}'] = ROOT/'results/h3_repair'/name
for p in (ROOT/'results/recheck_20260925').iterdir():
    if p.suffix in ('.json', '.csv', '.npz') and p.name != 'grid_example.csv':
        selected[f'evidence/recheck/{p.name}'] = p
for name in ('recheck.py', 'recheck_v3_numeric.py', 'recheck_v3_direction.py',
             'recheck_baseline_capping.py'):
    selected[f'evidence/audit_scripts/{name}'] = ROOT/'analysis'/name
manifest = []
for arc, source in selected.items():
    dest = STAGE/arc
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    manifest.append(f'{hashlib.sha256(dest.read_bytes()).hexdigest()}  {arc}')
(STAGE/'SOURCE_COMMIT').write_text(commit + '\n', encoding='utf-8')
manifest.append(f"{hashlib.sha256((STAGE/'SOURCE_COMMIT').read_bytes()).hexdigest()}  SOURCE_COMMIT")
(STAGE/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n', encoding='utf-8')
archive = ROOT/'output/pdf/jaric_mlss2027_supplement.zip'
archive.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for arc in selected:
        z.write(STAGE/arc, arcname=arc)
    z.write(STAGE/'SOURCE_COMMIT', arcname='SOURCE_COMMIT')
    z.write(STAGE/'MANIFEST.sha256', arcname='MANIFEST.sha256')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(f'{archive}: {archive.stat().st_size:,} bytes, {len(selected)+2} files')
