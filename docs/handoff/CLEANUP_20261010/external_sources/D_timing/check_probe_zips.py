import zipfile
for z in (r'D:/models/Track2_ProbeSlow8.zip', r'D:/models/Track2_ProbeDetail.zip'):
    with zipfile.ZipFile(z) as f:
        r = f.read('model_dir/runner.py').decode('utf-8')
        names = f.namelist()
        print(z)
        print('  entries:', len(names), '| output_dir:', sum(1 for n in names if n.startswith('output_dir/')),
              '| model_dir:', sum(1 for n in names if n.startswith('model_dir/')))
        print('  has 8x loop:', 'for _ in range(7)' in r)
        print('  has detail boost:', '_detail_boost' in r)
        print('  runner bytes:', len(r))
