# Experiment results (compressed)

`results/experiments/` (4.4 GB, 262 files) split into 6 independent `.tar.gz` archives,
each below 100 MB (GitHub's per-file limit). Every archive stores paths as
`experiments/...`, so extracting all of them inside `results/` rebuilds
`results/experiments/`:

```bash
cd results
(cd experiments_archive && sha256sum -c SHA256SUMS)   # optional integrity check
for f in experiments_archive/experiments.part*.tar.gz; do tar -xzf "$f"; done
```
