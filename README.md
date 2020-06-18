## Papers:
- [ ] Jasper https://arxiv.org/pdf/1904.03288.pdf
- [ ] QuartzNet https://arxiv.org/pdf/1910.10261.pdf

## Eval
Launch from docker:
```
python src/eval.py \
    --use-cuda \
    --path-to-csv data/numbers/test-example.csv \
    --path-to-weight weights.pth
```