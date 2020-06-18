## Papers:
- [x] Jasper https://arxiv.org/pdf/1904.03288.pdf
- [x] CTC https://www.cs.toronto.edu/~graves/icml_2006.pdf
- [ ] QuartzNet https://arxiv.org/pdf/1910.10261.pdf

## Short Description
Work based on Jasper and CTC.
The trained model is at the root directory: `weights.pth`

## Eval
Launch from docker:
```
python src/eval.py \
    --use-cuda \
    --path-to-csv data/numbers/test-example.csv \
    --path-to-weight weights.pth
```
