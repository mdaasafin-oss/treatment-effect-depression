# treatment-effect-depression
Treatment effect estimation on depression data
## Files
- `baseline_model.py` — baseline model
- `optimized_model.py` — optimized model  
- `ablation_study.py` — five-experiment ablation study

## Usage
Set `data_path` / `DATA_PATH` to point to `data_generated.csv` before running.
Random seed is fixed at 42 throughout.

## Reproducibility
Pre-trained checkpoints:
- `baseline_model.pt`
- `improved_causal_model_ultimate.pt`

Load with:
```python
checkpoint = torch.load('baseline_model.pt')
model.load_state_dict(checkpoint['model_state'])
```
