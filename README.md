# timesFM_fc

Standalone deployment bundle that packages the minimal TimesFM 2.5 inference stack,
local fine-tuning adapter, and an FC-oriented handler for scoring stocks.

## Layout

```
timesFM_fc/
├── requirements.txt           # Python dependencies
├── local_timesfm_model/       # local TimesFM 2.5 safetensors checkpoint
├── finetuned_adapter.pth      # optional residual adapter for custom stocks
├── src/
│   └── timesfm/               # minimal TimesFM 2.5 implementation (PyTorch)
└── examples/
    ├── fc_timesfm_score.py    # FC entrypoint / handler
    └── data.py                # OSS data loader helper
```

## Quick start

```bash
cd /Users/fengzhi/Downloads/git/timesFM_fc
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 以“源码形式”引用本地包（可选，便于调试修改 src）
pip install -e src
python examples/fc_timesfm_score.py
```

The handler can also be imported as an FC function:

```python
from examples.fc_timesfm_score import handler

result = handler({"codes": ["601006", "603993"]}, None)
```

## Configuration

Update `examples/data.py` with the correct OSS endpoint, bucket, and credentials.
Fine-tune behavior is controlled via `CONFIG` in `fc_timesfm_score.py`.

## Notes

- The bundle intentionally omits archived `v1/` code and other auxiliary tooling.
- Ensure the `local_timesfm_model/model.safetensors` file ships with the deployment artifact.
