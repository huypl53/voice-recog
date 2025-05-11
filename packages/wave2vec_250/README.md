# wave2vec-250h

## Installation

```bash
mkdir ./cache/
wget https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/resolve/main/vi_lm_4grams.bin.zip
unzip -q vi_lm_4grams.bin.zip -d ./cache/

# Test
uv run ./src/wave2vec_250/processor.py
```
