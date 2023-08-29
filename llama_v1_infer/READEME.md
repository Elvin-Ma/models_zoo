# llama_v1 address
[llama_v1 github](https://github.com/facebookresearch/llama.git)

# 2. env prepare
- commit id: 57b0eb62de0636e75af471e49e2f1862d908d9d8

**diff**
```python

```


# 3. run:
torchrun --nproc_per_node 1 example.py --ckpt_dir ./7B --tokenizer_path ./7B/tokenizer.model
