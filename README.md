
This repository uses `git-lfs`. Make sure to set up `git-lfs` before clone

```sh
git lfs install
```

Before running examples, prepare `model.onnx` according to README inside models/onnx directories.

```
.
├── README.md
├── embedding-with-builtin-tokenizer
├── embedding-with-custom-tokenizer
└── models/onnx
    ├── cl-nagoya
    │   └── sup-simcse-ja-base
    ├── optimum
    │   └── all-MiniLM-L6-v2
    └── sentence-transformers
        └── paraphrase-xlm-r-multilingual-v1
```

