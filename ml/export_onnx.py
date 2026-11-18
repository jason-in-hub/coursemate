"""Export the fine-tuned engine MiniLM (engine_model/minilm, from train_engine.py) to ONNX for the in-browser engine (../engine.js).
Output: engine_model/onnx/{config.json, tokenizer.json, tokenizer_config.json, onnx/model_quantized.onnx}, the layout transformers.js loads.
Weights: int8 with one fp32 scale per 32 input rows of each MatMul weight and per 32 dims of each embedding row, dequantised in the
graph with Cast/Mul/Reshape (standard ops, so onnxruntime-web WASM and WebGPU both run it). Activations stay fp32.
Measured against sentence-transformers on CS50 + bank texts: per-text cosine >= 0.9997 (onnxruntime's quantize_dynamic: 0.909).
Needs onnx on top of requirements.txt; keep it out of the repo venv, e.g.:
  .venv/bin/python -m pip install --target /tmp/onnxlib onnx  &&  ONNXLIB=/tmp/onnxlib .venv/bin/python export_onnx.py
Check the result with ../parity_engine.cjs."""
import os, sys, shutil
from pathlib import Path
if os.environ.get('ONNXLIB'): sys.path.append(os.environ['ONNXLIB'])  # appended, so the venv's own packages win
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch, onnx
from onnx import numpy_helper, helper, TensorProto
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / 'engine_model' / 'minilm', HERE / 'engine_model' / 'onnx'
BLOCK = 32
NAMES = ['input_ids', 'attention_mask', 'token_type_ids']


class Hidden(torch.nn.Module):  # BertModel -> last_hidden_state only; pooling and normalisation happen in engine.js
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state


def export(fp32):
    m, tok = AutoModel.from_pretrained(SRC).eval(), AutoTokenizer.from_pretrained(SRC)
    x = tok(['hello world', 'a longer example sentence'], padding=True, return_tensors='pt')
    torch.onnx.export(Hidden(m), tuple(x[k] for k in NAMES), str(fp32), input_names=NAMES, output_names=['last_hidden_state'],
                      dynamic_axes={k: {0: 'batch', 1: 'seq'} for k in NAMES + ['last_hidden_state']}, opset_version=17, dynamo=False)


def quantize(src, dst):
    m = onnx.load(str(src)); g = m.graph
    inits, uses, new = {i.name: i for i in g.initializer}, {}, []
    for n in g.node:
        for k, x in enumerate(n.input): uses.setdefault(x, []).append((n.op_type, k))
    def q(name, gather):
        W = numpy_helper.to_array(inits[name]).astype(np.float32)
        if gather: W = W.T                       # [vocab, dim] -> block along dim
        K, N = W.shape
        Wb = W.reshape(K // BLOCK, BLOCK, N)
        s = np.abs(Wb).max(axis=1, keepdims=True) / 127.0; s[s == 0] = 1
        Q = np.clip(np.round(Wb / s), -127, 127).astype(np.int8)
        if gather:                               # back to [vocab, dim] blocks: Q [N, K/B, B], s [N, K/B, 1]
            Q = Q.reshape(K, N).T.reshape(N, K // BLOCK, BLOCK).copy(); s = s.reshape(K // BLOCK, N).T.reshape(N, K // BLOCK, 1).copy(); shape = [N, K]
        else: shape = [K, N]
        g.initializer.remove(inits[name])
        g.initializer.extend([numpy_helper.from_array(Q, name + '_q'), numpy_helper.from_array(s.astype(np.float32), name + '_s'),
                              numpy_helper.from_array(np.array(shape, dtype=np.int64), name + '_shape')])
        new.extend([helper.make_node('Cast', [name + '_q'], [name + '_f'], to=TensorProto.FLOAT),
                    helper.make_node('Mul', [name + '_f', name + '_s'], [name + '_m']),
                    helper.make_node('Reshape', [name + '_m', name + '_shape'], [name])])
    for name, init in list(inits.items()):
        if init.data_type != TensorProto.FLOAT or len(init.dims) != 2 or init.dims[0] * init.dims[1] < 4096: continue
        ops = set(uses.get(name, []))
        if ops and ops <= {('MatMul', 1)}: q(name, False)
        elif ops and ops <= {('Gather', 0)}: q(name, True)
    for n in reversed(new): g.node.insert(0, n)
    onnx.checker.check_model(m); onnx.save(m, str(dst))


if __name__ == '__main__':
    (OUT / 'onnx').mkdir(parents=True, exist_ok=True)
    for f in ['config.json', 'tokenizer.json', 'tokenizer_config.json']: shutil.copy(SRC / f, OUT / f)
    fp32 = OUT / 'model_fp32.onnx'
    export(fp32); quantize(fp32, OUT / 'onnx' / 'model_quantized.onnx'); fp32.unlink()
    print('wrote', OUT / 'onnx' / 'model_quantized.onnx', (OUT / 'onnx' / 'model_quantized.onnx').stat().st_size // 1_000_000, 'MB')
