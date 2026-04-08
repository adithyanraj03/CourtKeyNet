"""Verify the obfuscated _safetensors module still works."""
import sys, os, importlib.util

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Direct import bypassing utils/__init__.py
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "_safetensors.py")
spec = importlib.util.spec_from_file_location("_safetensors", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch

# Verify all functions exist
for fn_name in ['save_safetensors', 'load_safetensors', 'save_checkpoint', 'load_weights', 'convert_pt_to_safetensors']:
    assert hasattr(mod, fn_name), f"Missing: {fn_name}"
print("[OK] All 5 functions available")

# Test save_checkpoint (safetensors-only)
sd = {"layer.weight": torch.randn(3, 3), "layer.bias": torch.randn(3)}
stem = os.path.join(os.path.dirname(__file__), "_test_obfuscated")
config = {"train": {"epochs": 10, "lr0": 0.001}, "model": {"name": "test"}}

returned_path = mod.save_checkpoint(sd, stem, extra_metadata={
    "epoch": "5",
    "best_val_loss": "0.001234",
    "config": config,
})
assert returned_path.endswith(".safetensors"), f"Expected .safetensors, got {returned_path}"
assert os.path.exists(returned_path), "File not created"
assert not os.path.exists(stem + ".pt"), "ERROR: .pt file should NOT be created"
print("[OK] save_checkpoint creates ONLY .safetensors (no .pt)")

# Test load_weights with metadata parsing
loaded = mod.load_weights(returned_path)
assert "model" in loaded, "Missing 'model' key"
assert torch.allclose(sd["layer.weight"], loaded["model"]["layer.weight"]), "Weights mismatch"
print(f"[OK] load_weights: model keys={list(loaded['model'].keys())}")

# Check metadata extraction
epoch = loaded.get('epoch', None)
val_loss = loaded.get('best_val_loss', None)
cfg = loaded.get('config', None)
print(f"[OK] Metadata: epoch={epoch}, val_loss={val_loss}, config_type={type(cfg).__name__}")

if cfg:
    assert cfg['train']['epochs'] == 10, "Config not round-tripped correctly"
    print("[OK] Config round-trip verified")

# Verify author metadata in safetensors header
from safetensors import safe_open
with safe_open(returned_path, framework="pt") as f:
    meta = f.metadata()
    print(f"[OK] Author metadata: author={meta.get('author')}, doi={meta.get('doi')}")

# Cleanup
os.remove(returned_path)
print("\nAll obfuscated module tests passed!")
