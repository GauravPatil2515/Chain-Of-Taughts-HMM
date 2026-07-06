"""Quick model loading test for Phase 1"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Testing model: {MODEL_ID}")
print("="*70)

print("\nStep 1: Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    exit(1)

print("\nStep 2: Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Model info
param_count = sum(p.numel() for p in model.parameters())
print(f"\nModel Information:")
print(f"  Parameters: {param_count/1e6:.1f}M")
print(f"  Device: {next(model.parameters()).device}")
print(f"  Dtype: {next(model.parameters()).dtype}")

# Quick generation test
print("\nStep 3: Testing generation...")
try:
    prompt = "What is 2+2?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Generation test passed")
    print(f"  Prompt: {prompt}")
    print(f"  Response: {response[:100]}...")
except Exception as e:
    print(f"❌ Generation test failed: {e}")
    exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - Model ready for batch generation!")
print("="*70)
