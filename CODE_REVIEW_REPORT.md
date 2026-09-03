# Code Review Report: Advanced Graph Contrastive Learning for Recommendation Systems

## Tổng quan Project
Đây là một project nghiên cứu academic về Graph Contrastive Learning cho hệ thống recommendation, bao gồm 4 model SOTA: LightGCN, XSimGCL, DirectAU, và AdaptiveGCL (multimodal).

---

## ✅ TRẠNG THÁI CÁC FIXES

### Priority 1 (Critical) - 4/4 ✅
| # | Issue | Status | File |
|---|-------|--------|------|
| 1 | Memory leak trong cache | ✅ Fixed | `adaptive_gcl.py` |
| 2 | Infinite loop risk | ✅ Fixed | `trainer.py` |
| 3 | No gradient clipping | ✅ Fixed | `trainer.py` |
| 4 | Inconsistent paths | ✅ Fixed | `checkpoints.py` |

### Priority 2 (High) - 4/4 ✅
| # | Issue | Status | File |
|---|-------|--------|------|
| 5 | Duplicate code | ✅ Fixed | `geometry.py` |
| 10 | Vectorized sampling | ✅ Fixed | `trainer.py` |
| 12 | Mixed precision | ✅ Fixed | `trainer.py` |
| 17 | Config validation | ✅ Fixed | `config_schemas.py`, `config.py` |

### Priority 3 (Medium) - 4/4 ✅
| # | Issue | Status | File |
|---|-------|--------|------|
| 6 | Config validation | ✅ Fixed | `config.py` (integrated) |
| 7 | Magic numbers | ✅ Fixed | `trainer.py` |
| 19 | Bare except clauses | ✅ Fixed | `representation.py` |
| 20 | No checkpoint validation | ✅ Fixed | `early_stopping.py` |
| 21 | Race condition | ✅ Fixed | `early_stopping.py` |

### Priority 4 (Low) - 3/4 ✅
| # | Issue | Status | File |
|---|-------|--------|------|
| 8 | Type hints | ✅ Fixed | Multiple files |
| 14 | GPU sync | ✅ Fixed | `trainer.py` |
| 15 | Strategy pattern | ✅ Fixed | `loss_strategies.py` |

### Not Fixed
| # | Issue | Priority | Notes |
|---|-------|----------|-------|
| 11 | Sparse tensor ops | Low | Not critical |
| 18 | Data validation | Low | Pandera not integrated |
| 22-23 | Tests | High | Unit tests not added |

---

## 📋 CHI TIẾT CÁC FIXES

### 1. Memory Leak trong Cache (adaptive_gcl.py)
**Trước:**
```python
if self._adj_cache_key is not norm_adj:  # Object identity check
```
**Sau:**
```python
if self._adj_cache_key != id(norm_adj):  # Using id() for proper cache tracking
```

### 2. Infinite Loop Risk (trainer.py)
**Trước:**
```python
while neg_items[idx] in hist:  # Có thể vô hạn
    neg_items[idx] = np.random.randint(0, num_items)
```
**Sau:**
```python
while neg_items[idx] in hist and attempts < max_attempts:
    neg_items[idx] = np.random.randint(0, num_items)
    attempts += 1
```

### 3. Gradient Clipping (trainer.py)
**Thêm:**
```python
total_loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.optimizer.step()
```

### 4. Centralized Checkpoint Paths (checkpoints.py - NEW)
```python
def get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="run"):
def find_checkpoint(model_name, sparsity, seed):
def ensure_checkpoint_dir(model_name):
```

### 5. Shared Geometry Utilities (geometry.py - NEW)
```python
# Trong src/utils/geometry.py
def compute_alignment(...)
def compute_uniformity(...)
def batch_pairwise_uniformity(...)
```

### 6. Config Validation Integration (config.py)
**Thêm validation vào load_config():**
```python
if validate:
    config = validate_config(config)
    config = validate_model_config(config, model_name)
```

### 7. Module Constants (trainer.py)
```python
MASK_VALUE: float = -1e9
GRADIENT_CLIP_VALUE: float = 1.0
DEFAULT_TERMINAL_WIDTH: int = 80
SYNC_CUDA: bool = False  # Optional CUDA sync
```

### 8. Type Hints
Đã thêm type hints cho:
- `sample_negative_items()`
- `_get_propagation_adj()`
- `_compute_svd_singular_values()`
- Tất cả helper functions mới

### 9. Proper Error Handling (representation.py)
```python
def _compute_svd_singular_values(centered_emb: torch.Tensor) -> np.ndarray:
    try:
        singular_vals = torch.linalg.svdvals(centered_emb)
        return singular_vals.detach().cpu().numpy()
    except (RuntimeError, AttributeError) as pytorch_err:
        # Fallback to NumPy
        ...
    except Exception as numpy_err:
        raise RuntimeError(...) from numpy_err
```

### 10. Checkpoint Validation (early_stopping.py)
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
if "model_state_dict" not in checkpoint:
    raise ValueError("Invalid checkpoint format")
```

### 11. Atomic Write (early_stopping.py)
```python
temp_path = checkpoint_path + ".tmp"
torch.save(state, temp_path)
os.replace(temp_path, checkpoint_path)  # Atomic on POSIX
```

### 12. Batch Seen Items Masking (evaluator.py)
**Trước:** Loop per-user tạo tensor riêng
**Sau:** Batch tất cả seen items cùng lúc
```python
all_seen_items = set()
for u in batch_u_idx:
    seen = self.train_history.get(u, None)
    if seen:
        all_seen_items.update(seen)
if all_seen_items:
    seen_tensor = torch.tensor(list(all_seen_items), ...)
    scores[:, seen_tensor] = MASK_VALUE
```

### 13. Vectorized Negative Sampling (trainer.py)
```python
def sample_negative_items(...):
    n = len(users)
    neg_items = np.random.randint(0, num_items, size=n)
    user_histories = [train_history.get(u, set()) for u in users]
    # Vectorized collision detection
    collisions = np.array([neg_items[i] in user_histories[i] for i in range(n)], dtype=bool)
    # Batched retry with limit
    ...
```

### 14. Mixed Precision Training (trainer.py)
```python
self.use_amp = AMP_AVAILABLE and device.type == "cuda"
self.scaler = GradScaler() if self.use_amp else None

# Trong training loop:
if self.use_amp and self.scaler is not None:
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    self.optimizer.step()
```

### 15. Strategy Pattern (loss_strategies.py - NEW)
```python
class LossStrategy(ABC):
    @abstractmethod
    def compute_loss(...):
        pass

class BPRStrategy(LossStrategy):
class XSimGCLStrategy(LossStrategy):
class DirectAUStrategy(LossStrategy):
class AdaptiveGCLStrategy(LossStrategy):

def get_loss_strategy(model_name: str, config: Dict) -> LossStrategy:
```

---

## 📁 FILES CHANGED

### Modified Files:
| File | Changes |
|------|---------|
| `src/training/trainer.py` | Negative sampling, gradient clipping, AMP, constants, config validation |
| `src/training/early_stopping.py` | Checkpoint validation, atomic write |
| `src/evaluation/evaluator.py` | Batch seen items masking, constants |
| `src/evaluation/representation.py` | Proper error handling, type hints |
| `src/models/adaptive_gcl.py` | Cache identity tracking |
| `src/losses/directau.py` | Use shared geometry utilities |
| `src/utils/config.py` | Integrated config validation |
| `src/utils/__init__.py` | Updated exports |
| `src/training/__init__.py` | Added strategy exports |

### New Files:
| File | Purpose |
|------|---------|
| `src/utils/checkpoints.py` | Centralized checkpoint path utilities |
| `src/utils/geometry.py` | Shared alignment/uniformity metrics |
| `src/utils/config_schemas.py` | Config validation with Pydantic fallback |
| `src/training/loss_strategies.py` | Strategy pattern for loss computation |

---

## ✅ ĐIỂM MẠNH CỦA PROJECT

1. **Well-structured architecture** với BaseRecommender abstract class
2. **Comprehensive metrics** - accuracy, beyond-accuracy, representation geometry
3. **Good documentation** - docstrings chi tiết cho các main functions
4. **Checkpoint/Resume support** - training có thể resume
5. **Multi-sparsity evaluation** - benchmark ở nhiều sparsity levels
6. **Statistical significance testing** - paired t-test và Wilcoxon
7. **Visual progress bars** - tqdm integration tốt
8. **LaTeX table generation** - ready cho academic publications

---

*Report generated: 2026-08-30*
*Last updated: 2026-08-30*
*Reviewer: Claude Code*
