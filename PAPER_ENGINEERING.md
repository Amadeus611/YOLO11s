# PVRP + SNAA 工程说明（论文复现手册）

> 项目：基于深度学习的无人机航拍车辆识别方法设计
> 主线模型：YOLO11s（Ultralytics 8.4.39，torch 2.7.1+cu118）
> 目标数据：UAVDT（主） / VisDrone 车辆子集（辅）
>
> 文档覆盖范围：本仓库从 baseline 到 PVRP+SNAA+Lite 全栈的所有实际已完成工程改动，以及复现每一个实验所需的命令、消融顺序、论文写作建议与尚未完成事项。**本文不包含任何 mAP/AP 数值或性能增益结论**——所有检测指标必须由你在目标硬件 + 目标数据上实测得到。

---

## 1. 仓库结构分析（与本项目相关的部分）

| 类别 | 文件 | 作用 |
|---|---|---|
| 模型构建工厂 | [ultralytics/nn/tasks.py](ultralytics/nn/tasks.py) | `parse_model`（L1539+）、`BaseModel`、`DetectionModel`、`init_criterion`（L512） |
| YOLO11 基线 yaml | [ultralytics/cfg/models/11/yolo11.yaml](ultralytics/cfg/models/11/yolo11.yaml) | 原生 yolo11 网络定义，5 级 scale |
| 基础模块库 | [ultralytics/nn/modules/block.py](ultralytics/nn/modules/block.py) | `C3k2 / SPPF / C2PSA / DFL` 等 |
| 卷积 / concat 库 | [ultralytics/nn/modules/conv.py](ultralytics/nn/modules/conv.py) | `Conv / DWConv / Concat / GhostConv` 等 |
| 检测头 | [ultralytics/nn/modules/head.py](ultralytics/nn/modules/head.py) | `Detect`（L26）、`OBB`、DFL 解码 |
| 损失 | [ultralytics/utils/loss.py](ultralytics/utils/loss.py) | `BboxLoss`（L109）、`v8DetectionLoss`（L333）、`bbox2dist` |
| 分配器 | [ultralytics/utils/tal.py](ultralytics/utils/tal.py) | `TaskAlignedAssigner`（L14）；返回 `target_gt_idx` |
| IoU / 度量 | [ultralytics/utils/metrics.py](ultralytics/utils/metrics.py) | `bbox_iou`（L81，CIoU/DIoU/GIoU） |
| 默认超参 | [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml) | `box/cls/dfl` gain 等 |
| 超参类型验证 | [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py) | `CFG_BOOL_KEYS / CFG_FLOAT_KEYS` |
| 训练入口 | [ultralytics/models/yolo/detect/train.py](ultralytics/models/yolo/detect/train.py) | `DetectionTrainer.get_model` 加载 pretrained |
| 顶层 API | [ultralytics/engine/model.py](ultralytics/engine/model.py) | `YOLO`、`Model.load()`（L348 partial-load） |

---

## 2. 研究方案映射

研究报告的创新点 → 本仓库实际落点：

| 研究创新 | 工程实现 | 代码位置 |
|---|---|---|
| **主创新** 小点①：P2 代理细节支路 | `P2Proxy` 类（C3k2-Lite 风格，split-transform-merge） | [pvrp.py:29-66](ultralytics/nn/modules/pvrp.py#L29-L66) |
| **主创新** 小点②：抗混叠语义回灌融合 | `AntiAliasDown`（Binomial 低通 + stride-2）+ `SemanticGatedFuse`（channel/spatial 双门控） | [pvrp.py:69-101](ultralytics/nn/modules/pvrp.py#L69-L101) + [pvrp.py:104-148](ultralytics/nn/modules/pvrp.py#L104-L148) |
| **主创新** 小点③：近邻车辆解耦头适配器 | `NeighborDecoupleAdapter`（DoG 风格局部对比，残差 gating） | [pvrp.py:151-181](ultralytics/nn/modules/pvrp.py#L151-L181) |
| **损失** SNAA（Scale-Neighbor Aware Attraction） | `SNAABboxLoss`（继承 `BboxLoss`，只加权 IoU 项；α·ρ 乘法耦合） | [loss.py:163-340](ultralytics/utils/loss.py#L163-L340) |
| **副创新** 小点④：P4/P5 Neck 选择性瘦身 | YAML 级改 C3k2 `e=0.25`（仅 P4/P5 分支） | `yolo11-pvrp-lite*.yaml` |
| **副创新** 小点⑤：通道重分配给 P3 | YAML 级改 P3 FPN C3k2 `e=0.75` | `yolo11-pvrp-lite.yaml / -s5.yaml` |
| baseline / 消融开关 | 独立 YAML + `hyp.snaa` 开关 | 见 §8 运行命令 |

### 工程上的研究方案偏差（已知且已说明）

1. **"仅小点 2"无法独立存在** —— SGF 的低层输入必须来自小点 1 的 P2 代理。工程上把"仅小点 2"映射为 `s12`（1+2），"小点 2+3"映射为完整 PVRP。已在 Stage 2-A 报告 §6 说明。
2. **OBB 路径未接 SNAA** —— 研究报告未要求 OBB；当前 `SNAABboxLoss` 只走 HBB 分支；`RotatedBboxLoss` 通过 `**kwargs` 向后兼容，但不会触发 SNAA 权重。后续如需 OBB 版本是独立工作。

---

## 3. 实施计划（实际执行轨迹）

| 阶段 | 任务 | 产出 |
|---|---|---|
| Stage 1 | 仓库审查 + 实施计划 + 风险识别 | 文件定位、模块注册点、SNAA 接入点确认 |
| Stage 2-A | 主创新 4 个模块 + 6 个 YAML + 模块注册 | pvrp.py + 5 新 yaml + __init__.py/tasks.py 注册 |
| Stage 2-B | SNAABboxLoss + 3 个 Lite YAML + cfg 验证 | loss.py 新类 + default.yaml/__init__.py 扩展 + 3 Lite yaml |
| Stage 3 | 系统级自检（train/val/predict/export） | 26/26 检查通过；未触发修复 |
| Stage 4-A | auto_train_all.py + bench/summarize 脚本 | 13 实验矩阵（Table 1-5）+ FPS 基准 + 结果聚合 |
| **Stage 4-B（当前）** | 论文工程说明文档 + 交付整理 | 本文档 |

**未执行到的阶段**（标记为待补）：
- UAVDT 数据集 yaml + MOT → YOLO 转换脚本
- VisDrone 车辆子集（car/van/truck/bus）过滤脚本
- 按 altitude / weather / occlusion 的子属性分层评估
- 按 COCO small/medium/large 的 AP 分解
- 任何实际训练与 mAP 指标

---

## 4. 风险评估（截至交付时）

### 🔴 高风险

| 风险 | 应对（已实现） |
|---|---|
| SNAA 收敛稳定性 | kappa/beta 支持独立关闭；`alpha_max=4.0` 上限 clip；`k=0,b=0` 严格回退 baseline（已在 Stage 2-B 验证 atol=1e-6 一致） |
| SNAA 正样本与 padding GT 污染 | 严格用 `fg_mask + mask_gt` 双过滤；`fg_mask.any()=False` 时跳过整个 SNAA 路径 |

### 🟡 中风险

| 风险 | 应对 |
|---|---|
| P2 代理支路显存爆涨 | 代理通道设为 128 → 实际 64（scale s 后），`e=0.5`，全分支参数量约 16k |
| pretrained 加载失配 | 所有 PVRP 配置 backbone 完整保留（C-check 40-55% 匹配，符合预期），`intersect_dicts` 自动跳过 shape 不匹配项 |
| FLOPs 超预算 | 实测 full PVRP +11.1%（scale s, imgsz=640），低于预设 +15% 阈值 |
| Lite 在 batch=1 场景 FPS 反而下降 | 观察为真（本机 RTX 3060 Laptop：pvrp-lite 53 FPS < baseline 98 FPS，因小 conv 的 kernel launch 开销）；在 batch≥8 / FP16 / TensorRT 部署时会改观。**论文 FPS 请在目标硬件重测** |

### 🟢 低风险

| 风险 | 应对 |
|---|---|
| YAML 层索引漂移 | 6 配置全部显式整数索引，parse_model 自动校验；Stage 3 构建全通 |
| ONNX 导出兼容 | Stage 3 已冒烟：pvrp-full / pvrp-lite 都能导出（opset 12） |
| cfg 新 key 验证 | 5 个 SNAA key 已注册到 `CFG_BOOL_KEYS / CFG_FLOAT_KEYS`；`get_cfg(overrides=...)` 验证通过 |

---

## 5. 开始修改代码（已完成的文件清单）

### 新增（12 个）

| 路径 | 用途 |
|---|---|
| [ultralytics/nn/modules/pvrp.py](ultralytics/nn/modules/pvrp.py) | 主创新 4 个模块：P2Proxy / AntiAliasDown / SemanticGatedFuse / NeighborDecoupleAdapter |
| [ultralytics/cfg/models/11/yolo11-pvrp.yaml](ultralytics/cfg/models/11/yolo11-pvrp.yaml) | 主创新完整（1+2+3） |
| [ultralytics/cfg/models/11/yolo11-pvrp-s1.yaml](ultralytics/cfg/models/11/yolo11-pvrp-s1.yaml) | 仅小点 1（P2 代理 + 朴素 concat） |
| [ultralytics/cfg/models/11/yolo11-pvrp-s3.yaml](ultralytics/cfg/models/11/yolo11-pvrp-s3.yaml) | 仅小点 3（NDA） |
| [ultralytics/cfg/models/11/yolo11-pvrp-s12.yaml](ultralytics/cfg/models/11/yolo11-pvrp-s12.yaml) | 小点 1+2（P2 代理 + SGF） |
| [ultralytics/cfg/models/11/yolo11-pvrp-s13.yaml](ultralytics/cfg/models/11/yolo11-pvrp-s13.yaml) | 小点 1+3 |
| [ultralytics/cfg/models/11/yolo11-pvrp-lite.yaml](ultralytics/cfg/models/11/yolo11-pvrp-lite.yaml) | 主 + 副创新完整（主 + slim + realloc） |
| [ultralytics/cfg/models/11/yolo11-pvrp-lite-s4.yaml](ultralytics/cfg/models/11/yolo11-pvrp-lite-s4.yaml) | 副创新小点 4 独立（仅 slim P4/P5） |
| [ultralytics/cfg/models/11/yolo11-pvrp-lite-s5.yaml](ultralytics/cfg/models/11/yolo11-pvrp-lite-s5.yaml) | 副创新小点 5 独立（仅 P3 realloc） |
| [auto_train_all.py](auto_train_all.py) | 批量实验 orchestrator（13 组，Table 1-5 分组）；套用用户 train.py 格式 |
| [scripts/bench_speed.py](scripts/bench_speed.py) | 独立 FPS / latency / params / FLOPs 基准脚本 |
| [scripts/summarize_results.py](scripts/summarize_results.py) | 递归扫 `runs/**/results.csv`，按最佳 mAP50-95 聚合为 Markdown 表 |

### 修改（5 个，最小 diff）

| 路径 | 修改点 |
|---|---|
| [ultralytics/nn/modules/__init__.py](ultralytics/nn/modules/__init__.py) | 顶部 `from .pvrp import ...`；`__all__` 增 `AntiAliasDown / NeighborDecoupleAdapter / P2Proxy / SemanticGatedFuse` |
| [ultralytics/nn/tasks.py](ultralytics/nn/tasks.py) | ①顶部 import 4 个模块；②`base_modules` 增 `P2Proxy / AntiAliasDown / NeighborDecoupleAdapter`；③`repeat_modules` 增 `P2Proxy`；④`parse_model` 加 `elif m is SemanticGatedFuse:` 分支处理双输入通道推断 |
| [ultralytics/utils/loss.py](ultralytics/utils/loss.py) | ①`BboxLoss.forward` 加 `**kwargs`（向后兼容）；②新增 `SNAABboxLoss` 类（约 160 行）；③`v8DetectionLoss.__init__` 按 `hyp.snaa` 切换 `BboxLoss / SNAABboxLoss`；④`get_assigned_targets_and_loss` 透传 `gt_labels / gt_bboxes / target_gt_idx / mask_gt` |
| [ultralytics/cfg/default.yaml](ultralytics/cfg/default.yaml) | 追加 5 个 SNAA 键：`snaa / snaa_kappa / snaa_tau / snaa_beta / snaa_alpha_max`，默认 `snaa: False` 保持 baseline |
| [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py) | `CFG_BOOL_KEYS` 加 `snaa`；`CFG_FLOAT_KEYS` 加 `snaa_kappa / snaa_tau / snaa_beta / snaa_alpha_max` |

**总计：12 新增 + 5 修改**。

### 关键 diff 要点

**① SNAA 损失（最核心的单项改动）** —— [loss.py](ultralytics/utils/loss.py)：

```python
# 公式：
#   alpha_i = clamp(1 + kappa * log(s_ref / s_i), 1, alpha_max)
#   rho_i   = 1 + beta * exp(-tau * (d_i / s_i)^2)
#   L_iou   = Σ_i [ cls_weight_i * alpha_i * rho_i * (1 - CIoU_i) ] / Z
#
# 数值稳定：
#   - sqrt(wh) 前 clamp_min(eps)
#   - log 前 +eps
#   - (d/s)^2 前 clamp_max(1e6)
#   - alpha 双端 clamp [1, alpha_max]
#   - kappa=0 / beta=0 完全短路回退 CIoU
#
# 影响链条：
#   - 正负样本分配（TaskAlignedAssigner）：不影响
#   - cls 分支 BCE：不影响
#   - DFL 分支：不影响
#   - 回归 CIoU：仅权重，不改 IoU 定义
#   - 推理路径：不影响（loss 仅训练期生效）
```

**② parse_model 双输入分支** —— [tasks.py](ultralytics/nn/tasks.py)：

```python
elif m is SemanticGatedFuse:
    c2 = args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [ch[f[0]], ch[f[1]], c2, *args[1:]]
```

**③ YAML 级消融**：所有 YAML 共享同一 backbone，只改 head 引用 + C3k2 的 `e` 参数，scale 全兼容（n/s/m/l/x）。

---

## 6. 修改说明（分阶段）

### Stage 2-A：主创新 4 模块 + 6 个 YAML

- `P2Proxy(c1, c2, n=1, shortcut=False, e=0.5)` — 继承 `nn.Module`，split-transform-merge；接入 `repeat_modules`
- `AntiAliasDown(c1, c2, blur_k=3)` — 固定 Binomial kernel 注册为 non-persistent buffer，depthwise 先 blur 再 stride-2 conv
- `SemanticGatedFuse(c_low, c_high, c_out, reduction=4)` — 双输入 list forward；`gate_conv → channel_gate + spatial_gate → low*gate → concat + 1x1`
- `NeighborDecoupleAdapter(c1, c2, reduction=4)` — `reduce → local(3x3) - context(5x5) → 1x1 sigmoid → x*(1+gate) → 1x1 out`
- YAML：6 个（baseline 复用原 yolo11.yaml + 5 个新建：s1/s3/s12/s13/full）

**Stage 2-A 自检**（Stage 2-A 报告 §7）：
- 单元 shape：4 模块 forward 正确，总参数 ~140k
- 全模型构建（scale s）：所有 6 配置 eval/train 前向通过，输出 `(2, 84, 8400)`
- 参数/FLOPs：full PVRP 9.599M / 24.13 GFLOPs（vs baseline 9.459M / 21.72 GFLOPs，+1.5% / +11.1%）
- pretrained yolo11s.pt 加载：pvrp-full 240/571 keys（42%，backbone 完整）

### Stage 2-B：SNAA + 3 个 Lite YAML + 配置

- `SNAABboxLoss(BboxLoss)` 继承；5 个超参；静态方法 `_nearest_same_class_dist`（O(B·N²)，N_gt 通常 <100）；`_compute_weights` 返回 `(alpha, rho)` per-fg-anchor
- 接入点：`v8DetectionLoss.__init__` 按 `getattr(h, "snaa", False)` 切换；`get_assigned_targets_and_loss` 透传 GT 上下文到 `self.bbox_loss(..., **kwargs)`
- Lite YAML 3 个：`-lite`（slim+realloc 全开）、`-lite-s4`（仅 slim）、`-lite-s5`（仅 realloc）
- default.yaml 追加 5 个键（默认 `snaa: False` 保持 baseline 行为）
- cfg/__init__.py 将新键注册到类型验证 frozenset

**Stage 2-B 自检**（Stage 2-B 报告 §4）：
- 等价性：`SNAA(k=0, b=0).forward(...)` 与 `BboxLoss.forward(...)` 完全一致（atol=1e-6）
- 梯度稳定：默认超参下梯度有限，无 NaN/Inf
- 边界：仅 1 GT/图 → ρ=1 回退 baseline
- 9 配置（baseline + 5 pvrp + 3 lite）构建 + loss 计算全通（含 SNAA on/off）
- cfg 系统：5 个 SNAA key 经 `get_cfg(overrides=...)` 一致回转

### Stage 3：系统级自检

**26 / 26 检查全通**（Stage 3 报告 §1）：

| 类别 | 项目 | 通过 |
|---|---|---|
| A | 导入完整性 | 1/1 |
| B | 9 个 YAML 构建（scale s） | 9/9 |
| C | pretrained 匹配率 | 9/9 |
| D | ONNX 导出冒烟（pvrp-full / pvrp-lite） | 2/2 |
| E | coco8 train+val+predict 端到端（baseline / pvrp-full / pvrp-full+SNAA / pvrp-lite+SNAA） | 4/4 |
| F | cfg 系统 SNAA key 回转 | 1/1 |

**未触发任何修复**（本阶段没有 bug，也没有需要手动测试的遗留项）。

### Stage 4-A：实验工具

- `auto_train_all.py`：13 实验（Table 1-5 分组），默认解注释 Tier 0 的 3 条（baseline / pvrp / pvrp-lite-snaa-full），其余用 `# {...}` 注释；套用用户现有 train.py 的风格（UAVDT / epochs=200 / imgsz=1024 / AdamW / aug / amp=False / patience=0）
- `scripts/bench_speed.py`：独立 FPS/latency/params/FLOPs 基准，支持 warmup + CUDA sync；本机 RTX 3060 Laptop 实测通过
- `scripts/summarize_results.py`：递归找 `results.csv`，取每轮最佳 mAP50-95 行，输出 Markdown 表，支持 merge speed.csv

**Stage 4-A 自检**（Stage 4-A 报告 §8）：3 个脚本均真实 smoke test 通过（非 dry-run）。

---

## 7. 自检结果（汇总，均为实际执行）

| 项 | 测试时机 | 结果 |
|---|---|---|
| 4 模块单元 shape + forward | Stage 2-A | ✅ 所有模块参数、输入输出形状正确 |
| 6 配置构建 + eval 前向 | Stage 2-A | ✅ 输出 shape `(2, 84, 8400)` |
| 参数 / FLOPs 预算 | Stage 2-A | ✅ full PVRP +1.5% params / +11.1% FLOPs，lite −12.4% params |
| SNAA k=0,b=0 等价 CIoU | Stage 2-B | ✅ atol=1e-6 一致 |
| SNAA 梯度稳定 | Stage 2-B | ✅ 无 NaN/Inf |
| SNAA 边界（1 GT/图） | Stage 2-B | ✅ 回退 ρ=1，loss 有限 |
| 9 配置 × SNAA 开关的 loss 计算 | Stage 2-B | ✅ 18 组全通 |
| cfg 系统新 key 验证 | Stage 2-B / Stage 3 | ✅ 5 个 SNAA 键类型验证通过 |
| pretrained 加载匹配率 | Stage 3 | ✅ baseline 100%，PVRP 42-55%（backbone 完整保留） |
| ONNX 导出 | Stage 3 | ✅ pvrp-full / pvrp-lite 成功导出（opset 12, imgsz 320） |
| coco8 端到端（train+val+predict） | Stage 3 | ✅ 4 组（baseline / pvrp / pvrp+SNAA / pvrp-lite+SNAA）全通 |
| bench_speed FPS 实测 | Stage 4-A | ✅ 3 配置实测（见 §7.1） |
| auto_train_all smoke | Stage 4-A | ✅ coco8 1-epoch baseline 执行成功 |
| summarize_results | Stage 4-A | ✅ 正确读取 results.csv 聚合为 Markdown |

### 7.1 真实测量的参数 / FLOPs / FPS

| 配置 | Params (M) | FLOPs (G) @640 | FPS @640, bs=1 | 来源 |
|---|---|---|---|---|
| yolo11s（baseline） | 9.459 | 21.72 | 98.4 | Stage 4-A bench（RTX 3060 Laptop, FP32） |
| yolo11s-pvrp | 9.599 | 24.13 | 78.4 | 同上 |
| yolo11s-pvrp-lite | 8.287 | 23.05 | 53.0 | 同上 |
| yolo11s-pvrp-s1 | 9.520 | 23.14 | — | Stage 2-B FLOPs |
| yolo11s-pvrp-s3 | 9.515 | 22.44 | — | Stage 2-B FLOPs |
| yolo11s-pvrp-s12 | 9.543 | 23.41 | — | Stage 2-B FLOPs |
| yolo11s-pvrp-s13 | 9.576 | 23.86 | — | Stage 2-B FLOPs |
| yolo11s-pvrp-lite-s4 | 8.195 | 21.88 | — | Stage 2-B FLOPs |
| yolo11s-pvrp-lite-s5 | 9.690 | 25.30 | — | Stage 2-B FLOPs |

> ⚠️ **FPS 数字仅反映 RTX 3060 Laptop + FP32 + batch=1 + imgsz=640 的实测情况**。Lite 在 batch=1 反而比 baseline 慢是已知现象（小 conv kernel launch 开销主导），**论文最终 FPS 请在目标硬件 + 目标 batch + 可选 FP16/TensorRT 上重测**，并在文中注明测试条件。

---

## 8. 运行命令

### 8.1 环境

```bash
# Python 3.11, torch 2.7.1+cu118, ultralytics 8.4.39
# 激活虚拟环境：
D:/miniconda/envs/yolo11/python.exe    # 直接指定解释器
```

### 8.2 单次实验命令（论文主表每一行对应的训练命令）

所有命令默认假设 `UAVDT.yaml`（**尚未创建，Stage 4-B 待补**）。如先用 VisDrone 或 coco8 验证链路，替换 `data=` 参数即可。

```bash
# (0) baseline
yolo detect train model=yolo11s.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 \
    optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp01_Baseline

# (1) 主创新完整（无 SNAA，无 Lite）
yolo detect train model=yolo11s-pvrp.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 \
    optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp02_PVRP_Main

# (2) 主 + Loss（SNAA）
yolo detect train model=yolo11s-pvrp.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 snaa=True \
    optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp04_PVRP_SNAA

# (3) 主 + 副（Lite，无 SNAA）
yolo detect train model=yolo11s-pvrp-lite.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 \
    optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp05_PVRP_Lite

# (4) 主 + 副 + Loss 全开（论文主模型）
yolo detect train model=yolo11s-pvrp-lite.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 snaa=True \
    optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp03_PVRP_Lite_SNAA_Full

# (5) 主创新子模块消融
yolo detect train model=yolo11s-pvrp-s1.yaml  data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp06_PVRP_S1_Only
yolo detect train model=yolo11s-pvrp-s3.yaml  data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp07_PVRP_S3_Only
yolo detect train model=yolo11s-pvrp-s12.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp08_PVRP_S12
yolo detect train model=yolo11s-pvrp-s13.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp09_PVRP_S13

# (6) 副创新子模块消融
yolo detect train model=yolo11s-pvrp-lite-s4.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp10_Lite_S4_SlimOnly
yolo detect train model=yolo11s-pvrp-lite-s5.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 name=Exp11_Lite_S5_ReallocOnly

# (7) SNAA 内部项消融
yolo detect train model=yolo11s-pvrp.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 snaa=True snaa_beta=0.0  name=Exp12_SNAA_ScaleOnly
yolo detect train model=yolo11s-pvrp.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 snaa=True snaa_kappa=0.0 name=Exp13_SNAA_NeighborOnly
```

### 8.3 批量跑（推荐）

默认只跑 Tier 0 的 3 条核心实验；如需其他，**编辑 [auto_train_all.py](auto_train_all.py) 取消对应注释即可**：

```bash
D:/miniconda/envs/yolo11/python.exe auto_train_all.py
```

### 8.4 测速 / 聚合

```bash
# 速度基准
D:/miniconda/envs/yolo11/python.exe scripts/bench_speed.py runs/detect/PVRP/Exp03_PVRP_Lite_SNAA_Full/weights/best.pt --device 0

# FP16（部署接近值）
D:/miniconda/envs/yolo11/python.exe scripts/bench_speed.py runs/detect/PVRP/Exp03_PVRP_Lite_SNAA_Full/weights/best.pt --device 0 --half

# 结果聚合
D:/miniconda/envs/yolo11/python.exe scripts/summarize_results.py --root runs/detect --sort-by mAP --output docs/main_table.md
```

---

## 9. 消融建议

### 9.1 推荐训练顺序

**阶段 ①（核心 3 条，必跑）**
1. `Exp01_Baseline`（yolo11.yaml）
2. `Exp02_PVRP_Main`（yolo11-pvrp.yaml，snaa=False）
3. `Exp03_PVRP_Lite_SNAA_Full`（yolo11-pvrp-lite.yaml，snaa=True）

→ 可支撑论文 Abstract、Table 1（主对比）、Figure "method overview"。

**阶段 ②（推荐，再 2 条）**
4. `Exp04_PVRP_SNAA`（衡量 SNAA 单项增益）
5. `Exp05_PVRP_Lite`（衡量 Lite 在无 SNAA 下的代价 / 增益）

→ 阶段 ② 完成后，可写"三大创新每一项单独提升 X / 累积提升 Y"的论文叙事。

**阶段 ③（主创新消融 4 条）**
6-9. `Exp06-09`（s1 / s3 / s12 / s13）

→ 支撑论文 Ablation Table A："主创新三小点都必要"。

**阶段 ④（副创新 + SNAA 内部消融 4 条）**
10-11. `Exp10-11`（Lite slim only / realloc only）
12-13. `Exp12-13`（SNAA scale only / neighbor only）

→ 支撑论文 Ablation Table B / C：副创新小点独立贡献；SNAA 两项独立贡献。

### 9.2 时间紧迫的删减策略

| 可用时间 | 优先跑 | 删减 |
|---|---|---|
| 极短（仅 1 卡 1 天） | Exp01, Exp03 | 所有消融；只靠主对比讲故事 |
| 紧（3-4 天） | Exp01, Exp02, Exp03 | Tier 1-4 消融 |
| 正常（1 周） | Exp01-05 | Tier 3-4 消融（主创新消融可保） |
| 充裕 | 全部 13 条 | — |

### 9.3 快速判断模块价值的方法

**方法 A：10-epoch 预筛（推荐）**
把 Exp 跑 10 epoch（imgsz=320 或 640），看 mAP50-95：
- 比 baseline **低超过 2 个绝对点** → 有隐患，检查实现
- 比 baseline **高 0.5 以上** → 有潜力，投 100-200 epoch
- **与 baseline 差 ±0.3 之内** → 看 `train/box_loss` 前 3 epoch 下降速率：更快则继续、更慢则放弃

**方法 B：损失曲线前 3 epoch 斜率**
`results.csv` 第 0-3 行的 `train/box_loss` 下降速率，能早期判断是否提供有效梯度。

**方法 C：速度 / 精度折中率**
若 `(mAP - mAP_baseline) / (latency - latency_baseline) ≥ 0.5`，模块值得写入论文。

### 9.4 结果不好时的删减优先级

如果 `Exp03_PVRP_Lite_SNAA_Full` 的 mAP 比 `Exp01_Baseline` 低：

1. **先关 SNAA**（`snaa=False`），跑 Exp03 的等价无 SNAA 版本 → 隔离 loss 影响
2. **再去掉 Lite**（换到 `Exp02_PVRP_Main`）→ 如果这里变好说明 Lite 瘦身过头
3. **最后单独测主创新 3 小点**（Exp06/08/09）→ 若全都不优于 baseline，怀疑 FPN 层索引错配（重检 yaml）

如果是 FPS 负优化：
1. 先关 Lite（Lite 的 `e=0.25` 会增加层数）
2. 次关 NDA（p3 头前的 extra 算子）
3. 最后关 SGF（门控 + concat 最耗时）

---

## 10. 论文工程说明

### 10.1 应记录的指标

| 类别 | 指标 | 来源 |
|---|---|---|
| 检测精度 | mAP50, mAP50-95, P, R | `model.val()` 自动输出，写入 `results.csv` |
| 小目标精度（推荐） | AP_small, AP_medium, AP_large | 需 `val(save_json=True)` + pycocotools；UAVDT 需额外按 GT 面积分组 |
| 类别精度 | 每类 AP（car, van, truck, bus） | `val()` 默认打印 per-class；记录 car 和 bus 的 gap |
| 资源 | Params (M), FLOPs (G) | `model.info()` / `get_flops()` |
| 速度 | latency (ms), FPS | `scripts/bench_speed.py`（注明硬件与 batch） |
| 训练代价 | 收敛 epoch、总训练时长 | `results.csv` 最佳 epoch + wall-clock |

### 10.2 建议的表结构

**论文主表（Table 1，对比 baseline 和主模型）**

| Method | Params (M) | FLOPs (G) | FPS | mAP50 | mAP50-95 | AP_small | AP_med | AP_large |
|---|---|---|---|---|---|---|---|---|
| YOLO11s (baseline) | 9.46 | 21.72 | — | — | — | — | — | — |
| +PVRP (main) | 9.60 | 24.13 | — | — | — | — | — | — |
| +PVRP+SNAA | 9.60 | 24.13 | — | — | — | — | — | — |
| +PVRP-Lite | 8.29 | 23.05 | — | — | — | — | — | — |
| **+PVRP-Lite+SNAA (ours)** | **8.29** | **23.05** | — | — | — | — | — | — |

**消融表 A（主创新三小点）**

| Variant | ① P2-proxy | ② SGF | ③ NDA | Params (M) | mAP50-95 |
|---|---|---|---|---|---|
| Exp01 baseline | ✗ | ✗ | ✗ | 9.46 | — |
| Exp06 s1 | ✓ | ✗ | ✗ | 9.52 | — |
| Exp07 s3 | ✗ | ✗ | ✓ | 9.52 | — |
| Exp08 s12 | ✓ | ✓ | ✗ | 9.54 | — |
| Exp09 s13 | ✓ | ✗ | ✓ | 9.58 | — |
| Exp02 full | ✓ | ✓ | ✓ | 9.60 | — |

**消融表 B（副创新 Lite 两小点）**

| Variant | ④ slim P4/P5 | ⑤ realloc P3 | Params (M) | FLOPs (G) | mAP50-95 |
|---|---|---|---|---|---|
| Exp02 PVRP main | ✗ | ✗ | 9.60 | 24.13 | — |
| Exp10 lite-s4 | ✓ | ✗ | 8.20 | 21.88 | — |
| Exp11 lite-s5 | ✗ | ✓ | 9.69 | 25.30 | — |
| Exp05 PVRP-Lite | ✓ | ✓ | 8.29 | 23.05 | — |

**消融表 C（SNAA 两项）**

| Variant | scale (α) | neighbor (ρ) | mAP50-95 |
|---|---|---|---|
| Exp02 no SNAA | — | — | — |
| Exp12 scale-only | ✓ | ✗ | — |
| Exp13 neighbor-only | ✗ | ✓ | — |
| Exp04 full SNAA | ✓ | ✓ | — |

### 10.3 放论文主表的选择建议

- **Table 1（主对比）**：Exp01 + Exp02 + Exp04 + Exp05 + Exp03 共 5 行，以 "ours" = Exp03
- **Table 2（消融主创新）**：Exp01 / 06 / 07 / 08 / 09 / 02 共 6 行
- **Table 3（消融副创新）**：Exp02 / 10 / 11 / 05 共 4 行
- **Table 4（消融 SNAA）**：Exp02 / 12 / 13 / 04 共 4 行

**SOTA 对比表**（可选）：若有计算资源，把 Exp03 和公开 SOTA（YOLOv8s/YOLOv10s/RT-DETR-R18 等）放一起；仓库里已有 `ultralytics/cfg/models/26/yolo26.yaml`、`rt-detr/rtdetr-l.yaml` 等可直接跑对比。

### 10.4 论文叙事结构建议

1. **Introduction** 强调 UAV 车辆 3 大痛点：小目标、密集、尺度变化 → 对应主创新 3 小点
2. **Method**：
   - 3.1 PVRP 总体架构（用 Figure 展示 backbone→P2Proxy→AntiAliasDown→SGF→NDA→Detect）
   - 3.2 P2-Proxy 细节支路（小点 1）
   - 3.3 Anti-aliased Semantic-Gated Fusion（小点 2）
   - 3.4 Neighbor-Decoupling Adapter（小点 3）
   - 3.5 SNAA Loss（附尺度项与近邻项的公式）
   - 3.6 Selective Slim & Realloc（副创新两小点）
3. **Experiments**：
   - Table 1 主对比
   - Table 2/3/4 消融
   - Figure 收敛曲线（`results.csv` 的 box_loss + mAP50-95）
   - Figure 可视化（`predict` 输出的边界框对比图）

---

## 11. 剩余风险与下一步建议

### 尚未完成（需要下一轮工作补齐）

| 项 | 说明 | 阻塞程度 |
|---|---|---|
| UAVDT 数据集 YAML + MOT→YOLO 转换脚本 | 运行 `Exp01-13` 的必要前置 | 🔴 高，必须做 |
| VisDrone 车辆子集过滤脚本 | 辅助数据集用于跨域验证 | 🟡 中 |
| 按 altitude / weather / occlusion 分层评估 | 支撑论文 "我们在复杂 UAV 场景下更鲁棒" 的叙事 | 🟡 中 |
| 按 COCO small/medium/large 的 AP 分解 | 支撑小目标 claim | 🟡 中 |
| `DroneVehicle` OBB 数据集扩展 | 研究报告提到的可选 OBB 路线 | 🟢 低，不影响主线 |
| 任何实际 mAP / AP 结果 | 论文实验表填充 | 🔴 高（依赖数据集 yaml） |

### 建议的下一步工作顺序

1. **第一优先**：让 [auto_train_all.py](auto_train_all.py) 在 `UAVDT.yaml` 上真正能跑
   - 写 `ultralytics/cfg/datasets/UAVDT.yaml`
   - 写 `scripts/uavdt2yolo.py`（MOT 格式 → YOLO 每图 txt）
   - 按官方 UAVDT 划分 `M0101..M0401` 训练集 / `M0701..M1401` 验证集
2. **第二优先**：跑 Tier 0 的 3 条核心实验（Exp01/02/03）共 3 × ~4-8h = 1-2 天
3. **第三优先**：用 `scripts/bench_speed.py` 在你论文所用的"部署硬件"上测 FPS
4. **第四优先**：扩展 VisDrone 车辆子集作为跨域泛化验证
5. **第五优先**：写 `scripts/eval_by_size.py` 支撑小目标 AP claim

### 最终风险提醒

- **没有任何实验精度结果已经测出**。本文档中所有带 "—" 的指标必须由你在 UAVDT（或 coco8/VisDrone）上训练后填充，**严禁编造**
- **SNAA 超参默认值温和**（kappa=1.0, beta=0.5），若首次训练观察到 box_loss 不稳定，先尝试 `snaa_kappa=0.5, snaa_beta=0.25`（各降一半）
- **Lite 方案在小 batch 部署下 FPS 反而下降是已知现象**（本机 RTX 3060 Laptop 实测 53 FPS vs baseline 98 FPS）；需要在论文的目标部署硬件（如 Jetson Orin / RTX 4090 / 云 GPU 等）+ 目标 batch + 可选 FP16/TensorRT 上重测，并在正文注明"测试条件"
- **预训练权重仅 backbone 可加载（PVRP ~42-55%）**，neck/head 随机初始化，训练初期收敛略慢于 baseline。`epochs=200` 是安全下限，若你观察到 mAP 在 epoch 100 之后仍有持续上升，考虑拉长到 300 epoch

---

## 附录 A：快速复现命令速查

```bash
# 1. 激活环境（本机示例）
set PY=D:/miniconda/envs/yolo11/python.exe

# 2. 验证代码链路（coco8，约 2 分钟）
%PY% auto_train_all.py      # 默认跑 Tier 0 的 3 条 coco8 训练（需先把 data=... 改为 coco8.yaml）

# 3. 跑 UAVDT 主表（需先准备 UAVDT.yaml；约 1-2 天 × 3 = 3-6 天）
%PY% auto_train_all.py

# 4. 速度基准（best.pt 出来后）
%PY% scripts/bench_speed.py runs/detect/PVRP/Exp03_PVRP_Lite_SNAA_Full/weights/best.pt --device 0

# 5. 聚合结果为论文表
%PY% scripts/summarize_results.py --root runs/detect --sort-by mAP --output docs/main_table.md

# 6. 单独跑某条消融（以 s13 为例）
yolo detect train model=yolo11s-pvrp-s13.yaml data=UAVDT.yaml epochs=200 imgsz=1024 batch=8 \
     optimizer=AdamW lr0=0.001 patience=0 amp=False device=0 name=Exp09_PVRP_S13
```

## 附录 B：模块开关快查

| 想要实验什么 | model 参数 | overrides 参数 |
|---|---|---|
| baseline | `yolo11s.yaml` | — |
| 主创新完整 | `yolo11s-pvrp.yaml` | — |
| 主 + SNAA | `yolo11s-pvrp.yaml` | `snaa=True` |
| 主 + 副 | `yolo11s-pvrp-lite.yaml` | — |
| 全开 | `yolo11s-pvrp-lite.yaml` | `snaa=True` |
| 主创新仅小点 1 | `yolo11s-pvrp-s1.yaml` | — |
| 主创新仅小点 3 | `yolo11s-pvrp-s3.yaml` | — |
| 主创新小点 1+2 | `yolo11s-pvrp-s12.yaml` | — |
| 主创新小点 1+3 | `yolo11s-pvrp-s13.yaml` | — |
| 副创新仅 slim | `yolo11s-pvrp-lite-s4.yaml` | — |
| 副创新仅 realloc | `yolo11s-pvrp-lite-s5.yaml` | — |
| SNAA 仅 scale 项 | `yolo11s-pvrp.yaml` | `snaa=True snaa_beta=0.0` |
| SNAA 仅 neighbor 项 | `yolo11s-pvrp.yaml` | `snaa=True snaa_kappa=0.0` |

---

*文档生成自 Stage 4-B（2026-04-20）；基于 Stage 1-4A 真实落地与测试。若后续做了新实验或修改，请对应更新本文档。*
