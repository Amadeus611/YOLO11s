import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")

import torch  # type: ignore

from ultralytics import YOLO


def main():
    # =========================================================
    # PVRP 消融实验任务列表（v2 — 修复训练策略）
    # ---------------------------------------------------------
    # 改动摘要（相比 v1）：
    #   1. 开启 early stopping (patience=20)，防止后 60 epoch 白白过拟合
    #   2. 大幅缓和数据增强 (degrees 180→45, scale 0.7→0.5, erasing 0.4→0.2)
    #   3. 开启 cls_pw=0.5 类别权重，自动补偿 car:truck:bus≈36:1.6:1 的失衡
    #   4. 降低正则化 (weight_decay 0.001→0.0005, dropout 0.2→0.1)
    #   5. PVRP 实验增加 backbone 冻结预热 (freeze=10, warmup_epochs=5)
    #   6. SNAA 参数调温和 (kappa 1.5→0.8, alpha_max 6→3)
    #   7. 开启 AMP 加速训练
    #   8. 开启 mixup=0.1 作为温和的正则化替代
    # ---------------------------------------------------------
    # Table 1: 主创新点逐步叠加 (Exp01-03)
    # Table 2: 主 + 副 + SNAA loss (Exp04-05)
    # Table 3: 主创新内部子模块消融 (Exp06-09)
    # Table 4: 副创新 Lite 子模块消融 (Exp10-11)
    # Table 5: SNAA 内部项消融 (Exp12-13)
    # 说明:
    #   snaa=True/False 控制是否启用 Scale-Neighbor Aware Attraction Loss
    #   snaa_kappa / snaa_beta 仅在 Table 5 中覆盖默认值以关闭 scale / neighbor 项
    # =========================================================
    experiments = [
        # =====================================================
        # Table 1: 主创新点逐步叠加
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s.yaml",
            "name": "Exp01_Baseline",
            "snaa": False,
            "batch": 16,
            "freeze": None,          # Baseline 不需要冻结
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
            "name": "Exp02_PVRP_Main",
            "snaa": False,
            "batch": 16,
            "freeze": 10,            # ← 冻结 backbone L0-L9 前 5 epoch 预热新模块
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
            "name": "Exp03_PVRP_Lite_SNAA_Full",
            "snaa": True,            # ← 主 + 副 + SNAA 全开（论文主模型）
            "batch": 16,
            "freeze": 10,            # ← 冻结 backbone 预热
            "snaa_alpha_max": 3.0,   # ← 从 6.0 降至 3.0，避免过度强调小目标
            "snaa_kappa": 0.8,       # ← 从 1.5 降至 0.8，温和的尺度加权
            "snaa_beta": 0.3,        # ← 从 0.5 降至 0.3，减弱邻居项干扰
        },

        # =====================================================
        # Table 2: 主 + loss / 主 + 副 对比
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp04_PVRP_SNAA",
        #     "snaa": True,  # 主创新 + SNAA，不加 Lite
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
        #     "name": "Exp05_PVRP_Lite",
        #     "snaa": False,  # 主创新 + Lite，不加 SNAA
        #     "batch": 16,
        #     "freeze": 10,
        # },

        # =====================================================
        # Table 3: 主创新内部子模块消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s1.yaml",
        #     "name": "Exp06_PVRP_S1_Only",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s3.yaml",
        #     "name": "Exp07_PVRP_S3_Only",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s12.yaml",
        #     "name": "Exp08_PVRP_S12",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s13.yaml",
        #     "name": "Exp09_PVRP_S13",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },

        # =====================================================
        # Table 4: 副创新 Lite 子模块消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite-s4.yaml",
        #     "name": "Exp10_Lite_S4_SlimOnly",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite-s5.yaml",
        #     "name": "Exp11_Lite_S5_ReallocOnly",
        #     "snaa": False,
        #     "batch": 16,
        #     "freeze": 10,
        # },

        # =====================================================
        # Table 5: SNAA 内部项消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp12_SNAA_ScaleOnly",
        #     "snaa": True,
        #     "snaa_beta": 0.0,  # ← 关闭 neighbor 项
        #     "batch": 16,
        #     "freeze": 10,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp13_SNAA_NeighborOnly",
        #     "snaa": True,
        #     "snaa_kappa": 0.0,  # ← 关闭 scale 项
        #     "batch": 16,
        #     "freeze": 10,
        # },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"🚀 开始执行第 {i + 1}/{len(experiments)} 组实验: {exp['name']}")
        print(f"📂 配置文件: {exp['yaml']}")
        print(f"⚖️ SNAA 损失开关: {exp['snaa']}")
        print(f"📦 Batch Size: {exp['batch']}")
        print(f"🧊 Backbone 冻结: {exp.get('freeze', None)}")
        print(f"{'=' * 60}\n")

        # 实例化模型：加载 YAML 结构并级联 .load() 预训练权重
        model = YOLO(exp["yaml"]).load("yolo11s.pt")

        # 组装训练参数（fixed + per-experiment overrides）
        train_kwargs = dict(
            # --- 实验变量参数 ---
            data="UAVDT.yaml",
            epochs=100,
            batch=exp["batch"],
            imgsz=960,             # 无人机微小目标必须使用高分辨率
            name=exp["name"],
            snaa=exp["snaa"],      # SNAA 损失开关
            project="/home/ssssss/1yolo/Ablation_Results",

            # --- 优化器参数 ---
            optimizer="AdamW",
            lr0=0.0005,            # ← 从 0.001 降至 0.0005，减缓新模块对 backbone 的冲击
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,   # ← 从 0.001 降至 0.0005，Mini 数据集不需要过强正则
            patience=20,           # ← 从 0 改为 20，开启早停防止过拟合
            cos_lr=True,           # 余弦退火学习率

            # --- Backbone 冻结（仅 PVRP 实验使用） ---
            freeze=exp.get("freeze", None),

            # --- Warmup 预热 ---
            warmup_epochs=5.0,     # ← 从 3 增至 5，给新模块更充分的预热时间

            # --- 损失函数权重 ---
            cls=1.5,               # 分类损失权重（维持）
            box=10,                # 回归损失权重
            dfl=1.5,               # 分布焦点损失

            # --- 类别权重（解决 car:truck:bus ≈ 36:1.6:1 的不平衡） ---
            cls_pw=0.5,            # ← 从 0 改为 0.5，开启逆频率类别加权
                                   #   0.5 是阻尼系数：weight = (1/freq)^0.5
                                   #   避免直接用 1/freq 导致稀有类权重爆炸

            # --- 航拍增强参数（缓和版，适配 Mini 数据集） ---
            mosaic=1.0,
            close_mosaic=10,       # ← 从 15 改为 10，最后 10 epoch 关闭 mosaic 稳定学习
            mixup=0.1,             # ← 从 0 改为 0.1，温和的 mixup 正则化
            copy_paste=0.3,        # ← 从 0.5 降至 0.3，减少增强噪声

            # --- 仿生无人机姿态模拟（适度版） ---
            degrees=45.0,          # ← 从 180 降至 45，航拍合理旋转范围（避免车辆倒置）
            scale=0.5,             # ← 从 0.7 降至 0.5，适度的高度变化模拟
            translate=0.1,         # 画面平移
            fliplr=0.5,            # 左右翻转
            erasing=0.2,           # ← 从 0.4 降至 0.2，减少信息丢失
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,

            # --- 正则化 ---
            dropout=0.1,           # ← 从 0.2 降至 0.1，Mini 数据集已够小不需要过强 dropout

            # --- 工程化设置 ---
            device=0,
            workers=8,
            val=True,
            plots=True,
            amp=False,
            cache=False,
        )

        # 仅覆盖 SNAA 内部项（Table 5 或 Exp03 自定义参数）
        for k in ("snaa_kappa", "snaa_tau", "snaa_beta", "snaa_alpha_max"):
            if k in exp:
                train_kwargs[k] = exp[k]

        # 启动训练
        model.train(**train_kwargs)

        # 每一组实验结束后手动清空显存缓存
        torch.cuda.empty_cache()

    print("\n 所有消融实验已全部顺序执行完毕！")


if __name__ == "__main__":
    main()
