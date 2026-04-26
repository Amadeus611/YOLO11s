import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")

import torch  # type: ignore

from ultralytics import YOLO


def main():
    # =========================================================
    # PVRP 消融实验任务列表
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
            "batch": 16,            # ← 降 batch 配合 imgsz=960+AMP
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
            "name": "Exp02_PVRP_Main",
            "snaa": False,
            "batch": 16,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
            "name": "Exp03_PVRP_Lite_SNAA_Full",
            "snaa": True,          # ← 主 + 副 + SNAA 全开（论文主模型）
            "batch": 16,
            "snaa_alpha_max": 6.0, # ← 增强稀有类加权上限
            "snaa_kappa": 1.5,     # ← 增强尺度敏感项
        },

        # =====================================================
        # Table 2: 主 + loss / 主 + 副 对比
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp04_PVRP_SNAA",
        #     "snaa": True,  # 主创新 + SNAA，不加 Lite
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
        #     "name": "Exp05_PVRP_Lite",
        #     "snaa": False,  # 主创新 + Lite，不加 SNAA
        #     "batch": 8,
        # },

        # =====================================================
        # Table 3: 主创新内部子模块消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s1.yaml",
        #     "name": "Exp06_PVRP_S1_Only",
        #     "snaa": False,
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s3.yaml",
        #     "name": "Exp07_PVRP_S3_Only",
        #     "snaa": False,
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s12.yaml",
        #     "name": "Exp08_PVRP_S12",
        #     "snaa": False,
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-s13.yaml",
        #     "name": "Exp09_PVRP_S13",
        #     "snaa": False,
        #     "batch": 8,
        # },

        # =====================================================
        # Table 4: 副创新 Lite 子模块消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite-s4.yaml",
        #     "name": "Exp10_Lite_S4_SlimOnly",
        #     "snaa": False,
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite-s5.yaml",
        #     "name": "Exp11_Lite_S5_ReallocOnly",
        #     "snaa": False,
        #     "batch": 8,
        # },

        # =====================================================
        # Table 5: SNAA 内部项消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp12_SNAA_ScaleOnly",
        #     "snaa": True,
        #     "snaa_beta": 0.0,  # ← 关闭 neighbor 项
        #     "batch": 8,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
        #     "name": "Exp13_SNAA_NeighborOnly",
        #     "snaa": True,
        #     "snaa_kappa": 0.0,  # ← 关闭 scale 项
        #     "batch": 8,
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
        print(f"{'=' * 60}\n")

        # 实例化模型：加载 YAML 结构并级联 .load() 预训练权重
        model = YOLO(exp["yaml"]).load("yolo11s.pt")

        # 组装训练参数（fixed + per-experiment overrides）
        train_kwargs = dict(
            # --- 实验变量参数 ---
            data="UAVDT.yaml",
            epochs=100,             # ← 从 60 缩减：峰值在 epoch 8-10，30 足够
            batch=exp["batch"],
            imgsz=960,             # 无人机微小目标必须使用高分辨率
            name=exp["name"],
            snaa=exp["snaa"],      # SNAA 损失开关
            project="/home/ssssss/1yolo/Ablation_Results",

            # --- 优化器参数 ---
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.001,    # ← 从 0.0005 增大，增强正则化
            patience=0,           # ← 开启早停：10 epoch 无改善则停止
            cos_lr=True,           # ← 启用余弦退火学习率

            # --- 损失函数权重 ---
            cls=1.5,               # ← 从 0.5 增大到 1.5，迫使模型学习 truck/bus
            box=10,               # 回归损失权重（保持默认）
            dfl=1.5,               # 分布焦点损失（保持默认）

            # --- 航拍增强参数（增强版） ---
            mosaic=1.0,
            close_mosaic=15,       # ← 从 5 延后到 10，最后 10 epoch 关 mosaic
            mixup=0.0,            # ← 从 0 开启到 0.15，增强正则化
            copy_paste=0.5,

            # --- 仿生无人机姿态模拟（加强版） ---
            degrees=180.0,          # ← 从 15 增大，模拟更大无人机旋转角度
            scale=0.7,             # ← 从 0.4 增大，模拟更多飞行高度变化
            translate=0.1,         # 模拟画面边缘截断
            fliplr=0.5,            # 左右翻转
            erasing=0.4,           # ← 增大随机擦除概率
            hsv_h=0.015,
            hsv_s=0.6,
            hsv_v=0.4,

            # --- 正则化 ---
            dropout=0.2,           # ← 新增 dropout，抑制过拟合

            # --- 工程化设置 ---
            device=0,
            workers=8,
            val=True,
            plots=True,
            amp=False,              # 开启半精度加速
            cache=False,           # 关闭缓存
        )

        # 仅 Table 5 覆盖 SNAA 内部项
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
