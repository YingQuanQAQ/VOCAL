# Learning to Ground Multi-Agent Communication with Autoencoders

精简后的 GitHub 上传版本，只保留项目的核心研究代码与最关键的说明文件。

## 这个目录适合做什么

- 直接作为一个新的 GitHub 仓库根目录
- 作为论文复现和代码展示版本
- 作为后续继续清理、补充实验说明和结果图的基础版本

## 包含内容

| 路径 | 内容 |
| --- | --- |
| `cifar-game/` | CIFAR Game 环境、模型和训练代码 |
| `marl-grid/env/` | FindGoal / RedBlueDoors 共用环境 |
| `marl-grid/find-goal/` | FindGoal 训练与模型 |
| `marl-grid/red-blue-doors/` | RedBlueDoors 训练与模型 |
| `marl-grid/overcooked/` | Overcooked 包装器与训练代码 |
| `marl-grid/build_vocab.py` | `vocal` 离线语义锚点构建脚本 |
| `marl-grid/vocal_task_specs.py` | 各任务的槽位语义词表 |
| `marl-grid/vqvib_utils.py` | `vqvib` 原型量化模块 |

## 不包含内容

为保证仓库清爽、便于上传和审阅，这个目录已经去掉了：

- `runs/` 训练日志、checkpoint、TensorBoard 文件
- 图片、视频、PDF、草图等展示产物
- 本地模型权重和缓存文件
- 与本地实验环境绑定过强的中间结果

## 方法简介

### `ae-comm`

使用自编码式通信约束学习消息表示，让通信通道既服务于协作任务，也更容易分析。

### `vocal`

在通信学习中加入语言锚点，把消息空间与人工设计的语义槽位词表对齐，增强可解释性。

### `vqvib`

通过向量量化原型层约束通信表示，让不同通信原型更稳定、更紧凑、更容易可视化分析。

## 快速开始

### 1. 初始化 Git 仓库

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. 安装网格环境

```bash
cd marl-grid/env
pip install -e .
```

### 3. 构建 `vocal` 词表嵌入

```bash
cd marl-grid
python build_vocab.py --task all --msg-dim 10
```

### 4. 运行训练

#### FindGoal

```bash
cd marl-grid/find-goal
python train_ae.py --method ae-comm --gpu 0 --set num_workers 8 env_cfg.comm_len 10
python train_ae.py --method vocal --gpu 0 --set num_workers 8 env_cfg.comm_len 10
```

#### RedBlueDoors

```bash
cd marl-grid/red-blue-doors
python train_ae.py --method ae-comm --gpu 0 --set num_workers 8 env_cfg.comm_len 10
python train_ae.py --method vocal --gpu 0 --set num_workers 8 env_cfg.comm_len 10
python train_ae.py --method vqvib --gpu 0 --set num_workers 8 env_cfg.comm_len 10
```

#### Overcooked

```bash
cd marl-grid/overcooked
python train.py --gpu 0 --set num_workers 1
python train_ae.py --method ae-comm --gpu 0 --set num_workers 1 env_cfg.max_steps 400
python train_ae.py --method vocal --gpu 0 --set num_workers 1 env_cfg.max_steps 400
python train_ae.py --method vqvib --gpu 0 --set num_workers 1 env_cfg.max_steps 400
```

## 上传前建议

- 把 README 里的绝对路径继续替换成相对路径或公开依赖地址
- 补一个仓库级别的 `LICENSE`
- 补充结果图、方法框架图和代表性实验结论
- 如果计划让他人复现，建议增加统一的环境安装脚本

## 引用

如果你使用了这个仓库中与 `VOCAL` 方法相关的整理、实现或实验设计，建议引用我的本科毕业论文：

```bibtex
@thesis{he2026vocal,
  title={VOCAL：面向多智能体协作的可解释离散通信方法},
  author={何乔},
  year={2026},
  school={中国地质大学（武汉）},
  type={学士学位论文}
}
```
