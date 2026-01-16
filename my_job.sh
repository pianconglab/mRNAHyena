#!/usr/bin/env bash
###############################################################################
# Slurm GPU 作业模板（PianLab HPC）
# 编写人：刘翯齐 https://github.com/metaphorme
# 日期：2025年12月23日
#
# 【重要说明】
# 1. 登录后【禁止】直接运行 nvidia-smi / 训练脚本 等高性能运算任务
# 2. 所有计算任务【必须】通过 srun / sbatch 提交
# 3. 不会用没关系：只需要改 3 个地方就能跑
#
# 使用方法（最简单）：
#   sbatch my_job.sh
#
# 或调试用（交互模式）：
#   srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=8G --pty bash
#
# 建议复制本文件后再修改，例如：
#   cp slurm_template.sh my_job.sh
#   vim my_job.sh
###############################################################################

########################
# 一、作业基本信息（可改）
########################

#SBATCH --job-name=example_gpu_task   # 作业名字（随便起）
#SBATCH --partition=gpu               # 计算资源池（不要改）
#SBATCH --output=slurm_logs/slurm-%x-%j.out         # 输出日志（%j=作业ID）
#SBATCH --error=slurm_logs/slurm-%x-%j.err          # 错误日志

########################
# 二、资源申请（重点）
########################

#SBATCH --gres=gpu:1                  # 申请 1 张 GPU（1~4）
#SBATCH --cpus-per-task=16            # CPU 线程数（最大48）
#SBATCH --mem=128G                     # 内存（最大256G）
#SBATCH --time=24:00:00               # 最长运行时间（hh:mm:ss）

# 注意：
# - 内存超了会被系统直接杀掉（OOM）
# - GPU / CPU / 内存等只在作业里生效，登录态不生效
# - 超过上限会被 Slurm 拒绝或直接杀掉

########################
# 三、环境准备（按需改）
########################

echo "================ 作业开始 ================"
echo "作业ID:        $SLURM_JOB_ID"
echo "运行节点:      $SLURM_NODELIST"
echo "使用GPU编号:   $CUDA_VISIBLE_DEVICES"
echo "CPU核心数:     $SLURM_CPUS_PER_TASK"
echo "内存申请:      ${SLURM_MEM_PER_NODE:-未指定} MB"
echo "开始时间:      $(date)"
echo "=========================================="

# 如果你用 conda（示例）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mRNA-hyena

# 如果你用 module（示例）
# module load cuda/12.2

########################
# 四、真正的计算命令（你最关心的）
########################

# 示例 1：查看 GPU（作业态下才允许）
nvidia-smi

# 示例 2：运行 Python 程序
python -m evo2.test.test_evo2_generation --model_name evo2_7b

# 示例 3：sleep 模拟任务（测试用）
sleep 10

########################
# 五、结束信息
########################

echo "=========================================="
echo "作业结束时间: $(date)"
echo "作业正常结束"
echo "=========================================="
