# LLM-NSGA-II for CMAPP

本项目按照 `LLM-NSGA-CMAPP_Implementation_Guide.md` 实现：单层染色体 `chi=[v1,...,vN]`，LLM 负责选择/交叉/变异，CMAPP Decoder 负责真实执行评价，NSGA-II 负责 Pareto 排序与环境选择。

## 目录
- `data/`：道路网络数据与集中配置
- `nsga/`：编码、解码器、目标评价、NSGA-II
- `llm/`：Anthropic-compatible 客户端与 Prompt/JSON 解析
- `algorithms/`：`LLM` 纯搜索与 `LLM-NSGA` 求解器
- `experiment/`：三个实验入口
- `results/`：实验输出

## 运行
```bash
python experiment/run_multi_seed.py
python experiment/run_coop_seed.py
python experiment/plot_pathplanning_llm_10task.py
```

## 配置
模型、API Key、种群与迭代次数集中在 `data/config.py`，默认支持环境变量覆盖：
- `LLM_MODEL_NAME`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_NSGA_EXPERIMENT_SEEDS`
- `LLM_NSGA_EXPERIMENT_POPULATION_SIZE`
- `LLM_NSGA_EXPERIMENT_GENERATIONS`
- 其他见 `data/config.py`

实验脚本默认使用 `EXPERIMENT_CONFIG` 的小规模快速配置，便于验证；正式全量实验请设置上述环境变量或修改 `ALGO_CONFIG`。
