# 代码考核：基于 LangChain 的“水利法规智能体”

> 一份**可直接落地**的代码考核说明：从目标与评分，到目录结构、运行说明、基线代码骨架与测试用例。

---

## 目标（What to build）

实现一个基于 **LangChain** 的智能体，具备：

1. **短期记忆**（会话窗口）+ **长期记忆**（法规知识库，向量检索）
2. **工具库**两件套（由智能体自主决定是否调用）
    - **代码生成与执行工具**（Python 代码，沙箱/超时/资源限制）
    - **水质可饮用性预测工具**：使用 Kaggle “Water Potability” 数据集（ https://www.kaggle.com/datasets/adityakadiwal/water-potability ），训练 **CatBoost** 分类器，并提供预测接口
3. 支持中文对话，能回答水利法规问答、执行小段代码、对给定水质参数做饮用性预测，并在需要时自动选择工具或直接回答。

---

## 评分标准（100 分）

-   **功能完整性（30 分）**
    -   ✔️ 可对话、记忆生效（短期+长期）
    -   ✔️ 法规问答走向量检索（RAG）
    -   ✔️ 两个工具可被智能体调用并返回结果
-   **智能体决策与鲁棒性（25 分）**
    -   根据任务类型合理选择“直接回答 / 法规检索 / 代码执行 / 水质预测”
    -   异常处理与报错信息清晰（如超时、非法代码、缺特征）
-   **模型与数据（20 分）**
    -   CatBoost 训练流程规范，划分验证集并报告指标（AUC/Accuracy/F1 至少其一）
    -   模型持久化与加载预测稳定
-   **工程质量（15 分）**
    -   清晰的项目结构、配置、日志/回调、README
    -   依赖与环境管理（可在本地一键跑通）
-   **可测试性与演示（10 分）**
    -   提供最小可复现实例与脚本

> 加分项（最多 +10）：
>
> -   对法规结果给出来源片段与相似度分数
> -   对 CatBoost 给出特征重要性可视化/SHAP 说明
> -   代码执行工具采用真正隔离（如 Docker/namespaces）并记录资源用量

---

## 技术要求与建议

-   **Python 3.10+**；建议使用 `uv` 或 `poetry` 或 `pip-tools` 管理依赖
-   **LangChain 0.2+**（建议模块化：`langchain`, `langchain_openai`/本地 LLM, `langchain_community`, `langchain_text_splitters`）
-   **向量库**：`Chroma` 或 `FAISS` 均可（示例用 Chroma）
-   **Embedding**：中文为主，建议 `bge-large-zh-v1.5` 或 `text2vec-large-chinese`（HuggingFace）
-   **LLM**：自选（OpenAI/深度求索/智谱/通义/本地 Ollama 均可），需提供 `.env` 配置
-   **CatBoost**：`catboost`, 评估用 `scikit-learn`
-   **数据**：Kaggle “Water Potability”
    -   允许本地下载后放入 `data/water_potability.csv`（免登录），或用 Kaggle API 下载

---
