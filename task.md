# 项目任务：基于语音大模型的噪声与混响鲁棒 ASR 服务

## 1. 项目目标

完成一个可以展示在 GitHub 和简历中的 ASR 工程项目，用于替换原简历中的 music tagging 项目。项目重点体现语音大模型、声学数据增强、参数高效微调、鲁棒性评估、服务部署和推理优化能力，不要求提出新算法或发表论文。

具体任务：使用 Qwen3-ASR 构建英文语音转写服务，使模型在背景噪声、房间混响以及噪声与混响同时存在时保持较好的识别准确率。

| 项目 | 定义 |
| --- | --- |
| 输入 | 干净、带噪、混响或带噪加混响的英文语音 |
| 输出 | 英文转写文本；服务版包含片段时间戳和处理耗时 |
| 训练方式 | 在干净语音上动态加入噪声和 RIR，再进行 LoRA SFT |
| 主要质量指标 | Clean WER、各退化条件下的 WER、平均 Robust WER |
| 辅助指标 | 相对 WER 增幅、插入/删除/替换错误、不同噪声类型和 SNR 下的 WER |
| 工程指标 | RTF、吞吐量、P50/P95 耗时、峰值 GPU 显存 |

第一版完成英文离线转写和受控鲁棒性评估。实时流式 ASR、说话人分离、语音增强模型训练和完整 voice agent 属于后续扩展。

## 2. 任务定义

对每条干净语音 `x` 和转写 `y`，动态生成以下四类训练输入：

1. 干净语音：`x`
2. 带噪语音：`x + n`
3. 混响语音：`x * h`
4. 带噪加混响语音：`x * h + n`

其中 `n` 是噪声片段，`h` 是房间脉冲响应，`*` 表示卷积。四类输入的监督文本都保持为 `y`。

初始采样概率建议均为 25%。这些概率必须配置化，并根据数据统计与开发集结果调整。噪声在目标 SNR 下混合，第一轮训练可从 `[-5, 0, 5, 10, 15, 20] dB` 中随机选择，也可以在 `[-5, 20] dB` 连续采样。混响之后再按实际语音功率计算噪声增益，避免 SNR 定义混乱。

训练数据增强必须在线或按 epoch 重采样。不能为每条语音永久生成一个固定退化版本，否则模型容易记住有限的噪声和 RIR 组合。

## 3. 硬件与模型

- 目标硬件：一张或多张 NVIDIA RTX A6000，每张 48GB 显存。
- 第一版基础模型：`Qwen/Qwen3-ASR-0.6B`。
- 后续对照模型：`Qwen/Qwen3-ASR-1.7B`。
- Qwen3-ASR 包含 log-Mel 前端、audio encoder、multimodal projector 和 Qwen3 language model，因此该任务同时涉及声学模型与 LLM。
- 优先从官方 `qwen-asr` 推理和 SFT 实现起步，固定实际使用的模型 revision、Transformers、PEFT、PyTorch 和 CUDA 版本。
- 先在单张 A6000 上跑通数据、前向、反向、adapter 保存和重载。多卡用于扩大 batch 或并行实验。

## 4. 数据来源

### 4.1 干净语音：LibriSpeech

官方入口：<https://www.openslr.org/12/>

第一版使用：

| 数据部分 | 用途 | 压缩包大小 |
| --- | --- | ---: |
| `train-clean-100` | 训练语音，可先抽取 20 至 30 小时调通流程 | 约 6.3GB |
| `dev-clean` | 干净开发集和合成鲁棒开发集 | 约 337MB |
| `test-clean` | 干净测试集和合成鲁棒测试集 | 约 346MB |
| `test-other` | 较困难但不受本项目控制的额外测试 | 约 328MB |

LibriSpeech 是约 1,000 小时的 16kHz 英文朗读语音，采用 CC BY 4.0。第一版不需要下载完整语料。

### 4.2 噪声：MUSAN

官方入口：<https://www.openslr.org/17/>

- 完整压缩包约 11GB，采用 CC BY 4.0。
- 包含 noise、music 和 speech。
- 第一版至少覆盖环境噪声、音乐和多人或干扰说话三类退化。
- 对 speech 类素材可随机叠加多个说话人生成 babble，但必须限制响度并避免把目标转写替换成干扰语音内容。
- 先根据实际文件和许可信息建立清单，记录每种噪声来源和时长。

### 4.3 RIR 与补充噪声：OpenSLR SLR28

官方入口：<https://www.openslr.org/28/>

- 压缩包约 1.3GB，采用 Apache 2.0。
- 包含真实和模拟 RIR、各向同性噪声和点声源噪声。
- 音频均为 16kHz、16 bit，便于与 ASR 语音直接配合。
- 第一版使用真实与模拟 RIR 生成房间混响，并将一部分 RIR 来源完全留作测试。

### 4.4 可选模拟 RIR：OpenSLR SLR26

官方入口：<https://www.openslr.org/26/>

- 16kHz 版本约 178MB，采用 Apache 2.0。
- 包含多种房间配置的模拟 RIR。
- 可用于扩大训练时的房间覆盖；测试仍需要保留未参与训练的真实 RIR。

### 4.5 可选大规模噪声：Microsoft DNS Challenge

官方入口：<https://github.com/microsoft/DNS-Challenge>

DNS Challenge 提供噪声、RIR、合成脚本和相关测试资源。完整数据很大，官方 DNS5 页面列出的 noise 部分约 58GB，RIR 约 5.9GB，完整解压数据接近 1TB。第一版不下载完整 DNS 数据。只有在 MUSAN 和 SLR28 已跑通且需要更多噪声覆盖时，才选择性加入 DNS 噪声子集。

### 4.6 可选真实噪声测试：CHiME-4

官方入口：<https://www.chimechallenge.org/challenges/chime4/data>

CHiME-4 包含公交车、咖啡馆、行人区和街道路口等真实及模拟噪声环境。它适合作为合成测试之外的真实噪声泛化评估。数据与 WSJ0 相关，执行前必须查看当前下载条件和使用要求。若获取成本过高，不阻塞第一版交付。

## 5. 数据划分与防止泄漏

语音、噪声和 RIR 都必须独立划分：

- 语音严格沿用 LibriSpeech 官方 train、dev、test 边界。
- 噪声按源文件划分 train、dev、test。来自同一长录音的切片只能属于一个 split。
- RIR 按房间或原始数据库来源划分，不能只随机拆分卷积后的语音。
- test 使用的噪声文件和 RIR 不得参与训练或调参。
- 合成测试必须由固定 manifest 决定，包括语音 ID、噪声 ID、RIR ID、SNR、增益、裁剪起点和随机种子。
- 训练可动态随机增强；dev 和 test 必须确定性复现。
- 记录每个 split 的文件数、时长、噪声类别、RIR 来源及 SNR 分布。

同一条测试语音可以生成多个受控版本，用于成对比较模型在 clean、noise、reverb 和 noise plus reverb 条件下的变化。计算总体 WER 时，必须明确每个条件的样本数量和权重。

## 6. 音频增强实现要求

### 6.1 RIR 卷积

- 统一采样率和声道格式。
- 对 RIR 去除明显前置静音并做能量归一化，但保留其衰减结构。
- 使用 FFT convolution 或等价正确实现。
- 对卷积后的长度做明确裁剪或保留尾部策略，并在训练、开发和测试中保持一致。
- 防止峰值 clipping，记录归一化和增益规则。
- 保存增强参数，不保存无法追溯来源的随机音频文件。

### 6.2 噪声混合

- 基于有效语音区域的 RMS 或明确的语音功率估计计算 SNR，不能直接用包含大量静音的整段 RMS。
- 噪声不足时允许循环、拼接或重新采样，但处理方式必须确定并记录。
- 加噪后检查 NaN、Inf、静音和 clipping。
- 可加入随机整体增益，避免模型把绝对响度当作捷径。

### 6.3 验证

- 为增强函数编写少量关键测试：identity RIR、已知 SNR、输出长度、确定性复现和无 clipping。
- 随机抽样保存波形和频谱图，人工确认增强结果合理。
- 统计实际实现后的 SNR 误差，而不是只相信配置值。

## 7. LoRA 训练策略

### 7.1 为什么要修改之前的 LoRA 方案

仅在 language model decoder 上训练 LoRA 更适合术语、语言风格和文本先验适配。这个项目的域偏移主要发生在声学输入，因此第一版应主要更新 audio encoder。Qwen3-ASR 的 audio encoder 包含 attention 中的 `q_proj`、`k_proj`、`v_proj` 和 `out_proj`，并通过 multimodal projector 将声学表示映射到语言模型空间。

### 7.2 首选参数高效方案

| 模块 | 第一轮策略 |
| --- | --- |
| Audio encoder attention | 加入 LoRA |
| Multimodal projector | 直接训练 |
| Language model | 冻结 |
| Convolutional frontend | 第一轮冻结 |
| Embedding 和 LM head | 冻结 |

建议初始配置：

| 参数 | 初始值 |
| --- | --- |
| LoRA targets | audio encoder 的 `q_proj`、`k_proj`、`v_proj`、`out_proj` |
| `r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| 学习率 | 从 `5e-5` 开始，根据 dev WER 调整 |
| 精度 | BF16，先确认硬件和实现支持 |
| 训练长度 | 先观察最多 1 个 epoch，验证仍改善时再继续 |

实现要求：

- 用完整模块路径限制 LoRA 只命中 `audio_tower`，避免把 language model 中同名投影层一并加入。
- 打印并保存所有命中的 LoRA 层、直接训练的 projector 参数和总可训练参数比例。
- 检查 LoRA 与 projector 参数有梯度，其余参数保持冻结。
- 验证 adapter 保存、重载和生成结果一致。
- 官方 Qwen3-ASR SFT 脚本没有直接提供 LoRA 开关，需要在保留其音频处理、label mask 和 forward 流程的基础上接入 PEFT 或等价实现。
- 使用生成转写后的 dev Robust WER 选 checkpoint。训练 loss 和 teacher-forced dev loss 只用于诊断。
- 如果 audio encoder LoRA 改善不足，按顺序尝试更高 rank、加入 encoder FFN、解冻最后若干 audio encoder 层、解冻 convolutional frontend。每次只改变一个主要因素。
- 可将 decoder-only LoRA 作为对照，验证它对声学鲁棒性的作用，不作为默认主方案。
- 全参数微调保留为后续对照，不是第一版要求。

LoRA 减少可训练参数，但不能保证不会过拟合。若训练 loss 下降而 dev Robust WER 上升，应回到最佳 checkpoint，并检查数据多样性、学习率、训练步数和增强分布。

## 8. 基准与实验矩阵

至少完成以下模型：

| 模型 | 用途 |
| --- | --- |
| 原始 Qwen3-ASR-0.6B | 零样本 baseline |
| Clean-only LoRA | 判断普通领域 SFT 是否损害或改善鲁棒性 |
| Noise and RIR augmented acoustic LoRA | 项目主模型 |

固定测试条件建议如下：

| 条件 | 设置 |
| --- | --- |
| Clean | 原始 `test-clean` |
| Noise | held-out noise，SNR 为 -5、0、5、10、20 dB |
| Reverb | held-out real 或 simulated RIR |
| Noise plus Reverb | held-out RIR 加 held-out noise，使用同一组 SNR |
| Natural hard | 原始 `test-other` |
| Real noise，可选 | CHiME-4 单通道真实噪声测试 |

第一版不需要把所有组合做成笛卡尔积。应预先固定每个条件的样本数量，在单张 A6000 能合理完成的时间内覆盖不同 SNR、噪声类别和 RIR 来源。

## 9. 评估指标

### 9.1 ASR 质量

- 使用 corpus-level WER：全测试集替换、删除、插入次数之和除以参考词数之和。
- 对预测和参考应用相同的大小写、标点、数字和特殊标记规范化规则，同时保留原始文本。
- 分别报告 Clean WER、Noise WER、Reverb WER 和 Noise plus Reverb WER。
- Robust WER 为预先规定的退化条件宏平均值，具体条件和权重必须写入配置。
- 报告每个 SNR 和噪声类别的 WER，避免平均值掩盖极端条件失败。
- 报告 clean regression，即微调模型相对于原始模型在 clean 条件下的变化。
- 对固定样本保存替换、删除和插入案例，特别检查低 SNR 下的 hallucination 和重复生成。

### 9.2 工程性能

- RTF = 处理耗时 / 音频时长，越低越好。
- 吞吐量 = 处理的音频秒数 / 实际墙钟时间。
- P50/P95 请求耗时必须注明音频长度和并发数。
- 分开记录冷启动和预热后的性能。
- 报告 GPU 型号、卡数、峰值显存、batch size、精度、模型 revision 和软件版本。
- 只填写实际运行结果，不能把官方 benchmark 当作项目实测结果。

## 10. 实施阶段

### 阶段 0：环境与原始模型

- 检查仓库、GPU、CUDA、磁盘和依赖。
- 下载模型并在一个英文样例上成功转写。
- 记录实际验证过的安装和运行命令。

### 阶段 1：数据与增强管线

- 下载 LibriSpeech、MUSAN 和 SLR28 的第一版所需部分。
- 构建 speech、noise 和 RIR manifest。
- 实现在线 RIR convolution、SNR mixing 和四类增强采样。
- 建立确定性的 dev/test corruption manifest。
- 完成关键增强测试与人工试听检查。

### 阶段 2：鲁棒性 baseline

- 用原始 Qwen3-ASR-0.6B 跑固定 clean 和 degraded 测试。
- 输出各条件 WER、各 SNR WER、错误类型和推理耗时。
- 根据 baseline 确定正式训练覆盖的 SNR 和退化组合。

### 阶段 3：LoRA 训练

- 先用少量数据检查 loss、梯度、冻结范围和 adapter 重载。
- 完成 Clean-only LoRA 对照。
- 完成 Noise and RIR augmented acoustic LoRA。
- 在固定 dev 条件上按 Robust WER 选 checkpoint 和早停。
- 在固定 test 条件上进行一次最终比较。

### 阶段 4：服务

- 提供音频上传或 API 输入。
- 返回转写文本、片段时间戳、音频时长和处理耗时。
- 处理采样率转换、双声道转单声道、长音频切分、无语音片段和失败反馈。
- 提供最小演示页面，允许用户上传一条干净或带噪音频并查看结果。

### 阶段 5：性能优化

- 分析音频读取、重采样、特征提取、audio encoder 和 token generation 耗时。
- 根据实测瓶颈优化 batching、按时长分桶、并发和模型服务。
- 如果部署后端不支持动态加载 Qwen3-ASR LoRA，评估合并 adapter 后导出，并验证合并前后固定样本输出和 WER。
- 在相同硬件和输入上保存优化前后质量、速度和显存结果。

### 阶段 6：可选扩展

1. 使用 CHiME-4 评估真实环境噪声泛化。
2. 加入 DNS noise 子集扩大噪声覆盖。
3. 对比 Qwen3-ASR-1.7B 的准确率与成本。
4. 增加一个预训练 speech enhancement 前端，比较 `enhance then ASR` 与端到端 acoustic LoRA。
5. 扩展到实时流式推理，并单独评估 chunk size、延迟和跨 chunk 文本稳定性。

## 11. 建议仓库结构

| 路径 | 用途 |
| --- | --- |
| `README.md` | 安装、数据、训练、评估、服务和实测结果 |
| `TASK.md` | 本任务说明 |
| `configs/` | 数据、增强、模型、训练和服务配置 |
| `src/data/` | manifest、数据加载和增强 |
| `src/model/` | Qwen3-ASR、LoRA 和 checkpoint 加载 |
| `src/eval/` | WER、错误分析和性能统计 |
| `src/service/` | API、长音频处理和结果格式 |
| `scripts/` | 数据准备、baseline、训练、评估和 benchmark 入口 |
| `tests/` | 增强、划分、WER 和 adapter 保存重载测试 |
| `reports/` | 配置快照、数据统计、结果表和错误案例 |

原始语音、噪声、RIR、缓存、模型权重、checkpoint 和认证信息不提交到 Git。只提交复现所需的代码、配置、合法的小样例、统计结果和说明。

## 12. 第一版完成标准

- [ ] 模型、数据和环境安装流程已实际验证。
- [ ] speech、noise 和 RIR 的 train/dev/test manifest 可复现且无已知泄漏。
- [ ] 动态 noise、RIR 和 combined augmentation 已实现并通过关键检查。
- [ ] 原始 0.6B 模型在固定 clean 和 degraded 测试上的 WER 已保存。
- [ ] Clean-only LoRA 对照已完成。
- [ ] Noise and RIR augmented acoustic LoRA 已完成，冻结范围和可训练参数已核对。
- [ ] adapter 可以保存、重载并用于生成。
- [ ] 已按 dev Robust WER 选择 checkpoint，并在固定 test 上完成最终比较。
- [ ] 有分条件、SNR 和错误类型的结果分析，且明确记录 clean regression。
- [ ] 有可运行的离线 ASR API 和最小演示界面。
- [ ] 至少完成一次基于实测瓶颈的性能优化并保存对比结果。
- [ ] README 中只陈述实际完成的功能和实测数据。

CHiME-4、DNS、1.7B、speech enhancement 和 streaming 都不阻塞第一版交付。

## 13. 给执行此任务的 Codex

先检查当前仓库和实际运行环境。第一步准备 LibriSpeech、MUSAN 和 SLR28 的 manifest 与增强管线，然后跑原始模型鲁棒性 baseline，再实现 acoustic LoRA。

不要默认当前执行环境已经具备 A6000、数据权限或足够磁盘。若资源暂时不可用，先完成代码、配置、单元验证和下载说明，并准确记录阻塞项。不得把未执行的下载、训练或评估标记为完成。

每完成一个阶段，简短说明修改了什么、如何验证、实测结果和下一步。简历 bullet 必须等最终结果产生后再撰写。

## 14. 官方参考资料

以下链接于 2026-09-11 核对，执行时应固定版本并确认接口变化。

- Qwen3-ASR 官方仓库：<https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR 官方 SFT：<https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md>
- Qwen3-ASR 技术报告：<https://arxiv.org/abs/2601.21337>
- Hugging Face Qwen3-ASR 实现：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_asr/modeling_qwen3_asr.py>
- PEFT LoRA：<https://huggingface.co/docs/peft/package_reference/lora>
- LibriSpeech：<https://www.openslr.org/12/>
- MUSAN：<https://www.openslr.org/17/>
- SLR28 RIR and Noise：<https://www.openslr.org/28/>
- SLR26 simulated RIR：<https://www.openslr.org/26/>
- Microsoft DNS Challenge：<https://github.com/microsoft/DNS-Challenge>
- CHiME-4：<https://www.chimechallenge.org/challenges/chime4/data>