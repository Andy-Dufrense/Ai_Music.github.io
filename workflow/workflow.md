# AI Music — 智能乐谱生成器 工作流文档

## 项目概述

上传 MP3 音乐文件 → 自动分离音轨 → 分析调性/和弦/BPM/歌曲结构 → 识别歌词 → 生成可弹奏的乐谱（钢琴谱、吉他谱、贝斯谱、架子鼓谱）。

---

## 技术选型总览

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| 前端 | 原生 HTML + CSS + JS | 单页应用，拖拽上传、进度轮询、乐谱展示 |
| 后端框架 | FastAPI + uvicorn | Python 异步 Web 框架，端口 8001 |
| 音源分离 | Demucs (htdemucs_6s) | Meta 开源模型，6 轨分离：vocals/drums/bass/guitar/piano/other |
| 音频分析 | Librosa | BPM 检测、调性识别 (Krumhansl-Kessler)、CQT 色谱 |
| 和弦检测 | chroma 模板匹配 | 12 种和弦模板，余弦相似度 + 贝斯音奖励 + 自然音阶奖励 |
| 歌词识别 | **OpenAI Whisper (base)** | 中文歌词识别，含繁简转换后处理 |
| 歌曲结构 | Librosa RMS energy + 和弦模式 | 前奏/主歌/副歌/桥段/间奏/尾奏六段式 |
| 旋律提取 | **torchcrepe (tiny)** | 深度学习音高追踪，~81% 准确率 |
| 乐谱渲染 | **手写 SVG** | 纯 Python 生成，无 LilyPond/Verovio 外部依赖 |
| 简谱标注 | MIDI → 调性音级度数 (1-7) | 附在音符/歌词上方 |
| 和弦指法 | 自研指法库 | 60+ 常用开放/横按和弦 + 算法 fallback |

---

## 文件结构

```
backend/
├── main.py                    # FastAPI 入口，路由 + 后台任务编排
├── config.py                  # 路径常量、乐谱类型定义
├── requirements.txt           # Python 依赖清单
├── services/
│   ├── common.py              # 共享常量、MIDI转换、简谱映射、繁简转换
│   ├── separator.py           # Demucs 音源分离 (6-stem)
│   ├── analyzer.py            # BPM / Key / Chord / TimeSig / Sections
│   ├── melody_rmvpe.py        # torchcrepe 旋律提取 + 贝斯音符提取
│   ├── lyrics_whisper.py      # Whisper base 歌词识别
│   ├── lyrics.py              # 歌词-旋律对齐 (beat-aware matching)
│   ├── notation.py            # 四种乐谱数据类型生成
│   └── svg_renderer.py        # 手写 SVG 乐谱渲染 (4 种风格)
├── uploads/                   # 用户上传的 MP3
└── outputs/                   # 每个 job 的输出
    └── {job_id}/
        ├── stems/             # 分离后的 6 轨 WAV
        ├── {type}_score.json  # 乐谱数据
        └── {type}_score.html  # 乐谱 HTML 页面
```

---

## 数据处理流程

```
用户上传 MP3 (max 50MB)
    ↓
[1] MP3 → WAV (pydub)
    ↓
[2] Demucs 音源分离 (CPU, shifts=0, overlap=0)
    → vocals.wav / drums.wav / bass.wav / guitar.wav / piano.wav / other.wav
    ↓
[3] Librosa 分析 (max 120s, 16000Hz)
    → BPM, 调性 (C major / A minor 等), 拍号 (4/4, 3/4, 6/8)
    ↓
[4] 和弦检测 (chroma_cqt + 模板匹配 + 贝斯辅助)
    → 和弦进行: I(C) → V(G) → vi(Am) → IV(F) ...
    ↓
[5] 歌曲结构分析 (vocal RMS energy + 和弦模式相似度)
    → [前奏] [主歌] [副歌] [间奏] [桥段] [尾奏]
    ↓
[6] 旋律提取 (torchcrepe tiny, 16000Hz, vocals stem)
    → MIDI 音高序列 + 时值
    ↓
[7] 贝斯音提取 (torchcrepe tiny, fmin=30 fmax=350, bass stem)
    → 贝斯 MIDI 音符序列
    ↓
[8] 歌词识别 (Whisper base, CPU, vocals stem)
    → 歌词全文 + 词级时间戳 + 繁简转换
    ↓
[9] 歌词-旋律对齐 (beat-aware window matching)
    → aligned_notes: MIDI + 时值 + 歌词 + 简谱
    ↓
用户选择乐谱类型 (钢琴/吉他/贝斯/鼓) + 背景音轨
    ↓
[10] notation.py 生成结构化乐谱数据
    ├── 钢琴: 右手旋律+和弦音 + 左手伴奏型 (大谱表)
    ├── 吉他: 和弦图 + 分解/扫弦 (六线 TAB) + 简谱歌词
    ├── 贝斯: 低音音符 + 节奏型 (四线 TAB) + 简谱
    └── 鼓:   Onset检测 + 6频段分类 (Kick/Snare/HH/Toms)
    ↓
[11] svg_renderer.py → 生成自包含 HTML 页面
    (含乐谱 SVG + 分析信息 + 音频播放器 + 保存按钮)
```

---

## 乐谱生成详情

### 钢琴谱
- **大谱表**: 高音谱号（右手旋律+和弦音 2-note texture）+ 低音谱号（左手伴奏）
- **段落感知伴奏**: 主歌/前奏→根+三+五 shell voicing, 副歌/桥段→根+三+五+八度 fuller
- **力度标记**: 前奏 p, 主歌 mp, 副歌 f, 桥段 mf, 尾奏 p
- **简谱标注**: 旋律音上方标 1-7 数字

### 吉他谱
- **六线 TAB**: 标准吉他定弦 (E A D G B E)
- **和弦图**: 60+ 常用和弦指法，barre chord 自动算法 fallback
- **风格切换**: 主歌/前奏/尾奏→分解 (picking), 副歌/桥段→扫弦 (strumming)
- **扫弦节奏** (按拍号):
  - 4/4: 前八后十六（主歌）/ 密集 16 分（副歌）
  - 3/4: 华尔兹型下-下上-上下上
  - 6/8: 复合拍节奏

### 贝斯谱
- **四线 TAB**: 标准贝斯定弦 (E A D G)
- **段落节奏型**: 每种段落有专属贝斯 pattern（根/五/三/八度音组合）
- **简谱标注**: 弹奏音上方标简谱度数

### 架子鼓谱
- **五线谱**: Kick (F4) / Snare (C5) / Hi-Hat (top line)
- **6 频段谱分析**: sub/lower/lower-mid/higher-mid/high/air
- **分类**: kick / snare / hihat_closed / tom_low / tom_high / crash / ride

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传 MP3，返回 job_id，启动后台分析 |
| GET | `/api/status/{job_id}` | 轮询处理进度 (0-100%) |
| POST | `/api/generate` | 指定 job_id + 乐谱类型 + 背景音轨 |
| GET | `/api/score/{job_id}/{type}` | 获取乐谱 HTML 页面 |
| GET | `/api/score_data/{job_id}/{type}` | 获取乐谱 JSON 数据 |
| GET | `/api/stems/{job_id}/{stem_name}` | 获取分离音轨 WAV |
| GET | `/` | 前端主页面 |

---

## 模型清单

| 模型 | 用途 | 位置 | 大小 |
|------|------|------|------|
| Whisper `base` | 歌词识别 | `~/.cache/whisper/base.pt` | ~139MB |
| Demucs `htdemucs_6s` | 6 音轨分离 | HuggingFace cache | ~300MB |
| torchcrepe `tiny` | 旋律/贝斯音高追踪 | torch hub cache | ~25MB |
| **总计** | | | **~464MB** |

---

## 处理耗时估算 (Intel CPU, 120s 音频)

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 音源分离 (Demucs) | 60-90s | shifts=0, overlap=0 |
| 音乐分析 | 5-10s | BPM/Key/Chords/Sections |
| 旋律提取 (torchcrepe) | 10-15s | tiny 模型, 16000Hz |
| 贝斯提取 (torchcrepe) | 5-10s | 同上, bass fmin=30 |
| 歌词识别 (Whisper) | 20-30s | base 模型, CPU |
| 歌词对齐 + 乐谱生成 | 1-2s | 纯计算 |
| SVG 渲染 | <1s | 每个乐谱类型 |
| **总计** | **~2-3 分钟** | |

---

## 速度优化措施

| 措施 | 效果 |
|------|------|
| Demucs shifts=0, overlap=0 | 减少 ~50% 分离时间 |
| torchcrepe tiny 模型 | 比 full 模型快 3-4x |
| 统一 16000Hz 采样率 | 减少 ~27% 数据量 |
| 分析时长上限 120s | 避免全曲分析 |
| CPU 全核线程 (OMP/MKL) | 充分利用多核 |
| Whisper 启动预加载 | 首次识别零等待 |
| 贝斯 note 预计算 | 避免生成时重复分析 |

---

## 已知局限

1. **歌词识别**: Whisper base 对中文歌词有漏字/错字，唱歌发音 vs 说话发音有差距
2. **和弦检测**: 12 种模板不覆盖爵士和声（altered, extensions）
3. **歌曲结构**: 基于 RMS energy + chord pattern 规则，非深度学习
4. **吉他指法**: 算法 fallback 使用 E-shape barre 模式，部分和弦可能不顺手
5. **架子鼓谱**: Onset 检测无法区分 Ghost Note / Rim Shot / 开镲细节
6. **仅 CPU**: 无 GPU 加速
7. **仅 MP3**: 不支持 WAV/FLAC/M4A 等其他格式
8. **无任务持久化**: 重启后历史 job 丢失

---

*最后更新: 2026-05-05*
