# AudioLLM 非实时音频分析 API

本文档面向外部调用方，说明如何通过服务地址 `http://172.16.0.3:8082` 上传一段音频，并获取 ASR 原始结果、ASR 清洗文本、情感标签和情感描述。

## 访问地址

| 项目 | 值 |
|---|---|
| Base URL | `http://172.16.0.3:8082` |
| Endpoint | `POST /api/audio/analyze` |
| 完整 URL | `http://172.16.0.3:8082/api/audio/analyze` |
| Content-Type | `multipart/form-data` |
| 鉴权 | 当前不需要 API Key 或 Token |

## 功能说明

接口会按以下流程处理音频：

1. 调用 ASR 模型识别音频文本，热词偏置来自当前 `hotword_pool_id` 对应的 RAG-ASR 热词池召回，以及优先注入的少量表单 `hotwords` 临时热词；临时热词会覆盖精确重复或整词同音（忽略声调）的召回词。
2. 对 ASR 文本做清洗，只处理标点、空格、重复词和明显格式问题。
3. 做情感理解，同时返回情感标签和情感描述。

注意：`hotwords` 是临时请求热词，默认最多 8 个优先进入 ASR prompt，不写入热词池；它不会传给文本清洗阶段，因此不会发生“根据热词事后替换 ASR 文本”的行为。

ASR 送模前会执行服务端人声门控与过滤。低人声证据音频可能不调用 ASR 模型并返回空 `asr.text`；已放行音频可能在送模前裁去非人声区间。该保护是服务端配置，请求不能关闭或放宽。

## 请求参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `audio` | file | 是 | WAV 音频文件。当前接口要求 WAV 容器；m4a/mp3 请先转 WAV |
| `language` | string | 否 | 语言代码，例如 `zh`、`en`。中文建议传 `zh` |
| `hotword_pool_id` | string | 否 | 推荐字段，热词池隔离 ID，默认 `default` |
| `hotwords` | string | 否 | 临时请求热词；去重限量后优先进入 ASR prompt，并覆盖精确重复或整词同音（忽略声调）的 RAG-ASR 召回热词，不写入热词池 |

音频建议：

| 项目 | 建议 |
|---|---|
| 格式 | WAV |
| 采样率 | 16 kHz 或常见采样率，服务端会转为 16 kHz mono |
| 声道 | mono 或 stereo 均可 |
| 文件大小 | 建议小于 50 MB |

## cURL 示例

请使用 `-F` 发送 `multipart/form-data`。不要手动添加 `Content-Type: multipart/form-data` 请求头，curl 会自动生成正确的 boundary；手动设置容易导致服务端返回 `400 There was an error parsing the body`。

```bash
curl -X POST "http://172.16.0.3:8082/api/audio/analyze" \
  -F "audio=@sample.wav" \
  -F "language=zh" \
  -F "hotword_pool_id=tenant-a" \
  -F "hotwords=挚音科技"
```

## Python 示例

```python
import json
from pathlib import Path

import requests

url = "http://172.16.0.3:8082/api/audio/analyze"
audio_path = Path("sample.wav")

with audio_path.open("rb") as f:
    resp = requests.post(
        url,
        files={"audio": (audio_path.name, f, "audio/wav")},
        data={
            "language": "zh",
            "hotwords": "挚音科技",
        },
        timeout=240,
    )

resp.raise_for_status()
result = resp.json()
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## ArkTS 示例

ArkTS 调这个接口时，推荐使用 HarmonyOS 常见的文件上传方式 `request.uploadFile(...)`，不要自己用字符串拼 multipart body。

注意事项：

- `audio` 必须是文件字段。
- `language`、`hotwords` 作为普通表单字段传递。
- 推荐使用 `request.uploadFile(context, uploadConfig)`。
- 如果你自己手写 multipart body 或错误设置 `Content-Type`，很容易触发 body 解析错误。

示例：

```ts
import common from '@ohos.app.ability.common';
import fs from '@ohos.file.fs';
import request from '@ohos.request';

const context = getContext(this) as common.UIAbilityContext;

async function callAudioAnalyzeApi(filePath: string) {
  const file = fs.openSync(filePath, fs.OpenMode.READ_ONLY);
  fs.closeSync(file);

  const uploadConfig = {
    url: 'http://172.16.0.3:8082/api/audio/analyze',
    header: {
      key1: 'Content-Type',
      key2: 'multipart/form-data'
    },
    method: 'POST',
    files: [
      {
        filename: 'sample.wav',
        name: 'audio',
        uri: 'internal://cache/sample.wav',
        type: 'wav'
      }
    ],
    data: [
      { name: 'language', value: 'zh' },
      { name: 'hotwords', value: '挚音科技' }
    ]
  };

  try {
    const uploadTask = await request.uploadFile(context, uploadConfig);
    uploadTask.on('complete', (taskStates) => {
      for (let i = 0; i < taskStates.length; i++) {
        console.info(`upload complete: ${JSON.stringify(taskStates[i])}`);
      }
    });
  } catch (err) {
    console.error(`request failed: ${JSON.stringify(err)}`);
  }
}
```

说明：

- 示例中的 `internal://cache/sample.wav` 代表应用缓存目录中的文件，请先把待上传音频写入缓存目录。
- `audio` 是服务端要求的文件字段名，不能改成别的名字。
- `language`、`hotwords` 在 `data` 数组里传递即可。

如果客户端声明了 `multipart/form-data`，但实际 body 不是合法 multipart，服务端会返回：

```json
{
  "detail": "There was an error parsing the body"
}
```

## 响应示例

```json
{
  "type": "audio_analysis",
  "duration_sec": 12.864,
  "language": "zh",
  "hotwords": ["挚音科技"],
  "asr": {
    "text": "欢迎使用挚音科技语音分析服务，本次演示会介绍非实时音频转写和情感理解能力。",
    "language": "zh"
  },
  "cleaned_asr": {
    "text": "欢迎使用挚音科技语音分析服务。本次演示会介绍非实时音频转写和情感理解能力。"
  },
  "emotion": {
    "type": "final_emotion_pair",
    "mode": "both",
    "ser": {
      "type": "final_emotion",
      "mode": "ser",
      "label": "Neutral",
      "text": "Neutral",
      "duration_sec": 12.864,
      "language": "zh"
    },
    "sec": {
      "type": "final_emotion",
      "mode": "sec",
      "label": "Neutral",
      "text": "说话人语气平稳、表达清晰，整体情绪中性偏积极，呈现出专业介绍和产品演示的状态。",
      "duration_sec": 12.864,
      "language": "zh"
    }
  }
}
```

## 关键字段说明

| 字段 | 说明 |
|---|---|
| `asr.text` | ASR 最终融合后的识别文本 |
| `cleaned_asr.text` | 清洗后的文本 |
| `emotion.ser.label` | 情感分类标签 |
| `emotion.sec.text` | 情感文本描述 |

## 错误响应

| HTTP 状态码 | 常见原因 |
|---|---|
| 400 | 音频为空、不是 WAV 容器或无法解码 |
| 413 | 文件超过 nginx 或服务端上传限制 |
| 422 | 表单字段缺失或类型不合法 |
| 502 | ASR、情感或文本清洗模型调用失败 |

错误示例：

```json
{
  "detail": "could not decode audio: Invalid WAV container: file does not start with RIFF id"
}
```

## m4a/mp3 转 WAV

当前接口要求上传 WAV。如果你手上是 m4a/mp3，可先转换：

```bash
ffmpeg -y -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 sample.wav
```
