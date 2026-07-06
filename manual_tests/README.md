# Manual API Tests

这个目录用于手动测试非实时音频分析接口。

## 目录结构

```text
manual_tests/
  audio/      # 放测试音频，建议 WAV
  outputs/    # 保存接口返回 JSON
  scripts/    # 测试脚本
```

`audio/` 和 `outputs/` 下的实际文件默认不提交到 git。

## 测试 `/api/audio/analyze`

先确认服务已启动：

```bash
bash scripts/restart_service.sh
```

把音频放到 `manual_tests/audio/`，例如：

```text
manual_tests/audio/sample.wav
```

运行：

```bash
.venv/bin/python manual_tests/scripts/test_audio_analyze.py \
  manual_tests/audio/sample.wav \
  --base-url http://172.16.0.3:8082 \
  --language zh \
  --hotwords "挚音科技,语音识别"
```

脚本会打印摘要，并把完整响应保存到 `manual_tests/outputs/`。
情感理解默认同时返回 `emotion.ser` 标签和 `emotion.sec` 文本描述。

注意：`--hotwords` 只传给 ASR 接口本身，用于 ASR 原生热词识别；Qwen 文本清洗不会接收热词。
