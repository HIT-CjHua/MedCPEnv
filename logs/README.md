# MedAgent - 日志目录

## 概述

本目录存放 MedAgent 项目运行过程中产生的日志文件。

## 日志类型

| 类型 | 说明 |
|------|------|
| 训练日志 | Agentic RL 训练过程中的日志 |
| 评测日志 | Benchmark 评测过程中的日志 |
| API 调用日志 | LLM API 调用日志 |
| 错误日志 | 运行错误日志 |

## 日志配置

### 日志级别

| 级别 | 说明 |
|------|------|
| DEBUG | 调试信息，详细内部状态 |
| INFO | 一般信息，进度更新 |
| WARNING | 警告信息，不影响运行 |
| ERROR | 错误信息，可能导致失败 |
| CRITICAL | 严重错误，程序终止 |

### 日志输出

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/medagent.log'),
        logging.StreamHandler()
    ]
)
```

## 日志清理

```bash
# 清理 7 天前的日志
find logs/ -name "*.log" -mtime +7 -delete

# 清理所有日志
rm -f logs/*.log
```
