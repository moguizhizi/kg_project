import re

# 预编译正则（性能更好）
_INVISIBLE_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def clean_text(text):
    """
    清理字符串中的：
    - 零宽空格 \u200b
    - 零宽非连接符 \u200c
    - 零宽连接符 \u200d
    - BOM \ufeff
    - 首尾空格

    非字符串类型会原样返回
    """
    if not isinstance(text, str):
        return text

    text = text.strip()
    text = _INVISIBLE_PATTERN.sub("", text)
    return text

def deep_clean(obj):
    """
    递归清理任意嵌套数据结构中的字符串内容。

    功能：
    - 删除字符串中的零宽字符（\u200b\u200c\u200d\ufeff）
    - 去除首尾空格
    - 支持嵌套结构（list / tuple / dict）
    - 非字符串类型保持原样返回

    适用场景：
    - JSON 读取后数据清洗
    - Excel / Parquet 读入后的字段标准化
    - 构建知识图谱前的统一文本清理

    参数:
        obj: 任意 Python 对象

    返回:
        清理后的对象（结构保持不变）
    """

    # 如果是字符串，直接调用 clean_text 清理
    if isinstance(obj, str):
        return clean_text(obj)

    # 如果是 list，递归清理每个元素
    elif isinstance(obj, list):
        return [deep_clean(i) for i in obj]

    # 如果是 tuple，递归清理后再转回 tuple
    elif isinstance(obj, tuple):
        return tuple(deep_clean(i) for i in obj)

    # 如果是 dict，同时清理 key 和 value
    elif isinstance(obj, dict):
        return {
            deep_clean(k): deep_clean(v)
            for k, v in obj.items()
        }

    # 其他类型（int / float / bool / None 等）原样返回
    else:
        return obj
