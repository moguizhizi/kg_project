"""
property_resolver_demo.py

用于手动测试 PropertyResolver 的主入口
"""

import logging

from TMMKG.services.property_resolver import init_property_resolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting PropertyResolver demo")

    # 1. 初始化 resolver（命中 lru_cache）
    resolver = init_property_resolver()

    # 2. 测试文本
    text = "出生日期"
    top_k = 5

    logger.info(f"Resolving property for text: '{text}'")

    # 3. 调用解析
    candidates = resolver.resolve(text, top_k=top_k)

    # 4. 输出结果
    if not candidates:
        logger.info("No candidates found")
        return

    logger.info("Candidates:")
    for c in candidates:
        logger.info(
            f"- property_id={c.property_id}, "
            f"alias='{c.alias_label}', "
            f"is_canonical={c.is_canonical}, "
            f"score={c.score:.4f}"
        )


if __name__ == "__main__":
    main()
