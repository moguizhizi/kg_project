# TemporalMultimodalMedicalKG

TemporalMultimodalMedicalKG 用于构建时序多模态医学知识图谱。当前流程包含：

- TMMKG ontology 数据库初始化
- TMMKG entity 数据库初始化
- HBUT、L2BA、OOTL 三类 KG 数据导入 Neo4j

本文档以实际执行和运维排错为主。

## 环境依赖

运行前需要确认以下服务和资源可用：

- Python 3.10+
- MongoDB
- Qdrant
- Neo4j
- Qwen3-Embedding-8B embedding 模型

项目代码通过 `src` 目录加载包，因此执行脚本时需要设置：

```bash
PYTHONPATH=src
```

如果机器上配置了 `http_proxy` 或 `https_proxy`，访问本机或内网服务时建议加：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121
no_proxy=localhost,127.0.0.1,10.30.1.121
```

## 配置文件

统一配置文件为：

```text
configs/tmmkg.yaml
```

MongoDB、Qdrant、Neo4j、embedding 模型路径、日志路径、数据路径等参数都从 YAML 中读取，不再依赖 `.env`。

默认配置包括：

- `infra.mongo.uri`
- `infra.qdrant.uri`
- `infra.neo4j.uri`
- `infra.neo4j.user`
- `infra.neo4j.password`
- `embedding.model_root`
- `logging.*`
- `pipelines.HBUT`
- `pipelines.L2BA`
- `pipelines.OOTL`

如需指定其他配置文件：

```bash
PYTHONPATH=src python src/TMMKG/create_L2BA_KG.py --config /path/to/tmmkg.yaml
```

## 初始化数据库

初始化 ontology 数据：

```bash
PYTHONPATH=src python -m TMMKG.create_tmmkg_ontology_db
```

初始化 entity 数据：

```bash
PYTHONPATH=src python -m TMMKG.create_tmmkg_entity_db
```

如果代理会影响 Qdrant 或本地服务访问：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src python -m TMMKG.create_tmmkg_entity_db
```

## 导入 KG

### L2BA

前台执行：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src python src/TMMKG/create_L2BA_KG.py
```

后台执行，并保留输出到 `nohup.out`：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src nohup python src/TMMKG/create_L2BA_KG.py &
```

### OOTL

前台执行：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src python src/TMMKG/create_OOTL_KG.py
```

后台执行：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src nohup python src/TMMKG/create_OOTL_KG.py &
```

### HBUT

前台执行：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py
```

后台执行：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src nohup python src/TMMKG/create_HBUT_KG.py &
```

HBUT 会初始化 `EntityResolver`，因此正式执行时需要 Qdrant 和 embedding 模型可用。

## 调试模式

三个 KG 导入脚本均支持 `--limit-records`：

```bash
PYTHONPATH=src python src/TMMKG/create_L2BA_KG.py --limit-records 10
PYTHONPATH=src python src/TMMKG/create_OOTL_KG.py --limit-records 10
PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py --limit-records 10
```

HBUT 的 `--limit-records` 会自动启用 dry-run：

- 不写入 Neo4j
- 不初始化 EntityResolver
- 不访问 Qdrant
- 只生成调试用 Parquet 和 facts 文件
- 只转换第一个满足必需列的 sheet

HBUT 也可以显式 dry-run：

```bash
PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py --dry-run
```

## Neo4j 约束

KG 导入脚本会在导入 facts 前自动确保以下唯一约束存在：

```text
Patient(id)
TaskInstanceSet(id)
TaskInstance(id)
Game(id)
Disease(id)
Symptom(id)
Unknown(id)
```

查看约束：

```cypher
SHOW CONSTRAINTS;
```

查看索引：

```cypher
SHOW INDEXES;
```

如果历史库中已经存在同 label、同 id 的重复节点，唯一约束会创建失败。需要先清理重复节点，例如：

```cypher
MATCH (n:TaskInstanceSet)
WITH n.id AS id, count(n) AS c
WHERE c > 1
RETURN id, c
ORDER BY c DESC
LIMIT 20;
```

确认某个 id 是否重复：

```cypher
MATCH (n:TaskInstanceSet {id: '20104719_20241210'})
RETURN id(n) AS neo4j_node_id, labels(n) AS labels, n
ORDER BY neo4j_node_id;
```

## 日志

日志配置在 `configs/tmmkg.yaml` 的 `logging` 中。

默认日志文件：

```text
logs/create_tmmkg_ontology_db.log
logs/create_tmmkg_entity_db.log
logs/create_L2BA_KG.log
logs/create_OOTL_KG.log
logs/create_HBUT_KG.log
```

查看实时日志：

```bash
tail -f logs/create_L2BA_KG.log
tail -f logs/create_OOTL_KG.log
tail -f logs/create_HBUT_KG.log
```

后台执行时，如果没有重定向，stdout 和 stderr 会进入：

```text
nohup.out
```

查看：

```bash
tail -f nohup.out
```

导入脚本入口已经捕获顶层异常并执行：

```python
logger.exception("Pipeline failed")
```

因此 pipeline 阶段失败时，调用栈会写入对应程序日志，同时异常仍会抛出并导致进程退出。

## 常见问题

### Qdrant 访问 localhost:6333 返回 502

通常是代理环境影响了本地访问。执行命令时加：

```bash
NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
no_proxy=localhost,127.0.0.1,10.30.1.121 \
PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py
```

### Neo4j 导入越来越慢

优先检查约束：

```cypher
SHOW CONSTRAINTS;
```

如果没有 `Label(id)` 唯一约束，`MERGE (n:Label {id: row.id})` 会越来越慢。

### 创建唯一约束失败

说明库里已经有重复节点。先查重复 id，清理后再创建约束。

### 后台任务 Exit 1 但看不到原因

不要把输出丢到 `/dev/null`。推荐：

```bash
PYTHONPATH=src nohup python src/TMMKG/create_L2BA_KG.py &
```

然后查看：

```bash
tail -n 100 nohup.out
tail -n 100 logs/create_L2BA_KG.log
```

### 疾病节点没有 name

当前 HBUT 的疾病关系导入主要通过 entity facts 创建关系。若疾病节点是关系尾节点自动 `MERGE` 出来的，默认只保证 `id` 存在，不一定写入 `name`。

如果需要疾病节点带标准名称，需要在疾病实体初始化阶段提前写入，或扩展 entity facts 结构携带 `tail_name` 并在关系导入时写入 tail 节点属性。
