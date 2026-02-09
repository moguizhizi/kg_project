# TMMKG/sql_templates.py

ATTRIBUTE_FACT_SQL = """
SELECT
    json_extract_string(json, '$[0]') AS head_id,
    json_extract_string(json, '$[1]') AS head_type,
    json_extract_string(json, '$[2]') AS head_name,
    json_extract_string(json, '$[3]') AS relation,
    json_extract_string(json, '$[4]') AS prop,
    json_extract_string(json, '$[5]') AS tail,
    json_extract_string(json, '$[6]') AS tail_type
FROM read_json_auto('{path}')
"""


ENTITY_FACT_SQL = """
SELECT
    json_extract_string(json, '$[0]')        AS head_id,
    json_extract_string(json, '$[1]') AS head_type,
    json_extract_string(json, '$[2]') AS head_name,
    json_extract_string(json, '$[3]') AS relation,
    json_extract_string(json, '$[4]') AS prop,
    json_extract_string(json, '$[5]')        AS tail,
    json_extract_string(json, '$[6]') AS tail_type
FROM read_json_auto('{path}')
"""


UPSERT_NODE_CYPHER = """
UNWIND $rows AS row
MERGE (n:`{label}` {{id: row.id}})
SET n += row
"""


# 关系 UPSERT 模板
UPSERT_REL_CYPHER = """
UNWIND $rows AS row
MERGE (h:`{h_label}` {{id: row.h_id}})
MERGE (t:`{t_label}` {{id: row.t_id}})
MERGE (h)-[r:`{r_name}`]->(t)
"""

# 创建约束
CREATE_CONSTRAINT_CYPHER = """
CREATE CONSTRAINT IF NOT EXISTS
FOR (n:{label})
REQUIRE n.{pk} IS UNIQUE
"""

# 统计指定标签下、id 在 ids 列表中的节点数量
COUNT_NODES_IN_IDS_CYPHER = """
MATCH (n:`{label}`)
WHERE n.id IN $ids
RETURN count(n) AS node_count
"""

# 获取指定标签下所有节点的 id
FETCH_NODE_IDS_CYPHER = """
MATCH (n:`{label}`)
RETURN n.id AS id
"""

# 统计指定标签节点的一跳关系数量
COUNT_1HOP_RELATIONS_CYPHER = """
MATCH (n:`{label}`)-[r]-()
WHERE n.id IN $ids
RETURN count(r) AS triples
"""

# 统计指定标签节点在 ids 列表中的 2-hop 关系数量
COUNT_2HOP_RELATIONS_CYPHER = """
MATCH (n:`{label}`)-[r1]-()-[r2]-()
WHERE n.id IN $ids
RETURN count(*) AS two_hop_triple_count
"""

# 查询度数大于指定阈值的高连接节点（Top N）
HIGH_DEGREE_NODES_CYPHER = """
MATCH (n)
WITH n, COUNT {{ (n)--() }} AS degree
WHERE degree > {min_degree}
RETURN labels(n) AS labels, n.id AS id, degree
ORDER BY degree DESC
"""

# 通用批量更新节点属性的 Cypher 模板（自动分页）
UPDATE_NODE_PROPERTY_CYPHER = f"""
MATCH (n:`{{label}}`)
WHERE n.`{{prop_name}}` IS NULL
WITH n LIMIT $batch_size
SET n.`{{prop_name}}` = $prop_value
RETURN count(n) AS updated_count
"""

# 通用批量更新节点属性的 Cypher 模板（支持 SKIP 分页）
UPDATE_NODE_PROPERTY_WITH_SKIP_CYPHER = f"""
MATCH (n:`{{label}}`)
WITH n SKIP $skip LIMIT $batch_size
SET n.`{{prop_name}}` = $prop_value
RETURN count(n) AS updated_count
"""
