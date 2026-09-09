# 12 · 检索层 chunks 化与 BM25 作用域缓存

> V2.1 决策记录。对应 `app/rag/models.py`、`vectorstore.py`、`retriever.py`、
> `bm25.py`、`fusion.py`、`app/kb/store.py`、`tests/test_rag_offline.py`、
> `tests/test_rag_db.py`。格式约定（CLAUDE.md 第 6 条）。

## 1. 一句话结论

**检索身份键从 `(law_id, article_no)` 换成 `chunk_id`；检索带知识库作用域
（`kb_id = ANY(%s)`）；BM25 倒排从"进程内一次性加载永不失效"改为"按作用域缓存 +
DB 指纹探针失效"，上传/删除文档后无需重启即可生效。**

## 2. 目标

- 让检索层与"片来自法条还是 PDF"解耦——它只认识切片。
- 多知识库隔离：不同库的片不能互相污染候选。
- 文档增删后 BM25 索引要能跟上（v1 的 `_load_corpus` 只在首次加载，语料变了得重启）。

## 3. 选中的方案

### 身份键与数据模型（`app/rag/models.py`）

| 模型 | v1 | V2.1 |
|---|---|---|
| 语料引用 | `ArticleRef(law_id, article_no, chapter, text)` | `ChunkRef(chunk_id, kb_id, doc_id, doc_title, seq, page, heading, text, source_law_id)` |
| 单路命中 | `LaneHit(law_id, article_no, score)` | `LaneHit(chunk_id, doc_id, seq, score)` |
| 融合命中 | `FusedHit(law_id, article_no, chapter, ...)` | `FusedHit(chunk_id, kb_id, doc_id, doc_title, seq, page, heading, text, source_law_id, ...)` |

- `LaneHit` 带 doc_id/seq 是给评测与调试用的：单路结果也要能对到"哪个文档的哪一段"，
  否则评测只能拿 chunk_id 猜，两路之间的身份也无法对齐。
- `chapter` 退役，由 `heading` 取代（内置库二者同值，见决策 11）。

### 融合与精排（不改逻辑，只换键）

`fusion._Key` 从 `tuple[str,int]` 放宽为 `Hashable`——RRF 只看相等性与顺序，
`chunk_id`（int）天然适配，原 tuple 键的测试也仍通过。rerank 从 `_by_key[chunk_id].text`
取文本，排序与截断逻辑原样。

### kb 作用域贯通

`search(query, top_k, use_rerank, kb_ids)`；`AgentInput/AgentState` 加 `kb_ids`，
`make_retrieve_node` 透传，路由把可见库集合放进 `invoke` 输入。Graph 拓扑零改动
（不加节点、不改边，只加状态字段）。

### BM25 缓存失效（本决策的重点）

```
_cache: dict[scope, _Bm25Entry(signature, by_key, index)]
scope = tuple(sorted(kb_ids))
signature = (count(chunks), max(chunk.id), max(documents.updated_at)) over scope
```

每次检索先用一条廉价聚合查询探指纹；指纹变了就在 `_cache_lock` 内 double-check
后重建索引。

**为什么指纹是这三项**：三条写路径全覆盖——
- 新增切片 → 片数 + max(id) 变；
- 删除切片/删整库（级联）→ 片数变；
- 同文档重传（`ON CONFLICT (doc_id, seq)` 原地更新，**id 不变**）→ `documents.updated_at` 变。

**为什么用探针而不是显式失效通知**：写路径分散在迁移脚本、上传流水线、删除接口三处，
各自通知缓存容易漏；探针在检索侧统一收口。代价是每次检索多一条聚合查询，205 条规模可忽略。

## 4. 备选及为什么不选

| 备选 | 为什么不选 |
|---|---|
| 给缓存加全局版本号，写方显式 bump | 写路径三处都要记得 bump，漏一处就是"检索到已删内容"的静默 bug；探针不依赖写方纪律 |
| 每次检索都全量重建 BM25 索引 | 205 条虽便宜，但每次问答都重算语料 + 分词，白烧 CPU；探针方案命中缓存时只多一条聚合 |
| 用 `chunks.updated_at` 做指纹 | chunks 的原地更新不会自动改时间戳（除非触发器），不如文档级时间戳可靠 |
| 每个库单独一个 retriever 实例 | 多库组合（"我可见的全部库"）无法表达；且实例管理、缓存共享都更复杂 |
| BM25 换 PG tsvector / ES | 当前量级不值得外置（决策 03 已记）；外置是量级上来后的事 |
| 跨库 BM25 各库取 top 后归并成一条 lane | **选中了它**，但排序会偏向前面的库——默认单库主路径不受影响，记为取舍（§6） |

## 5. 为什么这样最好

**换键是局部改动**：融合、精排、拒答核验、引用装配都不关心键的语义，只关心它稳定可哈希。
实测改动集中在 4 个文件、约 120 行，且迁移后评测指标逐位不变（见决策 11 §5）。
**缓存探针把"正确性"从写方纪律转移到读方自检**——这是分布式缓存失效的经典取舍，
在本项目规模下用一条聚合查询买下，性价比极高。

## 6. 已知取舍

1. **每次检索多一条聚合查询**：换取"语料变更自动生效"。若将来 QPS 高到这条查询成为瓶颈，
   可改为写方 bump + 探针降频（如 1 秒内只探一次）。
2. **多库合并 BM25 lane 的排序偏向前库**：跨库检索时 BM25 lane 是各库候选拼接，
   排序对靠前库略有利。默认单库主路径不受影响；彻底解法是每库独立打分后按分归并，
   但两路分数不可比（决策 03），实际收益存疑，暂不做。
3. **竞态保护是"跳过"而非"重取"**：向量路可能召回语料快照之后才入库的片，
   `by_key` 查不到就本轮跳过（下次检索指纹已变会重建并包含它）。代价是极端并发下
   某次结果少一条候选，不会报错也不会返回错内容。
4. **内存占用随库数线性增长**：每个作用域一份倒排索引。演示规模（几个库、几千片）无压力，
   量级上来要评估按 LRU 淘汰或外置。
