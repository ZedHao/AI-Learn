#
./scripts/run-backend.sh  aiBaseModel/qwen3-vl/8b-instruct
./scripts/run-front.sh


# read
模块	作用
config.py  doc/caibao 路径、缓存目录、分块大小、ES 地址与索引名、嵌入模型名
pdf_text.py 用 pypdf 从 PDF 抽文本
chunking.py 固定窗口分块（带重叠）[meta.json](../.rag_cache/faiss/meta.json)
embedder.py sentence-transformers 句向量（默认多语言 MiniLM，384 维，L2 归一化）
faiss_store.py FAISS IndexFlatIP 持久化与检索
es_store.py Elasticsearch 8：dense_vector + kNN 建索引、bulk 写入、检索
service.py 扫描 PDF → 分块 → 嵌入 → 写 FAISS / 可选 ES；retrieve；inject_rag_into_messages