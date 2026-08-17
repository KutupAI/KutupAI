# Launch the interactive RAG retrieval test with a UTF-8 PowerShell console.
# Usage from the repository root:
#   .\RAG\scripts\run_retrieval.ps1
# Highest measured Hit@1 default: vector-only fast precision profile.
# Comprehensive mode: .\RAG\scripts\run_retrieval.ps1 --mode hybrid --prf --reranker

chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python -m RAG.scripts.query_retrieval @args
