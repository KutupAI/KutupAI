# One-command interactive, grounded legal RAG agent.
# From repository root: .\RAG\scripts\run_legal_agent.ps1
chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m RAG.scripts.ask_legal_agent @args
