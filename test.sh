curl -X POST -s http://localhost:8100/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen3-0.6B",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}'