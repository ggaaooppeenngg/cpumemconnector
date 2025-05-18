curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen3-0.6B",
"prompt": "San Francisco is a amazing city. It is known for its iconic landmarks such as the Golden Gate Bridge and Alcatraz Island. The city is also famous for its diverse culture, vibrant neighborhoods, and stunning views of the Bay Area. Visitors can enjoy a variety of activities, including exploring Fisherman Wharf, taking a cable car ride, and visiting the historic Painted Ladies. With its rich history and beautiful scenery, San Francisco is a must-visit destination for travelers.",
"max_tokens": 100,
"temperature": 0
}'