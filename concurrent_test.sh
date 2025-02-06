#!/bin/bash

# Function to create a visualization with a given ID
create_visualization() {
    local id=$1
    curl -X POST http://localhost:8001/api/v1/visualize \
        -H "Content-Type: application/json" \
        -d "{
            \"doc_id\": \"concurrent_test_$id\",
            \"temporal_stages\": [\"stage$id\"],
            \"concept_evolution\": {\"concept$id\": [\"stage$id\"]},
            \"relationship_changes\": [{\"from\": \"concept$id\", \"to\": \"concept$(($id+1))\"}],
            \"coherence_metrics\": {\"metric$id\": 0.$(($id * 2))5}
        }" &
}

# Create multiple visualizations concurrently
for i in {1..5}; do
    create_visualization $i
done

# Wait for all requests to complete
wait
