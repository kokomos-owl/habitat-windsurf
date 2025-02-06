// Initialize network visualization
const network = new NetworkGraph('network-container');

// Stage selector functionality
const stageButtons = document.querySelectorAll('.stage-button');
let currentStage = null;

stageButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Update active state
        stageButtons.forEach(b => b.classList.remove('active'));
        button.classList.add('active');

        // Update visualization
        const stage = button.dataset.stage;
        currentStage = stage === 'all' ? null : stage;
        
        // Fetch and update data
        fetchVisualizationData();
    });
});

// Control buttons functionality
document.getElementById('toggle-labels').addEventListener('click', () => {
    const labels = document.querySelectorAll('.node text');
    labels.forEach(label => {
        label.style.display = label.style.display === 'none' ? 'block' : 'none';
    });
});

document.getElementById('reset-zoom').addEventListener('click', () => {
    network.svg.transition()
        .duration(750)
        .call(network.zoom.transform, d3.zoomIdentity);
});

document.getElementById('export-graph').addEventListener('click', () => {
    const svgData = new XMLSerializer().serializeToString(network.svg.node());
    const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
    const url = URL.createObjectURL(svgBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'knowledge_graph.svg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/api/v1/ws/client1`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    network.update(data, currentStage);
};

// Initial data fetch
async function fetchVisualizationData() {
    try {
        const response = await fetch('/api/v1/visualize/latest');
        if (!response.ok) throw new Error('Failed to fetch visualization data');
        
        const data = await response.json();
        network.update(data, currentStage);
    } catch (error) {
        console.error('Error fetching visualization data:', error);
    }
}

// Load initial data
fetchVisualizationData();
