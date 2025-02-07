console.log('Starting visualization initialization...');

// Initialize network visualization
const network = new NetworkGraph('network-container');
console.log('Network graph initialized');

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
    try {
        const data = JSON.parse(event.data);
        network.update(data, currentStage, null);
    } catch (error) {
        console.error('Error processing WebSocket data:', error);
        network.update({
            nodes: [],
            links: [],
            metadata: {},
            directed: true,
            multigraph: false,
            graph: {}
        }, currentStage, 'Failed to process real-time update');
    }
};

// Initial data fetch
async function fetchVisualizationData() {
    console.log('Fetching visualization data...');
    try {
        console.log('Fetching data from API...');
        const response = await fetch('/api/v1/visualize/latest');
        console.log('API response:', response);
        if (!response.ok) throw new Error('Failed to fetch visualization data');
        
        const data = await response.json();
        console.log('Parsed API response:', data);
        console.log('Network data:', data.network_data);
        network.update(data.network_data, currentStage, null);
    } catch (error) {
        console.error('Error fetching visualization data:', error);
        network.update({
            nodes: [],
            links: [],
            metadata: {},
            directed: true,
            multigraph: false,
            graph: {}
        }, currentStage, error.message || 'Failed to fetch data');
    }
}

// Load initial data
fetchVisualizationData();
