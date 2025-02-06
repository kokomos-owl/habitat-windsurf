class CoherenceChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.data = null;
    }

    update(data) {
        this.data = data;
        
        const trace = {
            x: Object.keys(data.metrics),
            y: Object.values(data.metrics),
            type: 'bar',
            marker: {
                color: '#4299e1',
                opacity: 0.8
            },
            hovertemplate: `
                Stage: %{x}<br>
                Coherence: %{y:.2f}<br>
                <extra></extra>
            `
        };

        const layout = {
            title: 'Coherence Metrics',
            plot_bgcolor: '#2d2d2d',
            paper_bgcolor: '#2d2d2d',
            font: {
                color: '#e0e0e0'
            },
            xaxis: {
                title: 'Stage',
                gridcolor: '#3d3d3d',
                zeroline: false
            },
            yaxis: {
                title: 'Coherence Score',
                gridcolor: '#3d3d3d',
                zeroline: false,
                range: [0, 1]
            },
            showlegend: false,
            margin: { t: 40, r: 10, b: 40, l: 60 }
        };

        Plotly.newPlot(this.container, [trace], layout, {
            responsive: true,
            displayModeBar: false
        });
    }

    highlightStage(stage) {
        if (!stage || !this.data) return;

        const colors = Object.keys(this.data.metrics).map(s => 
            s === stage ? '#63b3ed' : '#4299e1'
        );

        Plotly.restyle(this.container, {
            'marker.color': [colors]
        });
    }
}
