class TimelineChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.data = null;
    }

    update(data) {
        this.data = data;
        
        const traces = [];
        
        // Create a trace for each concept
        Object.entries(data.evolution).forEach(([concept, values]) => {
            traces.push({
                name: concept,
                x: values.map(v => v.stage),
                y: values.map(v => v.weight),
                mode: 'lines+markers',
                line: { shape: 'spline' },
                hovertemplate: `
                    <b>${concept}</b><br>
                    Stage: %{x}<br>
                    Weight: %{y:.2f}<br>
                    <extra></extra>
                `
            });
        });

        const layout = {
            title: 'Concept Evolution Timeline',
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
                title: 'Weight',
                gridcolor: '#3d3d3d',
                zeroline: false
            },
            showlegend: true,
            legend: {
                bgcolor: 'rgba(45, 45, 45, 0.8)',
                bordercolor: '#4d4d4d',
                borderwidth: 1
            },
            hovermode: 'closest',
            margin: { t: 40, r: 10, b: 40, l: 60 }
        };

        Plotly.newPlot(this.container, traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    }

    highlightStage(stage) {
        if (!stage || !this.data) return;

        const shapes = [{
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: stage,
            x1: stage,
            y0: 0,
            y1: 1,
            fillcolor: '#4299e1',
            opacity: 0.2,
            line: { width: 0 }
        }];

        Plotly.relayout(this.container, { shapes });
    }
}
