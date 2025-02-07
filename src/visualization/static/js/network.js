class NetworkGraph {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.width = this.container.node().getBoundingClientRect().width;
        this.height = this.container.node().getBoundingClientRect().height;
        this.zoom = d3.zoom().on('zoom', this.handleZoom.bind(this));
        this.tooltip = d3.select('#tooltip');
        this.initializeSVG();
        this.simulation = this.initializeSimulation();
        this.currentStage = null;
    }

    initializeSVG() {
        // Create SVG with dark theme grid background
        this.svg = this.container.append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .style('background-color', '#1e1e1e')
            .call(this.zoom);

        // Add grid pattern
        const defs = this.svg.append('defs');
        const pattern = defs.append('pattern')
            .attr('id', 'grid')
            .attr('width', 25)
            .attr('height', 25)
            .attr('patternUnits', 'userSpaceOnUse');

        pattern.append('path')
            .attr('d', 'M 25 0 L 0 0 0 25')
            .style('fill', 'none')
            .style('stroke', '#2d2d2d')
            .style('stroke-width', '1');

        this.svg.append('rect')
            .attr('width', '100%')
            .attr('height', '100%')
            .style('fill', 'url(#grid)');

        this.g = this.svg.append('g');
        this.linksGroup = this.g.append('g').attr('class', 'links');
        this.nodesGroup = this.g.append('g').attr('class', 'nodes');
    }

    initializeSimulation() {
        return d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-500))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(40));
    }

    handleZoom(event) {
        this.g.attr('transform', event.transform);
    }

    update(data, stage, error = null) {
        console.log('Updating graph with data:', data);
        console.log('Current stage:', stage);
        
        this.currentStage = stage;
        this.metadata = data?.metadata || {};
        
        // Update error box
        this.updateErrorBox(error);
        
        // Ensure data has required properties
        if (!data || !data.links || !data.nodes) {
            console.error('Invalid data structure:', data);
            return;
        }
        
        console.log('Filtered nodes:', data.nodes);
        console.log('Filtered links:', data.links);
        
        // Filter data for current stage if needed
        const filteredLinks = data.links.filter(link => !stage || link.stage === stage);
        const nodeIds = new Set([
            ...filteredLinks.map(l => l.source),
            ...filteredLinks.map(l => l.target)
        ]);
        const filteredNodes = data.nodes.filter(node => nodeIds.has(node.id));

        // Update links
        const links = this.linksGroup.selectAll('.link')
            .data(filteredLinks, d => `${d.source}-${d.target}`);

        links.exit().remove();

        const linksEnter = links.enter()
            .append('line')
            .attr('class', 'link')
            .style('stroke', '#4a5568')
            .style('stroke-opacity', 0.6)
            .style('stroke-width', d => Math.sqrt(d.weight) * 2);

        // Update nodes
        const nodes = this.nodesGroup.selectAll('.node')
            .data(filteredNodes, d => d.id);

        nodes.exit().remove();

        const nodesEnter = nodes.enter()
            .append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', this.dragStarted.bind(this))
                .on('drag', this.dragged.bind(this))
                .on('end', this.dragEnded.bind(this)));

        // Add node circles with glowing effect
        const circles = nodesEnter.append('g');
        
        circles.append('circle')
            .attr('r', 20)
            .attr('class', 'glow')
            .style('fill', '#4299e1')
            .style('filter', 'url(#glow)');

        circles.append('circle')
            .attr('r', 15)
            .style('fill', '#63b3ed')
            .style('stroke', '#2b6cb0')
            .style('stroke-width', '2px');

        nodesEnter.append('text')
            .attr('dy', -25)
            .attr('text-anchor', 'middle')
            .style('fill', '#e0e0e0')
            .style('font-size', '12px')
            .text(d => d.id);

        // Add interaction handlers
        nodesEnter
            .on('mouseover', this.showTooltip.bind(this))
            .on('mouseout', this.hideTooltip.bind(this));

        // Add glow filter
        const defs = this.svg.select('defs');
        const filter = defs.append('filter')
            .attr('id', 'glow');

        filter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur');

        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode')
            .attr('in', 'coloredBlur');
        feMerge.append('feMergeNode')
            .attr('in', 'SourceGraphic');

        // Update simulation
        console.log('Updating simulation with nodes:', filteredNodes);
        console.log('Updating simulation with links:', filteredLinks);
        
        this.simulation
            .nodes(filteredNodes)
            .force('link').links(filteredLinks);

        this.simulation.alpha(1).restart();

        // Update positions on tick
        this.simulation.on('tick', () => {
            this.linksGroup.selectAll('.link')
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            this.nodesGroup.selectAll('.node')
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    showTooltip(event, d) {
        const confidence = d.confidence || this.metadata.confidence_threshold || 0.95;
        const content = `
            <div class="tooltip-header">${confidence * 100}% confidence</div>
            <div class="tooltip-content">
                <div>Type: ${d.type || 'concept'}</div>
                <div>Weight: ${(d.weight || 1.0).toFixed(2)}</div>
                ${this.currentStage ? `<div>Stage: ${this.currentStage}</div>` : ''}
            </div>
        `;

        this.tooltip
            .style('display', 'block')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(content);
    }

    hideTooltip() {
        this.tooltip.style('display', 'none');
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    updateErrorBox(error) {
        const steps = [
            { id: 'data', message: 'Loading visualization data' },
            { id: 'nodes', message: 'Processing network nodes' },
            { id: 'links', message: 'Processing network links' },
            { id: 'simulation', message: 'Running force simulation' }
        ];

        // Create error box if it doesn't exist
        let errorBox = d3.select('#error-box');
        if (errorBox.empty()) {
            errorBox = this.container.append('div')
                .attr('id', 'error-box')
                .attr('class', 'error-box');

            const stepList = errorBox.append('ul')
                .attr('class', 'step-list');

            steps.forEach(step => {
                const item = stepList.append('li')
                    .attr('class', 'step-item')
                    .attr('data-step', step.id);

                item.append('div')
                    .attr('class', 'status-indicator');

                item.append('p')
                    .attr('class', 'step-message')
                    .text(step.message);
            });
        }

        // Update step statuses
        if (error) {
            // Find which step failed based on error message
            const failedStep = error.toLowerCase().includes('data') ? 'data' :
                             error.toLowerCase().includes('node') ? 'nodes' :
                             error.toLowerCase().includes('link') ? 'links' : 'simulation';

            steps.forEach(step => {
                const indicator = errorBox.select(`[data-step="${step.id}"] .status-indicator`);
                if (step.id === failedStep) {
                    indicator.attr('class', 'status-indicator error');
                    errorBox.select(`[data-step="${step.id}"] .step-message`)
                        .text(`${step.message} - ${error}`);
                } else if (steps.findIndex(s => s.id === step.id) < steps.findIndex(s => s.id === failedStep)) {
                    indicator.attr('class', 'status-indicator success');
                    errorBox.select(`[data-step="${step.id}"] .step-message`)
                        .text(step.message);
                } else {
                    indicator.attr('class', 'status-indicator');
                }
            });
        } else {
            // All steps succeeded
            steps.forEach(step => {
                errorBox.select(`[data-step="${step.id}"] .status-indicator`)
                    .attr('class', 'status-indicator success');
                errorBox.select(`[data-step="${step.id}"] .step-message`)
                    .text(step.message);
            });
        }
    }

    resetZoom() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    zoomIn() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.scaleBy, 1.2);
    }

    zoomOut() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.scaleBy, 0.8);
    }
}
