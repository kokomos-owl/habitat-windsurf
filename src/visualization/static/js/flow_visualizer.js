/**
 * Flow-based metric visualization using D3.js
 */

class FlowVisualizer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            width: options.width || 800,
            height: options.height || 600,
            margin: options.margin || { top: 20, right: 20, bottom: 30, left: 40 },
            transitionDuration: options.transitionDuration || 750,
            colors: options.colors || d3.schemeCategory10
        };
        
        this.svg = null;
        this.flowData = [];
        this.metricTypes = new Set();
        this.initialize();
    }
    
    initialize() {
        // Create SVG container
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height);
            
        // Add gradient definitions
        this.defineGradients();
        
        // Initialize components
        this.flowGroup = this.svg.append('g')
            .attr('class', 'flow-group')
            .attr('transform', `translate(${this.options.margin.left},${this.options.margin.top})`);
            
        this.tooltipDiv = d3.select('body').append('div')
            .attr('class', 'flow-tooltip')
            .style('opacity', 0);
    }
    
    defineGradients() {
        // Define gradients for flow visualization
        const defs = this.svg.append('defs');
        
        // Flow gradient
        const flowGradient = defs.append('linearGradient')
            .attr('id', 'flow-gradient')
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%');
            
        flowGradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#2196F3')
            .attr('stop-opacity', 0.8);
            
        flowGradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#2196F3')
            .attr('stop-opacity', 0.2);
            
        // Confidence gradient
        const confidenceGradient = defs.append('linearGradient')
            .attr('id', 'confidence-gradient')
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', '0%')
            .attr('y1', '100%')
            .attr('x2', '0%')
            .attr('y2', '0%');
            
        confidenceGradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#FF5722');
            
        confidenceGradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#4CAF50');
    }
    
    updateData(flowData) {
        this.flowData = flowData;
        this.metricTypes = new Set(flowData.map(d => d.metricType));
        this.render();
    }
    
    render() {
        const width = this.options.width - this.options.margin.left - this.options.margin.right;
        const height = this.options.height - this.options.margin.top - this.options.margin.bottom;
        
        // Create scales
        const xScale = d3.scaleTime()
            .domain(d3.extent(this.flowData, d => d.timestamp))
            .range([0, width]);
            
        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([height, 0]);
            
        // Create flow paths
        const line = d3.line()
            .x(d => xScale(d.timestamp))
            .y(d => yScale(d.confidence))
            .curve(d3.curveBasis);
            
        // Update flows
        const flows = this.flowGroup.selectAll('.flow-path')
            .data(Array.from(this.metricTypes).map(type => ({
                type,
                values: this.flowData.filter(d => d.metricType === type)
            })));
            
        // Enter new flows
        flows.enter()
            .append('path')
            .attr('class', 'flow-path')
            .merge(flows)
            .style('stroke', (d, i) => this.options.colors[i])
            .style('fill', 'none')
            .style('stroke-width', 2)
            .transition()
            .duration(this.options.transitionDuration)
            .attr('d', d => line(d.values));
            
        // Exit flows
        flows.exit().remove();
        
        // Add confidence indicators
        const confidenceMarkers = this.flowGroup.selectAll('.confidence-marker')
            .data(this.flowData);
            
        confidenceMarkers.enter()
            .append('circle')
            .attr('class', 'confidence-marker')
            .merge(confidenceMarkers)
            .attr('cx', d => xScale(d.timestamp))
            .attr('cy', d => yScale(d.confidence))
            .attr('r', 4)
            .style('fill', d => d3.interpolateRdYlGn(d.confidence))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
            
        confidenceMarkers.exit().remove();
        
        // Add axes
        const xAxis = d3.axisBottom(xScale);
        const yAxis = d3.axisLeft(yScale);
        
        this.flowGroup.selectAll('.x-axis').remove();
        this.flowGroup.selectAll('.y-axis').remove();
        
        this.flowGroup.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);
            
        this.flowGroup.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);
    }
    
    showTooltip(event, d) {
        this.tooltipDiv.transition()
            .duration(200)
            .style('opacity', .9);
            
        this.tooltipDiv.html(`
            <strong>${d.metricType}</strong><br/>
            Confidence: ${(d.confidence * 100).toFixed(1)}%<br/>
            Viscosity: ${(d.viscosity * 100).toFixed(1)}%<br/>
            Density: ${(d.density * 100).toFixed(1)}%<br/>
            Stability: ${(d.stability * 100).toFixed(1)}%
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 28) + 'px');
    }
    
    hideTooltip() {
        this.tooltipDiv.transition()
            .duration(500)
            .style('opacity', 0);
    }
    
    updateDimensions(width, height) {
        this.options.width = width;
        this.options.height = height;
        
        this.svg
            .attr('width', width)
            .attr('height', height);
            
        this.render();
    }
}
