/**
 * Topological-Temporal Expression Visualizer
 * 
 * This script provides the interactive visualization of the semantic field
 * and enables expression generation from areas of high potential.
 */

// Main visualization class
class TopologicalTemporalVisualizer {
    constructor() {
        // DOM elements
        this.container = document.getElementById('visualization-container');
        this.potentialThreshold = document.getElementById('potential-threshold');
        this.thresholdValue = document.getElementById('threshold-value');
        this.generateButton = document.getElementById('generate-expression');
        this.resetButton = document.getElementById('reset-view');
        this.expressionContainer = document.getElementById('expression-container');
        this.expressionTemplate = document.getElementById('expression-template');
        this.conceptInfo = document.getElementById('concept-info');
        
        // Visualization state
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.nodes = [];
        this.links = [];
        this.selectedNodes = [];
        this.currentMode = 'discover';
        this.currentDomain = 'all';
        this.showDissonance = true;
        this.potentialThresholdValue = 0.3;
        
        // D3 elements
        this.svg = null;
        this.simulation = null;
        this.nodeElements = null;
        this.linkElements = null;
        this.dissonanceZones = null;
        
        // Initialize the visualization
        this.initVisualization();
        this.initEventListeners();
        this.loadData();
        this.updatePotentialMetrics();
    }
    
    /**
     * Initialize the D3 visualization
     */
    initVisualization() {
        // Create SVG container
        this.svg = d3.select('#visualization-container')
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', [0, 0, this.width, this.height]);
        
        // Create groups for different elements
        this.svg.append('g').attr('class', 'dissonance-zones');
        this.svg.append('g').attr('class', 'links');
        this.svg.append('g').attr('class', 'nodes');
        this.svg.append('g').attr('class', 'intentionality-vectors');
        
        // Create force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));
    }
    
    /**
     * Initialize event listeners for UI controls
     */
    initEventListeners() {
        // Mode selection
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.updateVisualization();
            });
        });
        
        // Domain filter
        document.querySelectorAll('input[name="domain"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentDomain = e.target.value;
                this.updateVisualization();
            });
        });
        
        // Potential threshold
        this.potentialThreshold.addEventListener('input', (e) => {
            this.potentialThresholdValue = parseFloat(e.target.value);
            this.thresholdValue.textContent = this.potentialThresholdValue.toFixed(2);
            this.updateVisualization();
        });
        
        // Dissonance visibility
        document.getElementById('show-dissonance').addEventListener('change', (e) => {
            this.showDissonance = e.target.checked;
            this.updateVisualization();
        });
        
        // Generate expression button
        this.generateButton.addEventListener('click', () => {
            this.generateExpression();
        });
        
        // Reset view button
        this.resetButton.addEventListener('click', () => {
            this.resetView();
        });
        
        // Window resize handler
        window.addEventListener('resize', () => {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.updateVisualization();
        });
    }
    
    /**
     * Load semantic field data from the API
     */
    loadData() {
        // Show loading indicator
        this.showLoading(true);
        
        // Fetch semantic field data
        fetch('/api/semantic-field')
            .then(response => response.json())
            .then(data => {
                this.nodes = data.nodes;
                this.links = data.links;
                this.updateVisualization();
                this.showLoading(false);
            })
            .catch(error => {
                console.error('Error loading semantic field data:', error);
                this.showLoading(false);
                
                // Use sample data if API fails
                this.loadSampleData();
            });
    }
    
    /**
     * Load sample data for demonstration
     */
    loadSampleData() {
        // Sample nodes (concepts and predicates)
        this.nodes = [];
        this.links = [];
        
        // Create sample concepts
        const domains = ['social', 'environmental'];
        const types = ['concept', 'predicate'];
        const conceptNames = [
            'Climate Change', 'Social Justice', 'Biodiversity', 'Cultural Heritage',
            'Sustainability', 'Community Resilience', 'Ecosystem Services', 'Indigenous Knowledge',
            'Urban Development', 'Water Resources', 'Public Health', 'Energy Transition',
            'Food Security', 'Education Access', 'Land Rights', 'Technological Innovation'
        ];
        
        // Generate nodes
        for (let i = 0; i < conceptNames.length; i++) {
            const domain = domains[i % 2];
            const type = types[i % 3 === 0 ? 1 : 0]; // Make some predicates
            
            this.nodes.push({
                id: `node-${i}`,
                name: conceptNames[i],
                type: type,
                domain: domain,
                potential: 0.3 + Math.random() * 0.7,
                constructive_dissonance: Math.random() * 0.5,
                x: 100 + Math.random() * (this.width - 200),
                y: 100 + Math.random() * (this.height - 200)
            });
        }
        
        // Generate links
        for (let i = 0; i < this.nodes.length; i++) {
            // Create 2-3 links per node
            const numLinks = 2 + Math.floor(Math.random() * 2);
            
            for (let j = 0; j < numLinks; j++) {
                const targetIndex = Math.floor(Math.random() * this.nodes.length);
                
                // Avoid self-links and duplicates
                if (targetIndex !== i) {
                    this.links.push({
                        source: this.nodes[i].id,
                        target: this.nodes[targetIndex].id,
                        strength: 0.1 + Math.random() * 0.9
                    });
                }
            }
        }
        
        this.updateVisualization();
    }
    
    /**
     * Update the visualization based on current state
     */
    updateVisualization() {
        // Filter nodes based on domain and potential threshold
        const filteredNodes = this.nodes.filter(node => {
            const domainMatch = this.currentDomain === 'all' || node.domain === this.currentDomain;
            const potentialMatch = node.potential >= this.potentialThresholdValue;
            return domainMatch && potentialMatch;
        });
        
        // Filter links to only include connections between visible nodes
        const nodeIds = new Set(filteredNodes.map(node => node.id));
        const filteredLinks = this.links.filter(link => 
            nodeIds.has(link.source.id || link.source) && 
            nodeIds.has(link.target.id || link.target)
        );
        
        // Update links
        this.linkElements = this.svg.select('.links')
            .selectAll('.link')
            .data(filteredLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
            
        this.linkElements.exit().remove();
        
        const linkEnter = this.linkElements.enter()
            .append('line')
            .attr('class', 'link');
            
        this.linkElements = linkEnter.merge(this.linkElements)
            .style('stroke-width', d => d.strength * 3)
            .style('stroke-opacity', d => 0.2 + d.strength * 0.6);
        
        // Update nodes
        this.nodeElements = this.svg.select('.nodes')
            .selectAll('.node')
            .data(filteredNodes, d => d.id);
            
        this.nodeElements.exit().remove();
        
        const nodeEnter = this.nodeElements.enter()
            .append('g')
            .attr('class', d => `node domain-${d.domain} type-${d.type}`)
            .call(this.drag())
            .on('click', (event, d) => this.handleNodeClick(event, d))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
            
        nodeEnter.append('circle')
            .attr('r', d => 5 + d.potential * 15);
            
        nodeEnter.append('text')
            .attr('dx', d => 8 + d.potential * 10)
            .attr('dy', '.35em')
            .text(d => d.name);
            
        this.nodeElements = nodeEnter.merge(this.nodeElements);
        
        // Update node appearance based on potential
        this.nodeElements.select('circle')
            .attr('r', d => 5 + d.potential * 15)
            .style('opacity', d => 0.7 + d.potential * 0.3);
            
        // Update constructive dissonance zones
        if (this.showDissonance) {
            const dissonanceNodes = filteredNodes.filter(node => node.constructive_dissonance > 0.3);
            
            this.dissonanceZones = this.svg.select('.dissonance-zones')
                .selectAll('.dissonance-zone')
                .data(dissonanceNodes, d => d.id);
                
            this.dissonanceZones.exit().remove();
            
            const dissonanceEnter = this.dissonanceZones.enter()
                .append('circle')
                .attr('class', 'dissonance-zone');
                
            this.dissonanceZones = dissonanceEnter.merge(this.dissonanceZones)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y)
                .attr('r', d => 20 + d.constructive_dissonance * 50)
                .style('opacity', d => d.constructive_dissonance * 0.3);
        } else {
            this.svg.select('.dissonance-zones').selectAll('.dissonance-zone').remove();
        }
        
        // Update simulation
        this.simulation.nodes(filteredNodes)
            .force('link').links(filteredLinks);
            
        this.simulation.alpha(0.3).restart();
        
        // Update simulation tick handler
        this.simulation.on('tick', () => {
            // Update link positions
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            // Update node positions
            this.nodeElements
                .attr('transform', d => `translate(${d.x},${d.y})`);
                
            // Update dissonance zone positions
            if (this.showDissonance) {
                this.dissonanceZones
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
            }
        });
        
        // Update generate button state
        this.generateButton.disabled = this.selectedNodes.length === 0;
    }
    
    /**
     * Handle node click event
     */
    handleNodeClick(event, node) {
        // Toggle node selection
        const index = this.selectedNodes.findIndex(n => n.id === node.id);
        
        if (index === -1) {
            // Add to selection
            this.selectedNodes.push(node);
            d3.select(event.currentTarget).classed('selected', true);
        } else {
            // Remove from selection
            this.selectedNodes.splice(index, 1);
            d3.select(event.currentTarget).classed('selected', false);
        }
        
        // Update generate button state
        this.generateButton.disabled = this.selectedNodes.length === 0;
        
        // Update concept info panel
        this.updateConceptInfo(node);
    }
    
    /**
     * Update the concept info panel with details about the selected node
     */
    updateConceptInfo(node) {
        // Create concept info HTML
        const html = `
            <h6>${node.name}</h6>
            <p class="mb-1"><strong>Type:</strong> ${node.type}</p>
            <p class="mb-1"><strong>Domain:</strong> ${node.domain}</p>
            <p class="mb-1"><strong>Potential:</strong> ${node.potential.toFixed(2)}</p>
            <p class="mb-0"><strong>Constructive Dissonance:</strong> ${node.constructive_dissonance.toFixed(2)}</p>
        `;
        
        this.conceptInfo.innerHTML = html;
    }
    
    /**
     * Generate an expression based on selected nodes
     */
    generateExpression() {
        // Get selected node IDs
        const conceptIds = this.selectedNodes.map(node => node.id);
        
        // Get current intentionality mode
        const intentionality = this.currentMode;
        
        // Show loading state
        this.generateButton.disabled = true;
        this.generateButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
        
        // Call the expression generation API
        fetch('/api/generate-expression', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                concept_ids: conceptIds,
                intentionality: intentionality
            })
        })
        .then(response => response.json())
        .then(data => {
            // Add the expression to the UI
            this.addExpression(data);
            
            // Draw intentionality vector
            this.drawIntentionalityVector(data);
            
            // Reset button state
            this.generateButton.disabled = false;
            this.generateButton.textContent = 'Generate Expression';
        })
        .catch(error => {
            console.error('Error generating expression:', error);
            
            // Reset button state
            this.generateButton.disabled = false;
            this.generateButton.textContent = 'Generate Expression';
            
            // Generate a sample expression for demonstration
            this.addSampleExpression();
        });
    }
    
    /**
     * Add a generated expression to the UI
     */
    addExpression(data) {
        // Clone the expression template
        const expressionElement = this.expressionTemplate.cloneNode(true);
        expressionElement.id = '';
        expressionElement.classList.remove('d-none');
        
        // Fill in expression data
        expressionElement.querySelector('.expression-text').textContent = data.expression;
        expressionElement.querySelector('.expression-intentionality').textContent = data.intentionality;
        expressionElement.querySelector('.expression-potential').textContent = `Potential: ${data.potential.toFixed(2)}`;
        
        // Add to container
        this.expressionContainer.insertBefore(expressionElement, this.expressionContainer.firstChild);
    }
    
    /**
     * Add a sample expression for demonstration
     */
    addSampleExpression() {
        // Sample expressions for different intentionality types
        const expressions = {
            'discover': 'The interconnected nature of climate patterns reveals emergent properties within socio-ecological systems.',
            'create': 'Through indigenous knowledge frameworks, new approaches to environmental stewardship emerge.',
            'evolve': 'As cultural practices adapt to changing environments, their relationship with ecological processes transforms.',
            'connect': 'Social justice bridges the gap between human well-being and environmental sustainability.'
        };
        
        // Create sample expression data
        const data = {
            expression: expressions[this.currentMode] || expressions.discover,
            intentionality: this.currentMode,
            potential: 0.7 + Math.random() * 0.3,
            intentionality_vector: {
                x: Math.random() - 0.5,
                y: Math.random() - 0.5,
                magnitude: 0.5 + Math.random() * 0.5
            }
        };
        
        // Add to UI
        this.addExpression(data);
        
        // Draw intentionality vector
        this.drawIntentionalityVector(data);
    }
    
    /**
     * Draw an intentionality vector on the visualization
     */
    drawIntentionalityVector(data) {
        // Remove any existing vectors
        this.svg.select('.intentionality-vectors').selectAll('*').remove();
        
        // Calculate vector position
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        
        // Calculate vector endpoints
        const vectorLength = data.intentionality_vector.magnitude * 100;
        const endX = centerX + data.intentionality_vector.x * vectorLength;
        const endY = centerY + data.intentionality_vector.y * vectorLength;
        
        // Draw the vector
        this.svg.select('.intentionality-vectors')
            .append('line')
            .attr('class', 'intentionality-vector')
            .attr('x1', centerX)
            .attr('y1', centerY)
            .attr('x2', endX)
            .attr('y2', endY)
            .attr('marker-end', 'url(#arrow)');
            
        // Add arrowhead marker if it doesn't exist
        if (this.svg.select('defs').empty()) {
            this.svg.append('defs')
                .append('marker')
                .attr('id', 'arrow')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 8)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', '#ff4500');
        }
    }
    
    /**
     * Update potential metrics in the UI
     */
    updatePotentialMetrics() {
        // Fetch potential metrics from API
        fetch('/api/potential')
            .then(response => response.json())
            .then(data => {
                // Update progress bars
                this.updateMetricBar('evolutionary-potential', 
                    data.field_potential.avg_evolutionary_potential);
                this.updateMetricBar('constructive-dissonance', 
                    data.field_potential.avg_constructive_dissonance);
                this.updateMetricBar('topological-energy', 
                    data.topological_potential.topological_energy);
                this.updateMetricBar('manifold-curvature', 
                    data.topological_potential.manifold_curvature.average_curvature);
            })
            .catch(error => {
                console.error('Error loading potential metrics:', error);
                
                // Use sample metrics
                this.updateMetricBar('evolutionary-potential', 0.72);
                this.updateMetricBar('constructive-dissonance', 0.48);
                this.updateMetricBar('topological-energy', 0.65);
                this.updateMetricBar('manifold-curvature', 0.32);
            });
    }
    
    /**
     * Update a metric progress bar
     */
    updateMetricBar(id, value) {
        // Update progress bar
        const element = document.getElementById(id);
        if (element) {
            const percentage = value * 100;
            element.style.width = `${percentage}%`;
            element.setAttribute('aria-valuenow', percentage);
        }
        
        // Update value display
        const valueElement = document.getElementById(`${id}-value`);
        if (valueElement) {
            valueElement.textContent = value.toFixed(2);
        }
    }
    
    /**
     * Show or hide loading indicator
     */
    showLoading(show) {
        // Remove existing indicator if any
        const existing = document.querySelector('.loading-indicator');
        if (existing) {
            existing.remove();
        }
        
        if (show) {
            // Create loading indicator
            const indicator = document.createElement('div');
            indicator.className = 'loading-indicator';
            indicator.innerHTML = `
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading semantic field...</p>
            `;
            
            this.container.appendChild(indicator);
        }
    }
    
    /**
     * Show tooltip with node information
     */
    showTooltip(event, d) {
        // Remove any existing tooltip
        this.hideTooltip();
        
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.innerHTML = `
            <div><strong>${d.name}</strong></div>
            <div>Type: ${d.type}</div>
            <div>Domain: ${d.domain}</div>
            <div>Potential: ${d.potential.toFixed(2)}</div>
        `;
        
        // Position tooltip near mouse
        tooltip.style.left = `${event.pageX + 10}px`;
        tooltip.style.top = `${event.pageY + 10}px`;
        
        // Add to document
        document.body.appendChild(tooltip);
    }
    
    /**
     * Hide tooltip
     */
    hideTooltip() {
        const tooltip = document.querySelector('.tooltip');
        if (tooltip) {
            tooltip.remove();
        }
    }
    
    /**
     * Reset the visualization view
     */
    resetView() {
        // Clear selected nodes
        this.selectedNodes = [];
        this.nodeElements.classed('selected', false);
        
        // Reset domain and mode
        document.getElementById('all-domains').checked = true;
        document.getElementById('discover-mode').checked = true;
        this.currentDomain = 'all';
        this.currentMode = 'discover';
        
        // Reset potential threshold
        this.potentialThreshold.value = 0.3;
        this.potentialThresholdValue = 0.3;
        this.thresholdValue.textContent = '0.3';
        
        // Show dissonance
        document.getElementById('show-dissonance').checked = true;
        this.showDissonance = true;
        
        // Remove intentionality vectors
        this.svg.select('.intentionality-vectors').selectAll('*').remove();
        
        // Reset concept info
        this.conceptInfo.innerHTML = `
            <div class="alert alert-secondary">
                Click on a concept to see details.
            </div>
        `;
        
        // Update visualization
        this.updateVisualization();
        
        // Disable generate button
        this.generateButton.disabled = true;
    }
    
    /**
     * Create D3 drag behavior
     */
    drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
}

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const visualizer = new TopologicalTemporalVisualizer();
    
    // Update potential threshold value display
    document.getElementById('potential-threshold').addEventListener('input', (e) => {
        document.getElementById('threshold-value').textContent = parseFloat(e.target.value).toFixed(2);
    });
});
