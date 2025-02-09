import { VectorFieldState, CriticalPoint, FieldIndicator, PatternState, CollapseWarning } from './types';
import { select } from 'd3-selection';
import { scaleLinear, scaleSequential } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';
import { line } from 'd3-shape';

export class TopologyVisualizer {
    private svg: any;
    private width: number;
    private height: number;
    private config: VisualizerConfig;

    constructor(containerId: string, width: number, height: number, config?: Partial<VisualizerConfig>) {
        this.width = width;
        this.height = height;
        this.config = {
            ...defaultConfig,
            ...config
        };

        this.svg = select(`#${containerId}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width/20},${height/20})`);

        // Initialize layers
        this.svg.append('g').attr('class', 'vector-field-layer');
        this.svg.append('g').attr('class', 'critical-points-layer');
        this.svg.append('g').attr('class', 'warning-layer');
    }

    public updateVectorField(field: FieldIndicator): void {
        const vectorLayer = this.svg.select('.vector-field-layer');
        
        // Clear previous field
        vectorLayer.selectAll('*').remove();

        // Generate arrow field
        const arrows = this.generateArrowField(field);
        
        // Draw streamlines
        const streamlines = this.generateStreamlines(field);
        
        vectorLayer.selectAll('path.streamline')
            .data(streamlines)
            .enter()
            .append('path')
            .attr('class', 'streamline')
            .attr('d', line())
            .style('stroke', this.getFieldColor(field.divergence))
            .style('stroke-width', this.config.baseThickness * (1 + Math.abs(field.divergence)))
            .style('fill', 'none')
            .style('opacity', 0.6);

        // Add arrows
        vectorLayer.selectAll('path.arrow')
            .data(arrows)
            .enter()
            .append('path')
            .attr('class', 'arrow')
            .attr('d', this.generateArrowPath)
            .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.angle})`)
            .style('fill', this.getFieldColor(field.divergence))
            .style('opacity', 0.8);
    }

    public updateCriticalPoints(points: CriticalPoint[]): void {
        const pointLayer = this.svg.select('.critical-points-layer');
        
        // Clear previous points
        pointLayer.selectAll('*').remove();

        // Add critical points
        points.forEach(point => {
            const visual = this.renderCriticalPoint(point);
            
            const group = pointLayer.append('g')
                .attr('class', `critical-point ${point.type}`)
                .attr('transform', `translate(${point.position[0]},${point.position[1]})`);

            // Add glyph
            group.append('text')
                .attr('class', 'glyph')
                .text(visual.glyph)
                .style('font-size', `${visual.size}px`)
                .style('fill', visual.color)
                .style('text-anchor', 'middle')
                .style('dominant-baseline', 'middle');

            // Add pulse animation if needed
            if (visual.pulseRate > 0) {
                group.select('.glyph')
                    .style('animation', `pulse ${1000/visual.pulseRate}ms infinite`);
            }
        });
    }

    public showCollapseWarning(warning: CollapseWarning): void {
        const warningLayer = this.svg.select('.warning-layer');
        
        // Clear previous warnings
        warningLayer.selectAll('*').remove();

        if (!warning) return;

        const visual = this.generateWarningVisual(warning);
        
        // Add warning border
        warningLayer.append('rect')
            .attr('class', 'warning-border')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', this.width)
            .attr('height', this.height)
            .style('fill', 'none')
            .style('stroke', visual.border.color)
            .style('stroke-width', visual.border.width)
            .style('stroke-dasharray', '5,5')
            .style('opacity', 0.8);

        // Add warning icon
        const iconGroup = warningLayer.append('g')
            .attr('class', 'warning-icon')
            .attr('transform', `translate(${this.width-40},20)`);

        iconGroup.append('text')
            .text(visual.warning.icon)
            .attr('font-size', visual.warning.size)
            .style('animation', visual.warning.flash ? 'flash 1s infinite' : 'none');

        // Add tooltip
        const tooltip = warningLayer.append('g')
            .attr('class', 'warning-tooltip')
            .attr('transform', `translate(${this.width-200},50)`);

        tooltip.append('text')
            .text(visual.tooltip.title)
            .attr('font-weight', 'bold');

        tooltip.append('text')
            .text(visual.tooltip.details)
            .attr('y', 20)
            .style('font-size', '12px');
    }

    private renderCriticalPoint(point: CriticalPoint) {
        const glyphMap = {
            attractor: '◉',
            source: '◎',
            saddle: '⊗'
        };
        
        return {
            glyph: glyphMap[point.type],
            size: this.config.baseSize * point.strength,
            color: this.getStabilityColor(point.field_state),
            pulseRate: point.type === 'source' ? 
                      this.config.basePulseRate * point.strength : 0
        };
    }

    private generateWarningVisual(warning: CollapseWarning) {
        const intensity = warning.severity;
        const color = this.getRecoveryColor(warning.recovery_chance);
        
        return {
            border: {
                style: 'dashed',
                width: 2 + (3 * intensity),
                color: color
            },
            warning: {
                icon: '⚠️',
                size: 16 + (8 * intensity),
                flash: intensity > 0.7
            },
            tooltip: {
                title: 'Pattern Collapse Warning',
                details: this.getWarningDetails(warning)
            }
        };
    }

    private getFieldColor(divergence: number): string {
        const colorScale = scaleSequential(interpolateViridis)
            .domain([-1, 1]);
        return colorScale(divergence);
    }

    private getStabilityColor(state: VectorFieldState): string {
        if (state.divergence > this.config.collapseThreshold) return '#ff4444';
        if (state.divergence < -this.config.collapseThreshold) return '#44ff44';
        return '#ffff44';
    }

    private getRecoveryColor(chance: number): string {
        const colorScale = scaleLinear<string>()
            .domain([0, 0.5, 1])
            .range(['#ff4444', '#ffff44', '#44ff44']);
        return colorScale(chance);
    }

    private generateArrowField(field: FieldIndicator) {
        // Implementation details for arrow field generation
        // This would create a grid of arrows based on field properties
        return [];
    }

    private generateStreamlines(field: FieldIndicator) {
        // Implementation details for streamline generation
        // This would create flow lines based on field properties
        return [];
    }

    private generateArrowPath(d: any) {
        // Implementation details for arrow path generation
        return 'M0,-5L10,0L0,5Z';
    }

    private getWarningDetails(warning: CollapseWarning): string {
        return `Severity: ${(warning.severity * 100).toFixed(1)}%\nRecovery: ${(warning.recovery_chance * 100).toFixed(1)}%`;
    }
}

interface VisualizerConfig {
    baseSize: number;
    baseThickness: number;
    basePulseRate: number;
    collapseThreshold: number;
}

const defaultConfig: VisualizerConfig = {
    baseSize: 20,
    baseThickness: 2,
    basePulseRate: 2,
    collapseThreshold: 0.7
};
