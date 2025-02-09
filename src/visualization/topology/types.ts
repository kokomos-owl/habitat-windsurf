export interface VectorFieldState {
    magnitude: number;
    direction: number;
    divergence: number;
    curl: number;
    critical_points: CriticalPoint[];
}

export interface CriticalPoint {
    type: 'attractor' | 'source' | 'saddle';
    position: [number, number];
    strength: number;
    field_state: VectorFieldState;
}

export interface FieldIndicator {
    magnitude: number;
    direction: number;
    divergence: number;
    curl: number;
}

export interface PatternState {
    stability: number;
    coherence: number;
    emergence_rate: number;
    energy_state: number;
}

export interface CollapseWarning {
    severity: number;
    type: 'topology_based';
    recovery_chance: number;
    field_state: VectorFieldState;
}
