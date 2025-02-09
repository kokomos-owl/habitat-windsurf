"""Mock data generator for topology visualization testing."""

import asyncio
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from src.core.metrics.flow_metrics import MetricFlowManager, VectorFieldState
from src.core.processor import ClimateRiskProcessor, RiskMetric, ProcessingResult
from src.visualization.server.visualization_server import VisualizationServer

class MockDataGenerator:
    """Generates mock data for visualization testing."""
    
    def __init__(self, server: VisualizationServer):
        self.server = server
        self.processor = ClimateRiskProcessor()
        self.flow_manager = self.server.flow_manager
        
        # Load test data
        self.risk_data = Path(__file__).parent.parent.parent.parent / 'data' / 'climate_risk' / 'climate_risk_marthas_vineyard.txt'
        
    async def generate_mock_stream(self):
        """Generate a mock data stream from synthetic metrics."""
        # Create synthetic base metrics
        base_metrics = [
            RiskMetric(
                value=55.0,
                unit='percent',
                timeframe='2020-2050',
                risk_type='precipitation',
                confidence=0.85,
                source_text='Precipitation increase by 55%',
                semantic_weight=1.0
            ),
            RiskMetric(
                value=13.0,
                unit='percent',
                timeframe='2020-2050',
                risk_type='drought',
                confidence=0.9,
                source_text='Drought likelihood increases to 13%',
                semantic_weight=1.0
            ),
            RiskMetric(
                value=94.0,
                unit='days',
                timeframe='2020-2050',
                risk_type='wildfire',
                confidence=0.8,
                source_text='Fire danger days increase to 94',
                semantic_weight=1.0
            )
        ]
        
        # Generate evolving patterns
        patterns = [
            self._create_evolving_pattern("precipitation", base_metrics[0]),
            self._create_evolving_pattern("drought", base_metrics[1]),
            self._create_evolving_pattern("wildfire", base_metrics[2])
        ]
        
        # Stream patterns with evolution
        while True:
            for pattern in patterns:
                # Evolve pattern
                next_state = next(pattern)
                
                # Update flow manager
                if next_state['flow_id'] not in self.flow_manager.active_flows:
                    self.flow_manager.create_flow(
                        next_state['flow_id'],
                        next_state['metrics']['risk_type']
                    )
                
                flow = self.flow_manager.active_flows[next_state['flow_id']]
                for key, value in next_state['metrics'].items():
                    setattr(flow, key, value)
                flow.history.append(next_state['metrics'])
                
                # Get visualization state
                viz_state = self.server.connector.get_visualization_state()
                
                # Send update
                await self.server.send_update(viz_state)
                
                # Simulate processing time
                await asyncio.sleep(1)
    
    def _create_evolving_pattern(self, pattern_type: str, base_metric: RiskMetric):
        """Create a generator for evolving patterns."""
        flow_id = f"{pattern_type}_{datetime.now().timestamp()}"
        
        # Pattern evolution parameters
        stability = 1.0
        coherence = 1.0
        energy = 0.5
        
        while True:
            # Add some randomness to evolution
            stability += np.random.normal(0, 0.1)
            stability = max(0, min(1, stability))
            
            coherence += np.random.normal(0, 0.05)
            coherence = max(0, min(1, coherence))
            
            energy += np.random.normal(0, 0.15)
            energy = max(0, min(1, energy))
            
            # Create metric state
            metric_state = {
                'flow_id': flow_id,
                'metrics': {
                    'stability': stability,
                    'coherence': coherence,
                    'energy_state': energy,
                    'confidence': base_metric.confidence,
                    'temporal_stability': stability * coherence,
                    'risk_type': pattern_type
                }
            }
            
            yield metric_state
            
async def run_mock_visualization():
    """Run mock visualization server with generated data."""
    # Start visualization server
    server = VisualizationServer()
    server_task = asyncio.create_task(server.start_server())
    
    # Start mock data generator
    generator = MockDataGenerator(server)
    generator_task = asyncio.create_task(generator.generate_mock_stream())
    
    try:
        # Run both tasks
        await asyncio.gather(server_task, generator_task)
    except asyncio.CancelledError:
        pass
    
if __name__ == "__main__":
    asyncio.run(run_mock_visualization())
