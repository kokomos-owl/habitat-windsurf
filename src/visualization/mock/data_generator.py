"""Mock data generator for topology visualization testing."""

import asyncio
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.core.metrics.flow_metrics import MetricFlowManager, VectorFieldState
from src.core.processor import ClimateRiskProcessor, RiskMetric, ProcessingResult
from src.visualization.server.visualization_server import VisualizationServer

class MockDataGenerator:
    """Generates mock data for visualization testing."""
    
    def __init__(self, server: VisualizationServer):
        self.server = server
        self.processor = ClimateRiskProcessor()
        self.flow_manager = self.server.flow_manager
        self.logger = logging.getLogger(__name__)
        
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
        
        self.logger.info("Initializing mock data generator")
        
        # Create persistent patterns with fixed IDs
        pattern_flows = {}
        self.logger.info("Creating pattern flows...")
        
        for i, (pattern_type, metric) in enumerate([('precipitation', base_metrics[0]),
                                                   ('drought', base_metrics[1]),
                                                   ('wildfire', base_metrics[2])]):
            pattern_flows[pattern_type] = self._create_evolving_pattern(pattern_type, metric)
            self.logger.info(f"Created {pattern_type} pattern generator")
        
        # Initialize flows once
        self.logger.info("Initializing flows...")
        for pattern_type, pattern in pattern_flows.items():
            next_state = next(pattern)
            flow_id = next_state['flow_id']
            
            if flow_id not in self.flow_manager.active_flows:
                self.flow_manager.create_flow(
                    flow_id=flow_id,
                    source_pattern=pattern_type
                )
                self.logger.info(f"Created flow {flow_id} for pattern {pattern_type}")
        
        # Stream pattern evolution
        self.logger.info("Starting pattern evolution stream...")
        update_count = 0
        last_update_time = 0
        
        while True:
            update_count += 1
            current_time = asyncio.get_event_loop().time()
            
            # Only update every 2 seconds
            if current_time - last_update_time < 2:
                await asyncio.sleep(0.1)
                continue
                
            self.logger.info(f"\n--- Update Cycle {update_count} ---")
            last_update_time = current_time
            
            # Update all patterns
            pattern_updates = {}
            for pattern_type, pattern in pattern_flows.items():
                next_state = next(pattern)
                flow_id = next_state['flow_id']
                pattern_updates[pattern_type] = next_state
            
            # Apply all updates at once
            for pattern_type, next_state in pattern_updates.items():
                flow_id = next_state['flow_id']
                
                if flow_id not in self.flow_manager.active_flows:
                    self.logger.error(f"Flow {flow_id} not found in active flows!")
                    continue
                    
                flow = self.flow_manager.active_flows[flow_id]
                
                # Update flow metrics
                self.logger.info(f"Updating {pattern_type} flow metrics:")
                for key, value in next_state['metrics'].items():
                    old_value = getattr(flow, key, None)
                    setattr(flow, key, value)
                    self.logger.info(f"  {key}: {old_value if old_value is None else str(old_value)} -> {value}")
            
            # Get and send single visualization state update
            viz_state = self.server.connector.get_visualization_state()
            self.logger.info(f"Visualization state: {len(viz_state['nodes'])} nodes, {len(viz_state['links'])} links")
            await self.server.send_update(viz_state)
            self.logger.info("Sent update for all patterns")
            
            # Pause between updates
            await asyncio.sleep(2)
    
    def _create_evolving_pattern(self, pattern_type: str, base_metric: RiskMetric):
        """Create a generator for evolving patterns."""
        flow_id = f"{pattern_type}_flow"
        
        # Pattern evolution parameters with different ranges per type
        if pattern_type == 'precipitation':
            base_stability = 0.85
            base_coherence = 0.75
            base_energy = 0.65
            noise_scale = 0.02
        elif pattern_type == 'drought':
            base_stability = 0.65
            base_coherence = 0.85
            base_energy = 0.45
            noise_scale = 0.03
        else:  # wildfire
            base_stability = 0.75
            base_coherence = 0.65
            base_energy = 0.75
            noise_scale = 0.04
            
        # Current state
        current = {
            'stability': base_stability,
            'coherence': base_coherence,
            'energy_state': base_energy
        }
        
        while True:
            # Slowly drift metrics with controlled noise
            new_state = {}
            for key, base_value in current.items():
                # Add tiny drift toward base value
                drift = (base_value - current[key]) * 0.1
                # Add small random noise
                noise = np.random.normal(0, noise_scale)
                # Update with drift and noise
                new_value = current[key] + drift + noise
                # Clip to valid range
                new_value = np.clip(new_value, 0.4, 0.9)
                new_state[key] = new_value
                current[key] = new_value
            
            # Create metric state
            metric_state = {
                'flow_id': flow_id,
                'metrics': {
                    'stability': new_state['stability'],
                    'coherence': new_state['coherence'],
                    'energy_state': new_state['energy_state'],
                    'confidence': base_metric.confidence,
                    'temporal_stability': new_state['stability'] * new_state['coherence']
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
