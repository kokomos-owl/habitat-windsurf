"""Pattern evolution visualization server."""

import asyncio
import websockets
import json
from typing import Dict, Set, Optional
from pathlib import Path
import logging
import numpy as np

from src.core.metrics.flow_metrics import MetricFlowManager
from src.core.processor import ClimateRiskProcessor, RiskMetric
from src.visualization.topology.TopologyConnector import TopologyConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatternEvolutionServer:
    """Server for step-wise pattern evolution visualization."""
    
    def __init__(self):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.flow_manager = MetricFlowManager()
        self.processor = ClimateRiskProcessor()
        self.connector = TopologyConnector(self.flow_manager)
        
        # Evolution state
        self.current_step = 0
        self.max_steps = 100
        self.pattern_view = 0  # 0=initial, 1=t20, 2=t50, 3=t100
        self.base_patterns = None
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client."""
        self.clients.add(websocket)
        logger.info(f"Client connected from {websocket.remote_address}. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client."""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_update(self, message: Dict):
        """Send update to all connected clients."""
        if not self.clients:
            logger.warning("No clients connected, skipping update")
            return
        
        wrapped_message = {'type': 'state_update', 'state': message}
        message_str = json.dumps(wrapped_message)
        
        try:
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients]
            )
            logger.info("Update sent successfully to all clients")
        except Exception as e:
            logger.error(f"Error sending update: {str(e)}")
            
    async def send_error(self, message: str):
        """Send error message to all clients."""
        error_msg = {'type': 'error', 'message': message}
        try:
            await asyncio.gather(
                *[client.send(json.dumps(error_msg)) for client in self.clients]
            )
        except Exception as e:
            logger.error(f"Error sending error message: {str(e)}")
            
    def load_initial_state(self, file_path: str):
        """Load initial state from climate risk data."""
        try:
            # Reset state
            self.current_step = 0
            self.flow_manager = MetricFlowManager()
            self.connector = TopologyConnector(self.flow_manager)
            
            # Load and process climate risk data
            data_path = Path(__file__).parent.parent.parent.parent / 'data/climate_risk/climate_risk_patterns.json'
            logger.info(f"Loading data from {data_path}")
            
            with open(data_path, 'r') as f:
                risk_data = json.load(f)
                
            # Process patterns and create flows
            self.base_patterns = []
            for pattern_data in risk_data['patterns']:
                # Create RiskMetric object
                pattern = RiskMetric(
                    risk_type=pattern_data['risk_type'],
                    nodes=pattern_data['nodes'],
                    links=pattern_data['links']
                )
                
                # Set initial metrics
                pattern.stability = pattern_data['initial_metrics']['stability']
                pattern.coherence = pattern_data['initial_metrics']['coherence']
                pattern.energy_state = pattern_data['initial_metrics']['energy_state']
                
                self.base_patterns.append(pattern)
                
                # Create flow for this pattern
                flow = self.flow_manager.create_flow(
                    flow_id=f"{pattern.risk_type}_flow",
                    source_pattern=pattern.risk_type
                )
                
                # Set initial flow metrics
                flow.stability = pattern.stability
                flow.coherence = pattern.coherence
                flow.energy_state = pattern.energy_state
                flow.temporal_stability = pattern.stability * pattern.coherence
                flow.confidence = 0.9  # Start with high confidence
                
            logger.info(f"Loaded {len(self.base_patterns)} patterns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading initial state: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _evolve_patterns(self):
        """Internal method to evolve pattern states."""
        flows = self.flow_manager.active_flows
        time_factor = self.current_step / self.max_steps
        
        for pattern in self.base_patterns:
            flow_id = f"{pattern.risk_type}_flow"
            if flow_id not in flows:
                continue
                
            flow = flows[flow_id]
            
            # Pattern-specific evolution characteristics
            if 'precipitation' in pattern.risk_type:
                stability_base = 0.85
                coherence_base = 0.75
                energy_base = 0.65
                noise_scale = 0.02
            elif 'drought' in pattern.risk_type:
                stability_base = 0.65
                coherence_base = 0.85
                energy_base = 0.45
                noise_scale = 0.03
            else:  # wildfire
                stability_base = 0.75
                coherence_base = 0.65
                energy_base = 0.75
                noise_scale = 0.04
            
            # Calculate drifts toward base values
            stability_drift = (stability_base - flow.stability) * 0.1
            coherence_drift = (coherence_base - flow.coherence) * 0.1
            energy_drift = (energy_base - flow.energy_state) * 0.1
            
            # Add controlled noise scaled by time
            flow.stability = np.clip(
                flow.stability + stability_drift + np.random.normal(0, noise_scale) * time_factor,
                0.4, 0.9
            )
            flow.coherence = np.clip(
                flow.coherence + coherence_drift + np.random.normal(0, noise_scale) * time_factor,
                0.4, 0.9
            )
            flow.energy_state = np.clip(
                flow.energy_state + energy_drift + np.random.normal(0, noise_scale) * time_factor,
                0.3, 0.8
            )
            
            # Update derived metrics
            flow.temporal_stability = flow.stability * flow.coherence
            flow.confidence = max(0.6, 1.0 - (time_factor * 0.3))
    
    def evolve_step(self) -> Optional[Dict]:
        """Evolve patterns one step forward."""
        if not self.base_patterns:
            logger.error("No initial state loaded")
            return None
            
        if self.current_step >= self.max_steps:
            logger.warning("Maximum evolution steps reached")
            return None
            
        try:
            # Update step counter
            self.current_step += 1
            logger.info(f"Evolving step {self.current_step}")
            
            # Evolve patterns
            self._evolve_patterns()
            
            # Get updated visualization state
            state = self.connector.get_visualization_state()
            logger.info(f"Generated state with {len(state['nodes'])} nodes and {len(state['links'])} links")
            return state
            
        except Exception as e:
            logger.error(f"Error in evolution step: {str(e)}")
            return None
            
    def get_pattern_view(self) -> Optional[Dict]:
        """Get visualization state for current pattern view."""
        if not self.base_patterns:
            logger.error("Cannot get pattern view: No initial state loaded")
            return None
            
        try:
            # Map pattern view to time steps
            step_map = {
                0: 0,      # Initial
                1: 20,     # t=20
                2: 50,     # t=50
                3: 100     # t=100
            }
            
            # Reset flow manager to initial state
            self.flow_manager = MetricFlowManager()
            self.connector = TopologyConnector(self.flow_manager)
            
            # Recreate initial flows
            for pattern in self.base_patterns:
                self.flow_manager.create_flow(
                    flow_id=f"{pattern.risk_type}_flow",
                    source_pattern=pattern.risk_type
                )
            
            # Evolve to target step
            target_step = step_map[self.pattern_view]
            logger.info(f"Evolving to step {target_step} for pattern view {self.pattern_view}")
            
            # Run evolution steps
            for _ in range(target_step):
                self.current_step += 1
                self._evolve_patterns()
            
            # Get state at this step
            state = self.connector.get_visualization_state()
            logger.info(f"Generated view with {len(state['nodes'])} nodes and {len(state['links'])} links")
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating pattern view: {str(e)}")
            return None
            
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle client connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data['type']}")
                    
                    if data['type'] == 'load_initial':
                        if self.load_initial_state(data['file']):
                            state = self.connector.get_visualization_state()
                            await self.send_update(state)
                            logger.info("Initial state loaded and sent")
                        else:
                            await self.send_error("Failed to load initial state")
                            
                    elif data['type'] == 'evolve_step':
                        if not self.base_patterns:
                            await self.send_error("Please load initial state first")
                        else:
                            state = self.evolve_step()
                            if state:
                                await self.send_update(state)
                            else:
                                await self.send_error("Evolution step failed")
                            
                    elif data['type'] == 'reset':
                        if not self.base_patterns:
                            await self.send_error("Please load initial state first")
                        else:
                            # Reload initial state
                            self.current_step = 0
                            self.flow_manager = MetricFlowManager()
                            self.connector = TopologyConnector(self.flow_manager)
                            
                            # Recreate initial flows
                            for pattern in self.base_patterns:
                                self.flow_manager.create_flow(
                                    flow_id=f"{pattern.risk_type}_flow",
                                    source_pattern=pattern.risk_type
                                )
                                
                            state = self.connector.get_visualization_state()
                            await self.send_update(state)
                            logger.info("Reset to initial state")
                        
                    elif data['type'] == 'set_pattern':
                        if not self.base_patterns:
                            await self.send_error("Please load initial state first")
                        else:
                            self.pattern_view = data['pattern']
                            state = self.get_pattern_view()
                            if state:
                                await self.send_update(state)
                            else:
                                await self.send_error("Failed to get pattern view")
                            
                except json.JSONDecodeError:
                    await self.send_error("Invalid message format")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await self.send_error(f"Error processing message: {str(e)}")
                    
        finally:
            await self.unregister(websocket)
            
async def main():
    server = PatternEvolutionServer()
    async with websockets.serve(server.handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
