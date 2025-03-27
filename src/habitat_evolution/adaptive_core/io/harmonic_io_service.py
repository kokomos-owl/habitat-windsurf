"""
Harmonic I/O Service

This service harmonizes I/O operations with system rhythms, ensuring that
database operations don't disrupt the natural evolution of eigenspaces and
pattern detection.

The service implements a priority queue for operations, scheduling them
according to harmonic timing that preserves the natural flow of pattern
evolution in the system.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import numpy as np
from collections import defaultdict
import threading
import time
import queue
import logging
from enum import Enum

# Import for type hints only
try:
    from src.habitat_evolution.core.services.event_bus import LocalEventBus
except ImportError:
    # Define a placeholder for type hints if the actual class is not available
    class LocalEventBus: pass


class OperationType(Enum):
    """Types of I/O operations that can be harmonized."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"
    PROCESS = "process"


class HarmonicIOService:
    """
    Service that harmonizes I/O operations with system rhythms.
    
    This service ensures that database operations don't disrupt
    the natural evolution of eigenspaces and pattern detection.
    It uses harmonic timing to schedule operations in a way that
    preserves the continuity of pattern evolution.
    """
    
    def __init__(self, 
                 base_frequency_or_event_bus: Any = 0.1, 
                 harmonics: int = 3,
                 max_queue_size: int = 1000,
                 adaptive_timing: bool = True,
                 event_bus: Optional['LocalEventBus'] = None):
        """
        Initialize the harmonic I/O service.
        
        Args:
            base_frequency_or_event_bus: Either the base frequency for harmonic cycles (Hz)
                or an event bus instance for backward compatibility
            harmonics: Number of harmonic overtones to consider
            max_queue_size: Maximum size of operation queues
            adaptive_timing: Whether to adapt timing based on system state
            event_bus: Event bus for publishing and subscribing to events
        """
        # Handle backward compatibility
        # If first parameter looks like an event bus and no explicit event_bus is provided
        if event_bus is None and hasattr(base_frequency_or_event_bus, 'subscribe'):
            self.event_bus = base_frequency_or_event_bus
            self.base_frequency = 0.1  # Use default value
        else:
            self.event_bus = event_bus
            self.base_frequency = base_frequency_or_event_bus
            
        self.harmonics = harmonics
        self.max_queue_size = max_queue_size
        self.adaptive_timing = adaptive_timing
        
        # System state
        self.start_time = datetime.now()
        self.eigenspace_stability = 0.5  # Default stability
        self.resonance_level = 0.5  # Default resonance
        self.pattern_coherence = 0.5  # Default pattern coherence
        self.system_load = 0.0  # Default system load
        
        # Metrics and monitoring
        self.operation_metrics = defaultdict(list)
        self.pattern_buffers = {}
        self.tonic_patterns = {
            OperationType.READ.value: [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5],
            OperationType.WRITE.value: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            OperationType.UPDATE.value: [0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4],
            OperationType.DELETE.value: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            OperationType.QUERY.value: [0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5, 0.6, 0.7, 0.6]
        }
        self.pattern_indices = defaultdict(int)
        
        # Operation queues
        self.operation_queues = {
            op_type.value: queue.PriorityQueue(maxsize=max_queue_size)
            for op_type in OperationType
        }
        
        # Counter for breaking ties in priority queue
        self.operation_counter = 0
        
        # Processing threads
        self.processing_threads = {}
        self.running = False
        
        # Data transformation tracking
        self.transformation_log = []
        self.actant_transformations = defaultdict(list)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Event bus integration
        self._event_handlers = {}
        self._event_publishing_schedule = {}
        
    def start(self):
        """Start the harmonic I/O processing."""
        if self.running:
            self.logger.warning("HarmonicIOService is already running")
            return
            
        self.running = True
        
        # Start a processing thread for each operation type
        for op_type in OperationType:
            thread = threading.Thread(
                target=self._process_queue,
                args=(op_type.value,),
                name=f"harmonic-io-{op_type.value}"
            )
            thread.daemon = True
            thread.start()
            self.processing_threads[op_type.value] = thread
        
        # Start event bus integration if event_bus is provided
        if self.event_bus:
            self._setup_event_bus_integration()
            
        self.logger.info("HarmonicIOService started")
    
    def stop(self):
        """Stop the harmonic I/O processing."""
        if not self.running:
            self.logger.warning("HarmonicIOService is not running")
            return
            
        self.running = False
        
        # Wait for threads to terminate
        for op_type, thread in self.processing_threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    self.logger.warning(f"Thread for {op_type} did not terminate gracefully")
        
        # Clean up event bus handlers if event_bus is provided
        if self.event_bus and hasattr(self.event_bus, 'unsubscribe'):
            for event_type, handler in self._event_handlers.items():
                try:
                    self.event_bus.unsubscribe(event_type, handler)
                except Exception as e:
                    self.logger.warning(f"Error unsubscribing from {event_type}: {e}")
                    
        self.processing_threads = {}
        self.logger.info("HarmonicIOService stopped")
    
    def get_cycle_position(self, timestamp: Optional[datetime] = None) -> float:
        """
        Get the current position in the harmonic cycle.
        
        Args:
            timestamp: Timestamp to calculate cycle position for (default: now)
            
        Returns:
            Float between 0.0 and 1.0 representing position in cycle
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        elapsed = (timestamp - self.start_time).total_seconds()
        cycle_position = elapsed * self.base_frequency
        
        return cycle_position % 1.0
    
    def get_tonic_value(self, operation_type: str) -> float:
        """
        Get the current tonic value for an operation type.
        
        Args:
            operation_type: Type of operation (read, write, etc.)
            
        Returns:
            Current tonic value (0.0 to 1.0)
        """
        if operation_type not in self.tonic_patterns:
            return 0.5  # Default value
            
        pattern = self.tonic_patterns[operation_type]
        index = self.pattern_indices[operation_type] % len(pattern)
        self.pattern_indices[operation_type] += 1
        
        return pattern[index]
    
    def _setup_event_bus_integration(self):
        """
        Set up integration with the event bus.
        
        This method establishes the connection between the harmonic I/O service
        and the event bus, setting up harmonically timed event publishing and
        subscription based on the base_frequency parameter.
        """
        if not self.event_bus:
            return
            
        # Subscribe to system state events that might affect harmonic timing
        self.subscribe_to_event("field.state.updated", self._on_field_state_updated)
        self.subscribe_to_event("system.load.updated", self._on_system_load_updated)
        
        # Subscribe to meta-pattern detection events for feedback loop
        self.subscribe_to_event("pattern.meta.detected", self._on_meta_pattern_detected)
        
        # Subscribe to field gradient updates for topology metrics extraction
        self.subscribe_to_event("field.gradient.update", self._on_field_gradient_updated)
        
        # Initialize topology metrics tracking
        self._topology_metrics = {
            "resonance_centers": [],
            "interference_patterns": [],
            "field_density_centers": [],
            "flow_vectors": [],
            "effective_dimensionality": 0,
            "principal_dimensions": [],
            "stability_trend": [],
            "coherence_trend": [],
            "meta_pattern_influence": {}
        }
        
        # Start a thread for harmonically timed event publishing
        self._event_publishing_thread = threading.Thread(
            target=self._process_event_publishing,
            name="harmonic-event-publisher"
        )
        self._event_publishing_thread.daemon = True
        self._event_publishing_thread.start()
        
        self.logger.info("Event bus integration initialized")
    
    def subscribe_to_event(self, event_type: str, handler: Callable, harmonic_timing: bool = False):
        """
        Subscribe to an event with optional harmonic timing.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler function to call when event is received
            harmonic_timing: Whether to apply harmonic timing to event handling
        """
        if not self.event_bus:
            self.logger.warning("Cannot subscribe to events: no event bus provided")
            return
            
        if harmonic_timing:
            # Wrap the handler in a function that applies harmonic timing
            original_handler = handler
            def harmonic_handler(event):
                # Schedule the handler to be called at a harmonically optimal time
                priority = self.calculate_harmonic_priority("process", {"event_type": event_type})
                self.schedule_operation(
                    operation_type=OperationType.PROCESS.value,
                    repository=None,
                    method_name="_process_event",
                    args=(original_handler, event),
                    kwargs={},
                    data_context={"event_type": event_type}
                )
            handler = harmonic_handler
            
        # Store the handler for cleanup during stop()
        self._event_handlers[event_type] = handler
        
        # Subscribe to the event
        self.event_bus.subscribe(event_type, handler)
    
    def publish_event(self, event_type: str, data: Dict[str, Any], 
                     immediate: bool = False, harmonic_weight: float = 0.5):
        """
        Publish an event with optional harmonic timing.
        
        Args:
            event_type: Type of event to publish
            data: Event data
            immediate: Whether to publish immediately or schedule according to harmonic timing
            harmonic_weight: Weight to give to harmonic timing (0.0-1.0)
        """
        if not self.event_bus:
            self.logger.warning("Cannot publish event: no event bus provided")
            return
            
        if immediate:
            # Publish immediately
            if hasattr(self.event_bus, 'publish'):
                # Check if there's an Event.create method available
                if hasattr(self.event_bus, 'Event') and hasattr(self.event_bus.Event, 'create'):
                    event = self.event_bus.Event.create(type=event_type, data=data, source="harmonic_io_service")
                    self.event_bus.publish(event)
                else:
                    # Try to import Event class
                    try:
                        from src.habitat_evolution.core.services.event_bus import Event
                        event = Event.create(type=event_type, data=data, source="harmonic_io_service")
                        self.event_bus.publish(event)
                    except ImportError:
                        # Fallback to dictionary if Event class is not available
                        self.logger.warning("Event class not found, falling back to dictionary")
                        self.event_bus.publish({
                            "type": event_type,
                            "data": data,
                            "timestamp": datetime.now().isoformat()
                        })
        else:
            # Schedule for harmonic publishing
            self._event_publishing_schedule[event_type] = {
                "data": data,
                "harmonic_weight": harmonic_weight,
                "scheduled_at": datetime.now()
            }
    
    def _process_event_publishing(self):
        """
        Process event publishing according to harmonic timing.
        
        This method runs in a separate thread and publishes events
        at harmonically optimal times based on the base_frequency.
        """
        while self.running:
            now = datetime.now()
            cycle_pos = self.get_cycle_position(now)
            
            # Check for events to publish
            for event_type, schedule in list(self._event_publishing_schedule.items()):
                # Calculate harmonic priority
                priority = self.calculate_harmonic_priority(
                    OperationType.PROCESS.value,
                    {"event_type": event_type}
                )
                
                # Determine if it's time to publish
                time_factor = (now - schedule["scheduled_at"]).total_seconds() * self.base_frequency
                harmonic_factor = 1.0 - priority  # Invert priority (lower priority = higher value)
                
                # Combine time factor and harmonic factor
                weight = schedule["harmonic_weight"]
                publish_score = (time_factor * (1.0 - weight)) + (harmonic_factor * weight)
                
                if publish_score > 0.7:  # Threshold for publishing
                    # Publish the event
                    if hasattr(self.event_bus, 'publish'):
                        # Check if there's an Event.create method available
                        if hasattr(self.event_bus, 'Event') and hasattr(self.event_bus.Event, 'create'):
                            event = self.event_bus.Event.create(type=event_type, data=schedule["data"], source="harmonic_io_service")
                            self.event_bus.publish(event)
                        else:
                            # Try to import Event class
                            try:
                                from src.habitat_evolution.core.services.event_bus import Event
                                event = Event.create(type=event_type, data=schedule["data"], source="harmonic_io_service")
                                self.event_bus.publish(event)
                            except ImportError:
                                # Fallback to dictionary if Event class is not available
                                self.logger.warning("Event class not found, falling back to dictionary")
                                self.event_bus.publish({
                                    "type": event_type,
                                    "data": schedule["data"],
                                    "timestamp": now.isoformat()
                                })
                    
                    # Remove from schedule
                    del self._event_publishing_schedule[event_type]
            
            # Sleep for a short time
            sleep_time = 0.1 / self.base_frequency  # Adjust sleep time based on base_frequency
            time.sleep(min(0.1, sleep_time))  # Cap at 100ms to ensure responsiveness
    
    def _process_event(self, handler, event):
        """
        Process an event with a handler.
        
        Args:
            handler: Handler function to call
            event: Event to process
        """
        try:
            handler(event)
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
    
    def _on_field_state_updated(self, event):
        """
        Handle field state update events.
        
        Updates eigenspace stability and pattern coherence based on field state.
        
        Args:
            event: Field state update event
        """
        try:
            data = event.data if hasattr(event, 'data') else event.get('data', {})
            
            # Update eigenspace stability if available
            if 'stability' in data:
                self.eigenspace_stability = data['stability']
                
            # Update pattern coherence if available
            if 'coherence' in data:
                self.pattern_coherence = data['coherence']
                
            # Update resonance level if available
            if 'resonance' in data:
                self.resonance_level = data['resonance']
                
            self.logger.debug(f"Updated harmonic parameters from field state: "
                           f"stability={self.eigenspace_stability}, "
                           f"coherence={self.pattern_coherence}, "
                           f"resonance={self.resonance_level}")
        except Exception as e:
            self.logger.error(f"Error processing field state update: {e}")
    
    def _on_system_load_updated(self, event):
        """
        Handle system load update events.
        
        Updates system load parameter based on event data.
        
        Args:
            event: System load update event
        """
        try:
            data = event.data if hasattr(event, 'data') else event.get('data', {})
            
            # Update system load if available
            if 'load' in data:
                self.system_load = data['load']
                
            self.logger.debug(f"Updated system load: {self.system_load}")
        except Exception as e:
            self.logger.error(f"Error processing system load update: {e}")
            
    def _on_meta_pattern_detected(self, event):
        """
        Handle meta-pattern detection events for feedback loop.
        
        This method implements a feedback loop that adjusts harmonic parameters
        based on detected meta-patterns, creating a self-regulating system.
        
        Args:
            event: Meta-pattern detection event
        """
        try:
            # Extract meta-pattern data
            data = event.data if hasattr(event, 'data') else event.get('data', {})
            
            # Extract meta-pattern properties
            pattern_id = data.get('id', 'unknown')
            pattern_type = data.get('type', '')
            confidence = data.get('confidence', 0.5)
            frequency = data.get('frequency', 0)
            examples = data.get('examples', [])
            
            # Log meta-pattern detection
            self.logger.info(f"Meta-pattern feedback: {pattern_type} (id: {pattern_id})")
            self.logger.info(f"  Confidence: {confidence:.2f}, Frequency: {frequency}")
            self.logger.info(f"  Examples: {len(examples)} instances")
            
            # Track meta-pattern influence on topology
            self._topology_metrics["meta_pattern_influence"][pattern_id] = {
                "type": pattern_type,
                "confidence": confidence,
                "frequency": frequency,
                "detected_at": datetime.now().isoformat(),
                "influence_score": confidence * (min(frequency, 10) / 10)
            }
            
            # Adjust harmonic parameters based on meta-pattern
            self._adjust_parameters_from_meta_pattern(pattern_type, confidence, frequency)
            
            # Publish topology metrics update
            self._publish_topology_metrics_update()
            
        except Exception as e:
            self.logger.error(f"Error handling meta-pattern detection: {e}")
            
    def _on_field_gradient_updated(self, event):
        """
        Handle field gradient update events for topology metrics extraction.
        
        This method extracts topology-related metrics from field gradient updates,
        which can be used to derive insights about the system's structure.
        
        Args:
            event: Field gradient update event
        """
        try:
            # Extract gradient data
            data = event.data if hasattr(event, 'data') else event.get('data', {})
            
            # Extract topology information if available
            topology = data.get('topology', {})
            if not topology and 'gradient' in data:
                # Try alternate location for topology data
                topology = data['gradient'].get('topology', {})
                
            if topology:
                self.logger.info("Extracting topology metrics from field gradient")
                
                # Extract resonance centers
                if 'resonance_centers' in topology:
                    centers = topology['resonance_centers']
                    if isinstance(centers, dict):
                        self._topology_metrics['resonance_centers'] = list(centers.values())
                    elif isinstance(centers, list):
                        self._topology_metrics['resonance_centers'] = centers
                    self.logger.info(f"  Resonance centers: {len(self._topology_metrics['resonance_centers'])}")
                
                # Extract interference patterns
                if 'interference_patterns' in topology:
                    patterns = topology['interference_patterns']
                    if isinstance(patterns, dict):
                        self._topology_metrics['interference_patterns'] = list(patterns.values())
                    elif isinstance(patterns, list):
                        self._topology_metrics['interference_patterns'] = patterns
                    self.logger.info(f"  Interference patterns: {len(self._topology_metrics['interference_patterns'])}")
                
                # Extract field density centers
                if 'field_density_centers' in topology:
                    centers = topology['field_density_centers']
                    if isinstance(centers, dict):
                        self._topology_metrics['field_density_centers'] = list(centers.values())
                    elif isinstance(centers, list):
                        self._topology_metrics['field_density_centers'] = centers
                    self.logger.info(f"  Field density centers: {len(self._topology_metrics['field_density_centers'])}")
                
                # Extract flow vectors
                if 'flow_vectors' in topology:
                    vectors = topology['flow_vectors']
                    if isinstance(vectors, dict):
                        self._topology_metrics['flow_vectors'] = list(vectors.values())
                    elif isinstance(vectors, list):
                        self._topology_metrics['flow_vectors'] = vectors
                    self.logger.info(f"  Flow vectors: {len(self._topology_metrics['flow_vectors'])}")
                
                # Extract dimensionality information
                if 'effective_dimensionality' in topology:
                    self._topology_metrics['effective_dimensionality'] = topology['effective_dimensionality']
                    self.logger.info(f"  Effective dimensionality: {self._topology_metrics['effective_dimensionality']}")
                
                if 'principal_dimensions' in topology:
                    self._topology_metrics['principal_dimensions'] = topology['principal_dimensions']
                    self.logger.info(f"  Principal dimensions: {len(self._topology_metrics['principal_dimensions'])}")
                
                # Update stability and coherence trends
                self._topology_metrics['stability_trend'].append(self.eigenspace_stability)
                self._topology_metrics['coherence_trend'].append(self.pattern_coherence)
                
                # Keep only the last 10 trend values
                if len(self._topology_metrics['stability_trend']) > 10:
                    self._topology_metrics['stability_trend'] = self._topology_metrics['stability_trend'][-10:]
                if len(self._topology_metrics['coherence_trend']) > 10:
                    self._topology_metrics['coherence_trend'] = self._topology_metrics['coherence_trend'][-10:]
                
                # Calculate topology-derived metrics
                self._calculate_topology_derived_metrics()
        
        except Exception as e:
            self.logger.warning(f"Error extracting topology metrics: {e}")
    
    def calculate_harmonic_priority(self, 
                                   operation_type: str, 
                                   data_context: Dict[str, Any]) -> float:
        """
        Calculate harmonic priority for an operation.
        
        Lower values indicate higher priority.
        
        Args:
            operation_type: Type of operation (read, write, etc.)
            data_context: Context information about the data
            
        Returns:
            Priority value (lower is higher priority)
        """
        cycle_pos = self.get_cycle_position()
        tonic = self.get_tonic_value(operation_type)
        
        # Extract stability from data if available
        stability = data_context.get("stability", self.eigenspace_stability)
        coherence = data_context.get("coherence", self.pattern_coherence)
        
        # Base priority starts at middle value
        priority = 0.5
        
        # Adjust priority based on operation type
        if operation_type == OperationType.WRITE.value:
            # Writes are prioritized at stable points in the cycle
            # to avoid disrupting evolving patterns
            harmonic_factor = 0.5 + 0.5 * np.sin(2 * np.pi * cycle_pos)
            stability_factor = stability * tonic
            priority = 1.0 - (stability_factor * harmonic_factor)
            
        elif operation_type == OperationType.READ.value:
            # Reads are prioritized at transition points
            # to capture evolving patterns
            harmonic_factor = 0.5 + 0.5 * np.cos(2 * np.pi * cycle_pos)
            transition_factor = (1.0 - stability) * tonic
            priority = 1.0 - (transition_factor * harmonic_factor)
            
        elif operation_type == OperationType.UPDATE.value:
            # Updates are prioritized based on coherence
            # to ensure consistent pattern evolution
            harmonic_factor = 0.5 + 0.5 * np.sin(2 * np.pi * cycle_pos + np.pi/4)
            coherence_factor = coherence * tonic
            priority = 1.0 - (coherence_factor * harmonic_factor)
            
        elif operation_type == OperationType.DELETE.value:
            # Deletes are lowest priority during pattern formation
            # to avoid disrupting emerging patterns
            harmonic_factor = 0.5 + 0.5 * np.sin(2 * np.pi * cycle_pos + np.pi/2)
            priority = 0.8 - (0.3 * harmonic_factor)
            
        elif operation_type == OperationType.QUERY.value:
            # Queries are prioritized based on system load
            # to ensure responsive pattern detection
            harmonic_factor = 0.5 + 0.5 * np.cos(2 * np.pi * cycle_pos + np.pi/6)
            load_factor = (1.0 - self.system_load) * tonic
            priority = 1.0 - (load_factor * harmonic_factor)
            
        # Ensure priority is within bounds
        priority = max(0.0, min(1.0, priority))
            
        return priority
    
    def schedule_operation(self, 
                          operation_type: str, 
                          repository: Any, 
                          method_name: str, 
                          args: Tuple, 
                          kwargs: Dict[str, Any],
                          data_context: Dict[str, Any] = None) -> float:
        """
        Schedule an I/O operation with harmonic timing.
        
        Args:
            operation_type: Type of operation (read, write, etc.)
            repository: Repository instance to operate on
            method_name: Name of the direct method to call
            args: Arguments to pass to the method
            kwargs: Keyword arguments to pass to the method
            data_context: Context information about the data
            
        Returns:
            Priority assigned to the operation (lower is higher priority)
        """
        if data_context is None:
            data_context = {}
            
        # Validate operation type
        try:
            op_type = OperationType(operation_type)
        except ValueError:
            self.logger.warning(f"Invalid operation type: {operation_type}, using WRITE")
            op_type = OperationType.WRITE
            operation_type = op_type.value
            
        # Calculate priority
        priority = self.calculate_harmonic_priority(operation_type, data_context)
        
        # Create operation object
        operation = {
            "operation_type": operation_type,
            "repository": repository,
            "method_name": method_name,
            "args": args,
            "kwargs": kwargs,
            "data_context": data_context,
            "timestamp": datetime.now(),
            "priority": priority
        }
        
        # Add to appropriate queue
        try:
            # Increment counter for unique ordering
            self.operation_counter += 1
            
            # Use a tuple of (priority, counter, operation) to ensure unique ordering
            # The counter ensures that operations with the same priority are processed in FIFO order
            self.operation_queues[operation_type].put((priority, self.operation_counter, operation), block=False)
            
            # Record metrics
            self.operation_metrics[operation_type].append({
                "priority": priority,
                "timestamp": datetime.now(),
                "data_type": data_context.get("data_type", "unknown"),
                "status": "queued"
            })
            
        except queue.Full:
            self.logger.warning(f"Operation queue for {operation_type} is full, operation rejected")
            
            # Record rejection
            self.operation_metrics[operation_type].append({
                "priority": priority,
                "timestamp": datetime.now(),
                "data_type": data_context.get("data_type", "unknown"),
                "status": "rejected"
            })
            
            # Return negative priority to indicate rejection
            return -1.0
        
        return priority
    
    def _process_queue(self, operation_type: str):
        """
        Process the operation queue according to harmonic timing.
        
        Args:
            operation_type: Type of operations to process
        """
        queue_obj = self.operation_queues.get(operation_type)
        if queue_obj is None:
            self.logger.error(f"No queue found for operation type: {operation_type}")
            return
            
        self.logger.info(f"Started processing thread for {operation_type} operations")
        
        while self.running:
            try:
                if not queue_obj.empty():
                    # Get next operation
                    # The queue now contains (priority, counter, operation) tuples
                    priority, _, operation = queue_obj.get(block=False)
                    
                    # Extract operation details
                    repository = operation["repository"]
                    method_name = operation["method_name"]
                    args = operation["args"]
                    kwargs = operation["kwargs"]
                    data_context = operation["data_context"]
                    
                    # Get method to call
                    # First try with _direct_ prefix
                    method = getattr(repository, f"_direct_{method_name}", None)
                    
                    # If not found, try the method name directly
                    if method is None:
                        method = getattr(repository, method_name, None)
                        
                    if method is None:
                        self.logger.error(f"Method _direct_{method_name} not found in repository")
                        continue
                    
                    # Record start of execution
                    start_time = datetime.now()
                    self.operation_metrics[operation_type].append({
                        "priority": priority,
                        "timestamp": start_time,
                        "data_type": data_context.get("data_type", "unknown"),
                        "status": "executing"
                    })
                    
                    # Execute operation
                    try:
                        result = method(*args, **kwargs)
                        status = "success"
                    except Exception as e:
                        self.logger.error(f"Error executing {method_name}: {e}")
                        result = None
                        status = "error"
                    
                    # Record completion
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    self.operation_metrics[operation_type].append({
                        "priority": priority,
                        "timestamp": end_time,
                        "duration": duration,
                        "data_type": data_context.get("data_type", "unknown"),
                        "status": status
                    })
                    
                    # Update system load based on operation duration
                    self._update_system_load(operation_type, duration)
                    
                    # Track data transformation if context contains transformation data
                    if "source_domain" in data_context and "target_domain" in data_context and "actant_id" in data_context:
                        self.track_data_transformation(
                            source_domain=data_context["source_domain"],
                            target_domain=data_context["target_domain"],
                            actant_id=data_context["actant_id"],
                            transformation_type=operation_type,
                            metadata={
                                "method": method_name,
                                "duration": duration,
                                "priority": priority,
                                "status": status,
                                **{k: v for k, v in data_context.items() if k not in ["source_domain", "target_domain", "actant_id"]}
                            }
                        )
                    
                    # Calculate adaptive sleep based on system load and harmonics
                    if self.adaptive_timing:
                        cycle_pos = self.get_cycle_position()
                        tonic = self.get_tonic_value(operation_type)
                        
                        # Base sleep time varies with cycle position
                        base_sleep = 0.01 + 0.04 * np.sin(2 * np.pi * cycle_pos)
                        
                        # Adjust for system load
                        load_factor = 1.0 + self.system_load
                        
                        # Adjust for operation type
                        type_factor = 1.0
                        if operation_type == OperationType.WRITE.value:
                            type_factor = 1.2  # Writes need more time to settle
                        elif operation_type == OperationType.DELETE.value:
                            type_factor = 1.5  # Deletes need even more time
                            
                        # Calculate final sleep time
                        harmonic_sleep = base_sleep * load_factor * type_factor * tonic
                        
                        # Ensure reasonable bounds
                        harmonic_sleep = max(0.001, min(0.2, harmonic_sleep))
                        
                        time.sleep(harmonic_sleep)
                    else:
                        # Fixed sleep time
                        time.sleep(0.01)
                else:
                    # No operations, sleep briefly
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in {operation_type} processing loop: {e}")
                time.sleep(0.1)
    
    def _update_system_load(self, operation_type: str, duration: float):
        """
        Update system load metric based on operation duration.
        
        Args:
            operation_type: Type of operation
            duration: Duration of the operation in seconds
        """
        # Simple exponential moving average for system load
        alpha = 0.2  # Smoothing factor
        
        # Normalize duration to a load factor (0.0 to 1.0)
        # Assuming operations should typically take < 0.1 seconds
        load_factor = min(1.0, duration / 0.1)
        
        # Update system load
        self.system_load = (1 - alpha) * self.system_load + alpha * load_factor
    
    def update_eigenspace_stability(self, stability: float):
        """
        Update the current eigenspace stability metric.
        
        Args:
            stability: Stability value (0.0 to 1.0)
        """
        self.eigenspace_stability = max(0.0, min(1.0, stability))
        
    def update_pattern_coherence(self, coherence: float):
        """
        Update the current pattern coherence metric.
        
        Args:
            coherence: Coherence value (0.0 to 1.0)
        """
        self.pattern_coherence = max(0.0, min(1.0, coherence))
        
    def update_resonance_level(self, resonance: float):
        """
        Update the current resonance level metric.
        
        Args:
            resonance: Resonance value (0.0 to 1.0)
        """
        self.resonance_level = max(0.0, min(1.0, resonance))
        
    def track_data_transformation(self, 
                               source_domain: str,
                               target_domain: str,
                               actant_id: str,
                               transformation_type: str,
                               metadata: Dict[str, Any] = None) -> None:
        """
        Track a data transformation for visualization and analysis.
        
        Args:
            source_domain: Domain where the data originated
            target_domain: Domain where the data is being transformed to
            actant_id: ID of the actant being transformed
            transformation_type: Type of transformation being applied
            metadata: Additional metadata about the transformation
        """
        if metadata is None:
            metadata = {}
            
        # Create transformation record
        transformation = {
            "timestamp": datetime.now().isoformat(),
            "source_domain": source_domain,
            "target_domain": target_domain,
            "actant_id": actant_id,
            "transformation_type": transformation_type,
            "cycle_position": self.get_cycle_position(),
            "eigenspace_stability": self.eigenspace_stability,
            "pattern_coherence": self.pattern_coherence,
            "metadata": metadata
        }
        
        # Log the transformation
        self.logger.info(f"Data transformation: {actant_id} from {source_domain} to {target_domain} ({transformation_type})")
        
        # Add to transformation log
        self.transformation_log.append(transformation)
        
        # Track by actant
        self.actant_transformations[actant_id].append(transformation)
        
    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete transformation log.
        
        Returns:
            List of transformation records
        """
        return self.transformation_log
    
    def get_actant_transformations(self, actant_id: str) -> List[Dict[str, Any]]:
        """
        Get transformations for a specific actant.
        
        Args:
            actant_id: ID of the actant
            
        Returns:
            List of transformation records for the actant
        """
        return self.actant_transformations.get(actant_id, [])
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for the harmonic I/O service.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "system_state": {
                "eigenspace_stability": self.eigenspace_stability,
                "pattern_coherence": self.pattern_coherence,
                "resonance_level": self.resonance_level,
                "system_load": self.system_load
            },
            "queue_sizes": {
                op_type.value: self.operation_queues[op_type.value].qsize()
                for op_type in OperationType
            },
            "operation_counts": {
                op_type: len(metrics)
                for op_type, metrics in self.operation_metrics.items()
            },
            "cycle_position": self.get_cycle_position(),
            "transformation_metrics": {
                "total_transformations": len(self.transformation_log),
                "unique_actants": len(self.actant_transformations),
                "domains": list(set([t["source_domain"] for t in self.transformation_log] + 
                                  [t["target_domain"] for t in self.transformation_log]))
            }
        }
        
        # Add topology metrics if available
        if hasattr(self, '_topology_metrics'):
            metrics["topology"] = {
                "resonance_center_count": len(self._topology_metrics.get('resonance_centers', [])),
                "interference_pattern_count": len(self._topology_metrics.get('interference_patterns', [])),
                "field_density_center_count": len(self._topology_metrics.get('field_density_centers', [])),
                "flow_vector_count": len(self._topology_metrics.get('flow_vectors', [])),
                "effective_dimensionality": self._topology_metrics.get('effective_dimensionality', 0),
                "meta_pattern_count": len(self._topology_metrics.get('meta_pattern_influence', {}))
            }
        
        return metrics
        
    def _adjust_parameters_from_meta_pattern(self, pattern_type, confidence, frequency):
        """Adjust harmonic parameters based on meta-pattern detection.
        
        Args:
            pattern_type: Type of meta-pattern detected
            confidence: Confidence level of the meta-pattern (0.0 to 1.0)
            frequency: Frequency of the meta-pattern (number of instances)
        """
        # Calculate adjustment factors based on meta-pattern properties
        frequency_factor = min(1.0, frequency / 10.0)  # Cap at 1.0 for 10+ instances
        impact_score = confidence * frequency_factor
        
        # Log the adjustment calculation
        self.logger.info(f"Meta-pattern impact score: {impact_score:.4f}")
        self.logger.info(f"  Based on confidence: {confidence:.2f}, frequency factor: {frequency_factor:.2f}")
        
        # Store current parameters for logging
        old_frequency = self.base_frequency
        old_stability = self.eigenspace_stability
        old_coherence = self.pattern_coherence
        
        # Adjust parameters based on pattern type
        if pattern_type == 'object_evolution':
            # Increase base frequency to accelerate pattern detection in this promising area
            # Higher confidence and frequency = more acceleration
            adjustment_factor = 1.0 + (impact_score * 0.5)  # Max 50% increase
            self.set_base_frequency(self.base_frequency * adjustment_factor)
            
            # Adjust eigenspace stability based on confidence
            # Higher confidence = higher stability to preserve the detected patterns
            self.eigenspace_stability = max(0.3, min(0.9, self.eigenspace_stability + (impact_score * 0.2)))
            
            # Adjust pattern coherence to focus on similar patterns
            self.pattern_coherence = max(0.3, min(0.9, self.pattern_coherence + (impact_score * 0.15)))
            
        elif pattern_type == 'causal_cascade':
            # For causal cascades, prioritize stability over speed
            adjustment_factor = 1.0 + (impact_score * 0.3)  # Max 30% increase
            self.set_base_frequency(self.base_frequency * adjustment_factor)
            
            # Increase stability significantly to preserve causal relationships
            self.eigenspace_stability = max(0.4, min(0.95, self.eigenspace_stability + (impact_score * 0.3)))
            
            # Moderate increase in coherence
            self.pattern_coherence = max(0.3, min(0.9, self.pattern_coherence + (impact_score * 0.1)))
            
        elif pattern_type == 'convergent_influence':
            # For convergent influences, balance speed and coherence
            adjustment_factor = 1.0 + (impact_score * 0.4)  # Max 40% increase
            self.set_base_frequency(self.base_frequency * adjustment_factor)
            
            # Moderate increase in stability
            self.eigenspace_stability = max(0.3, min(0.9, self.eigenspace_stability + (impact_score * 0.15)))
            
            # Significant increase in coherence to focus on convergent patterns
            self.pattern_coherence = max(0.4, min(0.95, self.pattern_coherence + (impact_score * 0.25)))
            
        else:  # Default for unknown pattern types
            # Conservative adjustments for unknown pattern types
            adjustment_factor = 1.0 + (impact_score * 0.2)  # Max 20% increase
            self.set_base_frequency(self.base_frequency * adjustment_factor)
            
            # Small increases in stability and coherence
            self.eigenspace_stability = max(0.3, min(0.8, self.eigenspace_stability + (impact_score * 0.1)))
            self.pattern_coherence = max(0.3, min(0.8, self.pattern_coherence + (impact_score * 0.1)))
        
        # Log the parameter adjustments
        self.logger.info(f"Adjusted harmonic parameters based on meta-pattern: {pattern_type}")
        self.logger.info(f"  Base frequency: {old_frequency:.4f} → {self.base_frequency:.4f}")
        self.logger.info(f"  Eigenspace stability: {old_stability:.4f} → {self.eigenspace_stability:.4f}")
        self.logger.info(f"  Pattern coherence: {old_coherence:.4f} → {self.pattern_coherence:.4f}")
        
    def _calculate_topology_derived_metrics(self):
        """Calculate derived metrics from topology data for feedback."""
        try:
            # Calculate resonance density (number of resonance centers per unit space)
            resonance_count = len(self._topology_metrics['resonance_centers'])
            effective_dim = max(1, self._topology_metrics['effective_dimensionality'])
            
            # Resonance density normalized by effective dimensionality
            resonance_density = resonance_count / effective_dim if effective_dim > 0 else 0
            
            # Calculate interference complexity (based on interference patterns)
            interference_count = len(self._topology_metrics['interference_patterns'])
            interference_complexity = interference_count / (effective_dim * 2) if effective_dim > 0 else 0
            
            # Calculate flow coherence (based on alignment of flow vectors)
            flow_vectors = self._topology_metrics['flow_vectors']
            flow_coherence = 0.5  # Default value
            
            if len(flow_vectors) >= 2:
                # Simple measure: ratio of density centers to flow vectors
                # More aligned flow = fewer density centers per flow vector
                density_centers = len(self._topology_metrics['field_density_centers'])
                flow_count = len(flow_vectors)
                if flow_count > 0:
                    ratio = density_centers / flow_count
                    # Normalize: lower ratio = higher coherence
                    flow_coherence = max(0.1, min(0.9, 1.0 - (ratio / 5.0)))
            
            # Calculate stability trend
            stability_trend = self._topology_metrics['stability_trend']
            stability_direction = 0  # Default: no change
            if len(stability_trend) >= 3:
                # Simple linear regression slope
                x = list(range(len(stability_trend)))  # Time points
                y = stability_trend  # Stability values
                n = len(x)
                
                # Calculate slope using least squares
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
                sum_xx = sum(x_i * x_i for x_i in x)
                
                # Avoid division by zero
                if n * sum_xx - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    stability_direction = slope
            
            # Calculate coherence trend similarly
            coherence_trend = self._topology_metrics['coherence_trend']
            coherence_direction = 0  # Default: no change
            if len(coherence_trend) >= 3:
                x = list(range(len(coherence_trend)))  # Time points
                y = coherence_trend  # Coherence values
                n = len(x)
                
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
                sum_xx = sum(x_i * x_i for x_i in x)
                
                if n * sum_xx - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                    coherence_direction = slope
            
            # Calculate meta-pattern influence
            meta_influences = self._topology_metrics['meta_pattern_influence'].values()
            total_influence = sum(m.get('influence_score', 0) for m in meta_influences)
            
            # Log derived metrics
            self.logger.info("Topology-derived metrics:")
            self.logger.info(f"  Resonance density: {resonance_density:.4f}")
            self.logger.info(f"  Interference complexity: {interference_complexity:.4f}")
            self.logger.info(f"  Flow coherence: {flow_coherence:.4f}")
            self.logger.info(f"  Stability trend: {stability_direction:.4f}")
            self.logger.info(f"  Coherence trend: {coherence_direction:.4f}")
            self.logger.info(f"  Meta-pattern influence: {total_influence:.4f}")
            
            # Use these metrics to further adjust the system
            self._adjust_from_topology_metrics(
                resonance_density, 
                interference_complexity,
                flow_coherence,
                stability_direction,
                coherence_direction,
                total_influence
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating topology-derived metrics: {e}")
            
    def _adjust_from_topology_metrics(self, resonance_density, interference_complexity, 
                                     flow_coherence, stability_direction, coherence_direction,
                                     meta_influence):
        """Adjust system parameters based on topology-derived metrics.
        
        Args:
            resonance_density: Density of resonance centers
            interference_complexity: Complexity of interference patterns
            flow_coherence: Coherence of flow vectors
            stability_direction: Direction of stability trend
            coherence_direction: Direction of coherence trend
            meta_influence: Total influence of meta-patterns
        """
        # Store current parameters for logging
        old_frequency = self.base_frequency
        
        # Adjust base frequency based on topology metrics
        # Higher resonance density = slightly lower frequency (more careful processing)
        # Higher flow coherence = higher frequency (faster processing in coherent areas)
        # Positive stability trend = higher frequency (system is becoming more stable)
        
        # Calculate adjustment factor
        resonance_factor = max(0.8, 1.0 - (resonance_density * 0.2))  # Max 20% decrease
        coherence_factor = 1.0 + (flow_coherence * 0.2)  # Max 20% increase
        trend_factor = 1.0 + (stability_direction * 2.0)  # More significant impact
        
        # Combine factors with weights
        combined_factor = (resonance_factor * 0.3) + (coherence_factor * 0.4) + (trend_factor * 0.3)
        
        # Apply adjustment, but only if it's significant
        if abs(combined_factor - 1.0) > 0.05:  # 5% threshold
            # Ensure reasonable bounds
            adjustment = max(0.8, min(1.2, combined_factor))
            self.set_base_frequency(self.base_frequency * adjustment)
            
            # Log the adjustment
            self.logger.info("Adjusted parameters from topology metrics:")
            self.logger.info(f"  Resonance factor: {resonance_factor:.4f}")
            self.logger.info(f"  Coherence factor: {coherence_factor:.4f}")
            self.logger.info(f"  Trend factor: {trend_factor:.4f}")
            self.logger.info(f"  Combined factor: {combined_factor:.4f}")
            self.logger.info(f"  Base frequency: {old_frequency:.4f} → {self.base_frequency:.4f}")
            
    def set_base_frequency(self, new_frequency):
        """Set base frequency and notify observers.
        
        Args:
            new_frequency: New base frequency value
        """
        # Store old value for comparison
        old_frequency = self.base_frequency
        
        # Set new value with bounds
        self.base_frequency = max(0.01, min(2.0, new_frequency))
        
        # Only notify if there's a significant change
        if abs(old_frequency - self.base_frequency) > 0.001:
            # Notify observers about the frequency change
            if self.event_bus:
                self.publish_event(
                    "harmonic.frequency.changed",
                    {
                        "old_frequency": old_frequency,
                        "new_frequency": self.base_frequency,
                        "adjustment_factor": self.base_frequency / old_frequency if old_frequency > 0 else 1.0
                    },
                    immediate=True
                )
                
    def _publish_topology_metrics_update(self):
        """Publish topology metrics update event."""
        if self.event_bus:
            # Create a summary of topology metrics
            metrics_summary = {
                "resonance_center_count": len(self._topology_metrics['resonance_centers']),
                "interference_pattern_count": len(self._topology_metrics['interference_patterns']),
                "field_density_center_count": len(self._topology_metrics['field_density_centers']),
                "flow_vector_count": len(self._topology_metrics['flow_vectors']),
                "effective_dimensionality": self._topology_metrics['effective_dimensionality'],
                "principal_dimension_count": len(self._topology_metrics['principal_dimensions']),
                "meta_pattern_count": len(self._topology_metrics['meta_pattern_influence']),
                "stability": self.eigenspace_stability,
                "coherence": self.pattern_coherence,
                "resonance": self.resonance_level,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish the metrics update
            self.publish_event(
                "topology.metrics.updated",
                metrics_summary,
                immediate=False  # Use harmonic timing
            )
            
    def __del__(self):
        """Ensure threads are stopped on deletion."""
        self.stop()
