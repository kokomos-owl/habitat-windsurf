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


class OperationType(Enum):
    """Types of I/O operations that can be harmonized."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"


class HarmonicIOService:
    """
    Service that harmonizes I/O operations with system rhythms.
    
    This service ensures that database operations don't disrupt
    the natural evolution of eigenspaces and pattern detection.
    It uses harmonic timing to schedule operations in a way that
    preserves the continuity of pattern evolution.
    """
    
    def __init__(self, 
                 base_frequency: float = 0.1, 
                 harmonics: int = 3,
                 max_queue_size: int = 1000,
                 adaptive_timing: bool = True):
        """
        Initialize the harmonic I/O service.
        
        Args:
            base_frequency: Base frequency for harmonic cycles (Hz)
            harmonics: Number of harmonic overtones to consider
            max_queue_size: Maximum size of operation queues
            adaptive_timing: Whether to adapt timing based on system state
        """
        self.base_frequency = base_frequency
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
        
        # Processing threads
        self.processing_threads = {}
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
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
            self.operation_queues[operation_type].put((priority, operation), block=False)
            
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
                    priority, operation = queue_obj.get(block=False)
                    
                    # Extract operation details
                    repository = operation["repository"]
                    method_name = operation["method_name"]
                    args = operation["args"]
                    kwargs = operation["kwargs"]
                    data_context = operation["data_context"]
                    
                    # Get method to call
                    method = getattr(repository, f"_direct_{method_name}", None)
                    
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
            "cycle_position": self.get_cycle_position()
        }
        
        return metrics
        
    def __del__(self):
        """Ensure threads are stopped on deletion."""
        self.stop()
