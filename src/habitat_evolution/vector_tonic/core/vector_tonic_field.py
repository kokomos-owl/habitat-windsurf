"""
Vector-Tonic Field implementation for Habitat Evolution.

This module provides the core VectorTonicField class that represents
a vector field for statistical pattern analysis. It implements the
mathematical foundation for detecting and analyzing patterns in
time-series data using vector-tonic methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FieldPoint:
    """
    Represents a single point in a vector-tonic field.
    
    Attributes:
        coordinates: The coordinates of the point in the field
        value: The scalar value at this point
        gradient: The gradient vector at this point
        potential: The potential value at this point
    """
    coordinates: Tuple[float, ...]
    value: float
    gradient: Tuple[float, ...]
    potential: float


class VectorTonicField:
    """
    Represents a vector-tonic field for statistical pattern analysis.
    
    A vector-tonic field treats statistical patterns as fields with their
    own dynamics rather than static correlations. This enables the detection
    of resonance between patterns and the calculation of potential gradients
    where new patterns are likely to emerge.
    
    The field is represented as a multi-dimensional grid where each point
    has both scalar and vector properties. The field evolves over time
    according to the field equations defined in this class.
    """
    
    def __init__(self, dimensions: Tuple[int, ...], resolution: float = 1.0):
        """
        Initialize a new vector-tonic field.
        
        Args:
            dimensions: The dimensions of the field (e.g., (10, 10) for a 2D field)
            resolution: The resolution of the field (distance between grid points)
        """
        self.dimensions = dimensions
        self.resolution = resolution
        self.field_data = np.zeros(dimensions)
        self.gradient_field = np.zeros(dimensions + (len(dimensions),))
        self.potential_field = np.zeros(dimensions)
        self.resonance_map = {}
        
    def _calculate_shape(self) -> Tuple[int, ...]:
        """
        Calculate the shape of the field arrays based on dimensions.
        
        Returns:
            The shape tuple for numpy arrays
        """
        return self.dimensions
    
    def set_value(self, coordinates: Tuple[int, ...], value: float) -> None:
        """
        Set the value at a specific point in the field.
        
        Args:
            coordinates: The coordinates of the point
            value: The value to set
        """
        self.field_data[coordinates] = value
        
    def get_value(self, coordinates: Tuple[int, ...]) -> float:
        """
        Get the value at a specific point in the field.
        
        Args:
            coordinates: The coordinates of the point
            
        Returns:
            The value at the specified coordinates
        """
        return float(self.field_data[coordinates])
    
    def calculate_gradients(self) -> None:
        """
        Calculate the gradient field based on the current field data.
        
        This computes the partial derivatives in each dimension to
        determine the direction and magnitude of change at each point.
        """
        # For each dimension, calculate the gradient using central differences
        for dim in range(len(self.dimensions)):
            # Create slices for forward and backward differences
            slice_forward = [slice(None)] * len(self.dimensions)
            slice_backward = [slice(None)] * len(self.dimensions)
            
            # Handle boundary conditions
            slice_forward[dim] = slice(0, -2)
            slice_backward[dim] = slice(2, None)
            
            # Calculate central difference
            forward_values = self.field_data[tuple(slice_forward)]
            backward_values = self.field_data[tuple(slice_backward)]
            
            # Pad the gradient to match original dimensions
            pad_width = [(0, 0)] * len(self.dimensions)
            pad_width[dim] = (1, 1)
            
            # Store the gradient for this dimension
            gradient = np.pad((forward_values - backward_values) / (2 * self.resolution), pad_width)
            
            # Update the gradient field
            slice_all = [slice(None)] * len(self.dimensions)
            self.gradient_field[tuple(slice_all + [dim])] = gradient
    
    def calculate_potential(self) -> None:
        """
        Calculate the potential field based on the gradient field.
        
        The potential field represents areas where new patterns are likely
        to emerge, based on the convergence or divergence of the gradient field.
        """
        # Calculate divergence of the gradient field
        divergence = np.zeros(self.dimensions)
        
        for dim in range(len(self.dimensions)):
            # Create slices for forward and backward differences
            slice_forward = [slice(None)] * len(self.dimensions) + [dim]
            slice_backward = [slice(None)] * len(self.dimensions) + [dim]
            
            # Handle boundary conditions
            slice_forward[dim] = slice(0, -2)
            slice_backward[dim] = slice(2, None)
            
            # Calculate central difference of the gradient
            forward_gradient = self.gradient_field[tuple(slice_forward)]
            backward_gradient = self.gradient_field[tuple(slice_backward)]
            
            # Pad the divergence to match original dimensions
            pad_width = [(0, 0)] * len(self.dimensions)
            pad_width[dim] = (1, 1)
            
            # Add to the divergence
            divergence += np.pad(
                (forward_gradient - backward_gradient) / (2 * self.resolution), 
                pad_width
            )
        
        # The potential is the negative of the divergence
        self.potential_field = -divergence
    
    def detect_patterns(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect patterns in the field based on potential values.
        
        Args:
            threshold: The minimum potential value to consider as a pattern
            
        Returns:
            A list of detected patterns with their properties
        """
        patterns = []
        
        # Find local maxima in the potential field
        # For simplicity, we'll use a basic approach here
        # In a production system, this would use more sophisticated algorithms
        
        # Create a mask for points above the threshold
        mask = self.potential_field > threshold
        
        # Find connected components in the mask
        # This is a simplified approach - in practice, we would use
        # more sophisticated clustering algorithms
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(mask)
        
        for i in range(1, num_features + 1):
            # Get the coordinates of this pattern
            pattern_coords = np.where(labeled_array == i)
            
            # Convert to a list of coordinate tuples
            coords_list = list(zip(*pattern_coords))
            
            if not coords_list:
                continue
                
            # Find the point with maximum potential
            max_potential_idx = np.argmax([self.potential_field[coord] for coord in coords_list])
            center_coord = coords_list[max_potential_idx]
            
            # Calculate pattern properties
            pattern = {
                "id": f"pattern_{i}",
                "center": center_coord,
                "potential": float(self.potential_field[center_coord]),
                "size": len(coords_list),
                "coordinates": coords_list,
                "average_value": float(np.mean([self.field_data[coord] for coord in coords_list])),
                "quality_state": self._determine_quality_state(
                    float(self.potential_field[center_coord]), 
                    len(coords_list)
                )
            }
            
            patterns.append(pattern)
            
        return patterns
    
    def _determine_quality_state(self, potential: float, size: int) -> str:
        """
        Determine the quality state of a pattern based on its properties.
        
        Args:
            potential: The potential value of the pattern
            size: The size of the pattern (number of points)
            
        Returns:
            The quality state: "hypothetical", "emergent", or "stable"
        """
        if potential > 0.8 and size > 10:
            return "stable"
        elif potential > 0.6:
            return "emergent"
        else:
            return "hypothetical"
    
    def detect_resonance(self, other_field: 'VectorTonicField', threshold: float = 0.7) -> Dict[str, float]:
        """
        Detect resonance between this field and another field.
        
        Resonance occurs when patterns in different fields amplify or
        dampen each other. This method identifies such relationships.
        
        Args:
            other_field: Another vector-tonic field to compare with
            threshold: The minimum correlation to consider as resonance
            
        Returns:
            A dictionary mapping resonance IDs to strength values
        """
        # Ensure fields have the same dimensions
        if self.dimensions != other_field.dimensions:
            raise ValueError("Fields must have the same dimensions to detect resonance")
        
        # Calculate correlation between the two fields
        correlation = np.corrcoef(
            self.field_data.flatten(), 
            other_field.field_data.flatten()
        )[0, 1]
        
        # Calculate resonance based on correlation and potential fields
        resonance_strength = correlation * np.mean(
            np.abs(self.potential_field * other_field.potential_field)
        )
        
        # Only return resonance above the threshold
        if resonance_strength > threshold:
            resonance_id = f"resonance_{id(self)}_{id(other_field)}"
            self.resonance_map[resonance_id] = resonance_strength
            return {resonance_id: float(resonance_strength)}
        
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the field to a dictionary representation.
        
        Returns:
            A dictionary representation of the field
        """
        return {
            "dimensions": self.dimensions,
            "resolution": self.resolution,
            "field_data": self.field_data.tolist(),
            "potential_field": self.potential_field.tolist(),
            "resonance_map": self.resonance_map
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorTonicField':
        """
        Create a field from a dictionary representation.
        
        Args:
            data: A dictionary representation of the field
            
        Returns:
            A new VectorTonicField instance
        """
        field = cls(
            dimensions=tuple(data["dimensions"]),
            resolution=data["resolution"]
        )
        field.field_data = np.array(data["field_data"])
        field.potential_field = np.array(data["potential_field"])
        field.resonance_map = data["resonance_map"]
        
        # Recalculate gradients
        field.calculate_gradients()
        
        return field
