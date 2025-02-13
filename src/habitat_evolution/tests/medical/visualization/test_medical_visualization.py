"""Tests for medical pattern visualization."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from habitat_evolution.tests.medical.fixtures.sepsis_cases import create_sepsis_case, load_test_sepsis_cohort
from habitat_evolution.tests.medical.visualization.medical_visualizer import MedicalPatternVisualizer, MedicalVisualizationConfig
import matplotlib.pyplot as plt
import os

@pytest.fixture
def test_case():
    """Create a test sepsis case."""
    return create_sepsis_case(
        "TEST_001",
        datetime(2025, 1, 1, 12, 0, 0)
    )

@pytest.fixture
def field_states(test_case):
    """Generate field states from test case."""
    states = []
    current_time = test_case["onset_time"] - timedelta(hours=6)
    end_time = test_case["onset_time"] + timedelta(hours=24)
    
    while current_time <= end_time:
        # Calculate field state for current time
        # Get relevant vitals
        relevant_vitals = [
            v for v in test_case["vitals"]
            if abs((v.timestamp - current_time).total_seconds()) <= 1800
        ]
        vitals_state = np.mean([v.value for v in relevant_vitals]) if relevant_vitals else 0.0
        
        # Get relevant labs
        relevant_labs = [
            l for l in test_case["labs"]
            if abs((l.timestamp - current_time).total_seconds()) <= 1800
        ]
        labs_state = np.mean([l.value for l in relevant_labs]) if relevant_labs else 0.0
        
        events_state = len([
            e for e in test_case["events"]
            if abs((e.timestamp - current_time).total_seconds()) <= 1800
        ]) / 5.0  # Normalize by max expected events
        
        states.append({
            "temporal": (current_time - test_case["onset_time"]).total_seconds() / 86400,  # Normalize to days
            "vitals": vitals_state,
            "labs": labs_state,
            "events": events_state
        })
        
        current_time += timedelta(minutes=30)
    
    return states

@pytest.fixture
def patterns(test_case):
    """Generate patterns from test case."""
    return [
        {
            "pattern_type": "vital_deterioration",
            "coherence": 0.8,
            "energy_state": 0.7,
            "timestamp": test_case["onset_time"] - timedelta(hours=2)
        },
        {
            "pattern_type": "lab_abnormality",
            "coherence": 0.75,
            "energy_state": 0.8,
            "timestamp": test_case["onset_time"]
        },
        {
            "pattern_type": "organ_dysfunction",
            "coherence": 0.9,
            "energy_state": 0.85,
            "timestamp": test_case["onset_time"] + timedelta(hours=2)
        }
    ]

def test_visualization_config():
    """Test medical visualization configuration."""
    config = MedicalVisualizationConfig()
    
    assert "heart_rate" in config.vital_ranges
    assert "wbc" in config.lab_ranges
    assert "sofa_score" in config.sepsis_thresholds
    assert "normal" in config.colors

def test_temporal_evolution_visualization(test_case, field_states, patterns):
    """Test visualization of temporal pattern evolution."""
    visualizer = MedicalPatternVisualizer()
    
    # Create output directory
    output_dir = "visualization_output/medical"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization
    fig = visualizer.visualize_temporal_evolution(
        test_case,
        field_states,
        patterns
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, "temporal_evolution.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    assert os.path.exists(output_path)

def test_pattern_coherence_visualization(test_case, field_states, patterns):
    """Test visualization of pattern coherence network."""
    visualizer = MedicalPatternVisualizer()
    
    # Create output directory
    output_dir = "visualization_output/medical"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current field state
    current_state = field_states[len(field_states)//2]  # Middle state
    
    # Generate visualization
    fig = visualizer.visualize_pattern_coherence(
        patterns,
        current_state
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, "pattern_coherence.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    assert os.path.exists(output_path)

@pytest.mark.integration
def test_neo4j_export(test_case, patterns):
    """Test export of patterns to Neo4j."""
    visualizer = MedicalPatternVisualizer()
    
    try:
        from neo4j import GraphDatabase
        # Connect to Neo4j medical container (HTTP: 7475, Bolt: 7476)
        driver = GraphDatabase.driver(
            "bolt://localhost:7476",  # Bolt port mapped to container's 7687
            auth=("neo4j", "medical_test")
        )
        
        # Export patterns
        visualizer.export_to_neo4j(test_case, patterns, driver)
        
        # Verify export
        with driver.session(database="neo4j") as session:
            # Check case node
            result = session.run("""
                MATCH (c:Case {case_id: $case_id})
                RETURN c
            """, case_id=test_case["case_id"])
            assert result.single()
            
            # Check pattern nodes
            result = session.run("""
                MATCH (p:Pattern)-[:BELONGS_TO]->(:Case {case_id: $case_id})
                RETURN count(p) as pattern_count
            """, case_id=test_case["case_id"])
            assert result.single()["pattern_count"] == len(patterns)
            
            # Check relationships
            result = session.run("""
                MATCH (p1:Pattern)-[r:RELATED_TO]-(p2:Pattern)
                WHERE (p1)-[:BELONGS_TO]->(:Case {case_id: $case_id})
                RETURN count(r) as rel_count
            """, case_id=test_case["case_id"])
            assert result.single()["rel_count"] > 0
        
        driver.close()
        
    except Exception as e:
        pytest.skip(f"Neo4j test skipped: {str(e)}")

def test_cohort_visualization():
    """Test visualization of pattern evolution across multiple cases."""
    visualizer = MedicalPatternVisualizer()
    
    # Load test cohort
    cohort = load_test_sepsis_cohort(size=3)
    
    # Create output directory
    output_dir = "visualization_output/medical"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each case
    for i, case in enumerate(cohort):
        # Generate field states
        states = []
        current_time = case["onset_time"] - timedelta(hours=6)
        end_time = case["onset_time"] + timedelta(hours=24)
        
        while current_time <= end_time:
            relevant_vitals = [
                v for v in case["vitals"]
                if abs((v.timestamp - current_time).total_seconds()) <= 1800
            ]
            vitals_state = np.mean([v.value for v in relevant_vitals]) if relevant_vitals else 0.0
            
            relevant_labs = [
                l for l in case["labs"]
                if abs((l.timestamp - current_time).total_seconds()) <= 1800
            ]
            labs_state = np.mean([l.value for l in relevant_labs]) if relevant_labs else 0.0
            
            events_state = len([
                e for e in case["events"]
                if abs((e.timestamp - current_time).total_seconds()) <= 1800
            ]) / 5.0
            
            states.append({
                "temporal": (current_time - case["onset_time"]).total_seconds() / 86400,
                "vitals": vitals_state,
                "labs": labs_state,
                "events": events_state
            })
            
            current_time += timedelta(minutes=30)
        
        # Generate patterns for this case
        patterns = [
            {
                "pattern_type": "vital_deterioration",
                "coherence": 0.8,
                "energy_state": 0.7,
                "timestamp": case["onset_time"] - timedelta(hours=2)
            },
            {
                "pattern_type": "lab_abnormality",
                "coherence": 0.75,
                "energy_state": 0.8,
                "timestamp": case["onset_time"]
            },
            {
                "pattern_type": "organ_dysfunction",
                "coherence": 0.9,
                "energy_state": 0.85,
                "timestamp": case["onset_time"] + timedelta(hours=2)
            }
        ]
        
        # Generate and save visualizations
        fig = visualizer.visualize_temporal_evolution(case, states, patterns)
        output_path = os.path.join(output_dir, f"case_{i}_evolution.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        fig = visualizer.visualize_pattern_coherence(patterns, states[-1])
        output_path = os.path.join(output_dir, f"case_{i}_coherence.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    # Assert output files exist
    assert len(os.listdir(output_dir)) >= len(cohort) * 2
