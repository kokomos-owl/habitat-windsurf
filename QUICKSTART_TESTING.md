# Quick-Start Testing Guide

**Last Updated**: 2025-02-20

## Pattern-Aware RAG Testing Guide

This guide provides a quick start for testing the Pattern-Aware RAG system with its natural evolution capabilities.

### 1. Sequential Testing 🔄
```python
class TestPatternAwareSequence:
    """Test Pattern-Aware RAG sequence validation"""
    
    async def test_pattern_aware_sequence(self, pattern_aware_rag):
        # Process test pattern
        response, context = await pattern_aware_rag.process_with_patterns(
            query="test pattern",
            context={'mode': 'sequence_validation'}
        )
        
        # Validate context components
        assert context.temporal_context is not None
        assert context.state_space is not None
        assert context.evolution_metrics is not None
        
        # Verify metrics
        assert 0.0 <= context.coherence_level <= 1.0
        assert context.evolution_metrics.stability > 0.5
```

### 2. Metrics Validation 📊
```python
class PatternMetricsValidator:
    """Validate Pattern-Aware RAG metrics"""
    
    def __init__(self):
        self.metrics = PatternMetrics(
            coherence=0.8,          # Integration quality
            emergence_rate=0.7,      # Formation velocity
            cross_pattern_flow=0.6,  # Pattern interactions
            energy_state=0.8,        # System vitality
            adaptation_rate=0.7,     # Responsiveness
            stability=0.9           # Consistency
        )
    
    def validate_metrics(self, context):
        # Verify all metrics are within natural bounds
        return all(
            0.0 <= getattr(context.evolution_metrics, attr) <= 1.0
            for attr in self.metrics.__dict__
        )
```

### 3. Capacity Testing 🔬
```python
class TestSequenceCapacity:
    """Test Pattern-Aware RAG capacity"""
    
    async def test_sequence_capacity(self, pattern_aware_rag):
        test_cases = [
            "Simple test pattern",
            "Complex pattern with multiple concepts",
            "Linked pattern sequence"
        ]
        
        results = []
        for case in test_cases:
            # Process through sequence
            result = await pattern_aware_rag.process_with_patterns(
                query=case,
                context={'test_case': case}
            )
            results.append(result)
        
        # Analyze results
        metrics = {
            'total_tests': len(results),
            'successful_sequences': sum(
                1 for r in results 
                if r[0].sequence_completed
            ),
            'average_coherence': sum(
                r[1].coherence_level 
                for r in results
            ) / len(results)
        }
        
        return metrics
```
```python
# Location: tests/pattern_aware_rag/integration/test_pattern_aware_rag_integration.py

class TestPatternAwareRAGIntegration:
    """Test Pattern-Aware RAG with Learning Windows"""
    
    async def test_pattern_flow_control(self):
        """First Test: Basic pattern flow through windows"""
        rag = PatternAwareRAG()
        result = await rag.process_with_patterns("test query")
        
        # Key assertions:
        assert result.window_state in [CLOSED, OPENING, OPEN]
        assert result.stability_score > 0
```

### 2. Natural Flow Control 🟡
```python
# Location: tests/pattern_aware_rag/integration/test_natural_flow.py

class TestNaturalFlow:
    """Test natural flow control"""
    
    async def test_basic_flow(self):
        """Second Test: Natural flow control"""
        window = LearningWindow()
        
        # Process patterns with delays
        for pattern in test_patterns:
            delay = await window.calculate_delay(pattern)
            await asyncio.sleep(delay)
            result = await window.process_pattern(pattern)
            
            # Key assertions:
            assert result.is_processed
            assert window.stability.is_maintained
```

### 3. Basic Integration Test 🟡
```python
# Location: tests/pattern_aware_rag/integration/test_basic_integration.py

class TestBasicIntegration:
    """Test basic system integration"""
    
    async def test_simple_query(self):
        """Third Test: Simple query processing"""
        system = PatternAwareRAG()
        
        # Process single query
        result = await system.process_query("test query")
        
        # Key assertions:
        assert result.is_successful
        assert result.patterns_processed > 0
```

## Test Order and Focus

1. **Start Here** → Pattern Flow
   - Basic pattern processing
   - Window state changes
   - Simple stability checks

2. **Then** → Natural Flow
   - Delay calculations
   - Pattern processing
   - Basic stability

3. **Finally** → Integration
   - Simple queries
   - Basic patterns
   - Core functionality

## Key Points to Watch

### Natural Flow
- Window states transition correctly
- Delays increase with instability
- Patterns process in order

### Stability
- System remains stable under load
- Back pressure works correctly
- Pattern memory forms properly

### Integration
- Components work together
- Natural flow maintained
- System protects itself

## Common Issues to Check

1. **Window States**
   - CLOSED → OPENING transition
   - Proper state progression
   - No invalid states

2. **Delays**
   - Increase with instability
   - Maintain ordering
   - Natural progression

3. **Pattern Processing**
   - Correct order
   - Proper memory formation
   - Safe evolution

## Quick Commands

```bash
# Run specific test
PYTHONPATH=/Users/prphillips/Documents/GitHub/habitat-windsurf/src python -m pytest tests/pattern_aware_rag/integration/test_pattern_aware_rag_integration.py -v

# Run all integration tests
PYTHONPATH=/Users/prphillips/Documents/GitHub/habitat-windsurf/src python -m pytest tests/pattern_aware_rag/integration/ -v

# Run with detailed output
PYTHONPATH=/Users/prphillips/Documents/GitHub/habitat-windsurf/src python -m pytest tests/pattern_aware_rag/integration/ -vv
```

## Next Steps After Basic Tests Pass

1. Add more complex pattern tests
2. Test multiple concurrent windows
3. Implement full PEI integration
4. Add E2E flow tests

Remember:
- Start simple
- Build gradually
- Watch stability
- Trust natural patterns

The system should feel natural and stable. If something feels forced, it probably needs adjustment.
