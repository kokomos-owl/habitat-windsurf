# Quick-Start Testing Guide

## Natural System Testing Approach

### 1. Basic POC Verification 🟡
```python
# Test current capacity
async def verify_basic_sequence(self):
    # Create test pattern
    pattern = await self.create_test_pattern()
    
    # Verify processing
    result = await self.process_pattern(pattern)
    assert result.pattern_processed
    
    # Record emergence point
    self.record_emergence({
        'type': 'basic_processing',
        'metrics': result.metrics
    })
```

### 2. Emergence Observation 🌱
```python
# Track emergence points
class EmergenceTracker:
    def record_point(self, observation):
        self.points.append({
            'timestamp': time.now(),
            'metrics': observation.metrics,
            'potential': observation.emergence_indicators
        })
```

### 3. Evolution Preparation 🔄
```python
# Mark evolution paths
class EvolutionMarker:
    def mark_potential(self, point):
        return {
            'current_capacity': point.metrics,
            'growth_potential': point.evolution_indicators,
            'adaptation_paths': point.future_possibilities
        }
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
