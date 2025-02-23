# Ghost Koenig Integration Implementation Guide

## Overview
This document details the technical implementation of the Habitat pattern system integration with Ghost's Koenig editor, based on our working toolbar implementation.

## 1. Toolbar Implementation

### Dynamic Toolbar Integration
```javascript
// Initialize toolbar with cleanup
(function() {
    // Clean up existing observers
    if (window.habitatToolbarObserver) {
        window.habitatToolbarObserver.disconnect();
    }
    
    // Add Ghost-compliant styles
    const styleEl = document.createElement('style');
    styleEl.textContent = `
        .settings-menu-container {
            transition: transform 600ms ease-out !important;
        }
    `;
    document.head.appendChild(styleEl);
    
    // Initialize observer
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.addedNodes) {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) {
                        const toolbar = node.matches('.not-kg-prose.fixed ul') ? 
                            node : node.querySelector('.not-kg-prose.fixed ul');
                        if (toolbar) {
                            addGraphButton(toolbar);
                        }
                    }
                });
            }
        }
    });
    
    window.habitatToolbarObserver = observer;
})();
```

### Button Implementation
```javascript
function addGraphButton(toolbar) {
    // Ghost-compliant button structure
    const li = document.createElement('li');
    li.className = 'group relative m-0 flex p-0 first:m-0';
    li.setAttribute('data-kg-toolbar-button', 'graph');
    
    const button = document.createElement('button');
    button.className = 'my-1 flex h-8 w-9 cursor-pointer items-center justify-center rounded-md transition hover:bg-grey-200/80 dark:bg-grey-950 dark:hover:bg-grey-900 bg-white';
    button.setAttribute('aria-label', 'Graph Selection');
    
    // Ghost-style icon
    button.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="size-6 overflow-visible transition text-black dark:text-white">
            <circle cx="12" cy="6" r="2.5" fill="currentColor"/>
            <circle cx="6" cy="17" r="2.5" fill="currentColor"/>
            <circle cx="18" cy="17" r="2.5" fill="currentColor"/>
            <path stroke="currentColor" stroke-width="1.5" stroke-linecap="round" d="M12 8.5L7.5 15M12 8.5l4.5 6.5M7.5 17h10"/>
        </svg>
    `;
}
```

### Event Handling
```javascript
function initializeEvents(button) {
    button.onclick = () => {
        // Get current selection
        const selection = window.getSelection();
        
        // Process selection
        const selectionData = {
            text: selection.toString(),
            range: {
                startOffset: selection.getRangeAt(0).startOffset,
                endOffset: selection.getRangeAt(0).endOffset
            }
        };
        
        // Trigger settings menu
        const settingsToggle = document.querySelector('.settings-menu-toggle');
        if (settingsToggle) {
            settingsToggle.click();
        }
        
        // Dispatch pattern request
        document.dispatchEvent(new CustomEvent('habitatGraphRequest', {
            detail: selectionData
        }));
    };
}
```

## 2. Testing Implementation

### UI Component Tests
```javascript
describe('HabitatToolbar', () => {
    beforeEach(() => {
        document.body.innerHTML = `
            <div class="not-kg-prose fixed">
                <ul></ul>
            </div>
        `;
    });
    
    test('adds graph button to toolbar', () => {
        const toolbar = document.querySelector('.not-kg-prose.fixed ul');
        addGraphButton(toolbar);
        
        const button = toolbar.querySelector('[data-kg-toolbar-button="graph"]');
        expect(button).toBeTruthy();
        expect(button.querySelector('svg')).toBeTruthy();
    });
    
    test('handles dark mode classes', () => {
        const toolbar = document.querySelector('.not-kg-prose.fixed ul');
        addGraphButton(toolbar);
        
        const button = toolbar.querySelector('button');
        expect(button.className).toContain('dark:bg-grey-950');
        expect(button.className).toContain('dark:hover:bg-grey-900');
    });
});
```

### Integration Tests
```javascript
describe('HabitatIntegration', () => {
    test('handles selection and settings menu', () => {
        // Setup
        document.body.innerHTML = `
            <div class="not-kg-prose fixed">
                <ul></ul>
            </div>
            <button class="settings-menu-toggle"></button>
        `;
        
        const toolbar = document.querySelector('.not-kg-prose.fixed ul');
        addGraphButton(toolbar);
        
        // Mock selection
        const selection = {
            toString: () => 'Selected text',
            getRangeAt: () => ({
                startOffset: 0,
                endOffset: 12
            })
        };
        window.getSelection = () => selection;
        
        // Test button click
        const button = toolbar.querySelector('[data-kg-toolbar-button="graph"] button');
        button.click();
        
        // Verify settings menu interaction
        const settingsToggle = document.querySelector('.settings-menu-toggle');
        expect(settingsToggle.click).toHaveBeenCalled();
    });
});
```

### Event System Tests
```javascript
describe('HabitatEvents', () => {
    test('dispatches graph request event', () => {
        // Setup
        const eventSpy = jest.fn();
        document.addEventListener('habitatGraphRequest', eventSpy);
        
        // Mock selection
        const selection = {
            toString: () => 'Test text',
            getRangeAt: () => ({
                startOffset: 0,
                endOffset: 9
            })
        };
        window.getSelection = () => selection;
        
        // Trigger button click
        const button = document.querySelector('[data-kg-toolbar-button="graph"] button');
        button.click();
        
        // Verify event
        expect(eventSpy).toHaveBeenCalledWith(
            expect.objectContaining({
                detail: {
                    text: 'Test text',
                    range: {
                        startOffset: 0,
                        endOffset: 9
                    }
                }
            })
        );
    });
});
```

## 3. Selection Management

### Cross-Card Selection Handler
```javascript
// Selection management across cards
class SelectionManager {
    handleCrossCardSelection(range) {
        const cards = this.getCardsInRange(range);
        return {
            cards: cards,
            boundaries: this.calculateBoundaries(cards, range),
            metadata: this.gatherMetadata(cards)
        };
    }

    getCardsInRange(range) {
        return Array.from(document.querySelectorAll('.koenig-card'))
            .filter(card => range.intersectsNode(card));
    }
}
```

### Selection State Preservation
```javascript
// Selection state preservation
class SelectionStateManager {
    preserveState(selection) {
        return {
            range: {
                startContainer: this.serializeNode(selection.range.startContainer),
                endContainer: this.serializeNode(selection.range.endContainer),
                startOffset: selection.range.startOffset,
                endOffset: selection.range.endOffset
            },
            cards: this.serializeCards(selection.cards),
            metadata: selection.metadata
        };
    }

    restoreState(state) {
        const range = document.createRange();
        // Restore selection state
        return this.createSelectionFromState(state);
    }
}
```

## 4. Testing Implementation

### Card System Tests
```javascript
describe('HabitatPatternCard', () => {
    let card;
    
    beforeEach(() => {
        card = new HabitatPatternCard();
    });

    test('card registration', () => {
        expect(card.type).toBe('dom');
        expect(card.name).toBe('habitat-pattern');
    });

    test('card lifecycle', () => {
        card.didInsertElement();
        expect(card.events.size).toBeGreaterThan(0);
    });
});
```

### Toolbar Tests
```javascript
describe('HabitatToolbar', () => {
    test('tool registration', () => {
        const toolbar = new HabitatToolbar();
        toolbar.registerHabitatTool();
        
        const tool = editor.getTool('habitat-pattern');
        expect(tool).toBeDefined();
        expect(tool.visibility).toBe('text-selected');
    });

    test('selection handling', () => {
        const toolbar = new HabitatToolbar();
        const selection = mockSelection();
        
        toolbar.updateState(selection);
        expect(toolbar.selectionState).toBeDefined();
    });
});
```

### Integration Tests
```javascript
describe('Habitat Integration', () => {
    test('end-to-end pattern creation', async () => {
        // Setup
        const editor = await setupKoenigEditor();
        const toolbar = new HabitatToolbar(editor);
        
        // Create selection
        const selection = await createTestSelection(editor);
        
        // Trigger pattern creation
        toolbar.handlePatternRequest(selection);
        
        // Verify
        const pattern = await getCreatedPattern();
        expect(pattern).toBeDefined();
        expect(pattern.selection).toEqual(selection);
    });
});
```

## 5. Error Handling

### Card Error Boundaries
```javascript
class HabitatCardErrorBoundary {
    handleError(error) {
        console.error('Card Error:', error);
        this.setState({ hasError: true });
        this.notifyEditor(error);
    }

    recover() {
        this.resetState();
        this.reloadCard();
    }
}
```

### Selection Error Recovery
```javascript
class SelectionRecoveryManager {
    handleSelectionError(error) {
        const lastValidState = this.getLastValidState();
        if (lastValidState) {
            this.restoreState(lastValidState);
        } else {
            this.resetSelection();
        }
    }
}
```

## 6. Performance Considerations

### Selection Optimization
```javascript
class SelectionOptimizer {
    optimizeRange(range) {
        if (this.isLargeSelection(range)) {
            return this.createOptimizedRange(range);
        }
        return range;
    }

    isLargeSelection(range) {
        return this.getSelectionSize(range) > this.threshold;
    }
}
```

### State Management Optimization
```javascript
class StateManager {
    constructor() {
        this.stateCache = new Map();
        this.lastUpdate = null;
    }

    shouldUpdateState(newState) {
        return this.hasSignificantChanges(newState);
    }

    hasSignificantChanges(newState) {
        return !this.stateCache.has(this.getStateHash(newState));
    }
}
```

## Next Steps

1. **Implementation Priority**
   - Card system integration
   - Toolbar extension
   - Selection management
   - Error handling
   - Performance optimization

2. **Testing Focus**
   - Unit test coverage
   - Integration test scenarios
   - Performance benchmarks
   - Error recovery testing
   - Cross-browser testing

3. **Documentation**
   - API documentation
   - Integration guide
   - Test coverage report
   - Performance guidelines
   - Troubleshooting guide
