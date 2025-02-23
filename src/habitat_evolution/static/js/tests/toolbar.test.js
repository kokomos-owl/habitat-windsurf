/**
 * Test suite for Ghost toolbar integration
 */

describe('HabitatToolbar', () => {
    let toolbar;
    let container;
    
    beforeEach(() => {
        // Setup test DOM
        container = document.createElement('div');
        container.innerHTML = `
            <div class="not-kg-prose fixed">
                <ul class="toolbar-container"></ul>
            </div>
        `;
        document.body.appendChild(container);
        
        // Initialize toolbar
        eval(document.querySelector('script[src*="toolbar_extension.js"]').textContent);
    });
    
    afterEach(() => {
        // Cleanup
        document.body.removeChild(container);
        if (window.habitatToolbarObserver) {
            window.habitatToolbarObserver.disconnect();
        }
    });
    
    test('toolbar button creation', () => {
        const button = document.querySelector('[data-kg-toolbar-button="graph"]');
        expect(button).toBeTruthy();
        
        // Check button structure
        expect(button.querySelector('svg')).toBeTruthy();
        expect(button.getAttribute('aria-label')).toBe('Graph Selection');
    });
    
    test('tooltip functionality', () => {
        const tooltip = document.querySelector('[data-kg-toolbar-button="graph"] .tooltip');
        expect(tooltip).toBeTruthy();
        expect(tooltip.textContent).toContain('Graph Selection');
    });
    
    test('text selection handling', () => {
        const button = document.querySelector('[data-kg-toolbar-button="graph"]');
        const mockSelection = 'Test selection with drought and wildfire patterns';
        
        // Mock window.getSelection
        const originalGetSelection = window.getSelection;
        window.getSelection = jest.fn().mockReturnValue({
            toString: () => mockSelection
        });
        
        // Mock event dispatch
        const dispatchEventSpy = jest.spyOn(document, 'dispatchEvent');
        
        // Click button
        button.click();
        
        // Verify event dispatch
        expect(dispatchEventSpy).toHaveBeenCalledWith(
            expect.objectContaining({
                type: 'habitatGraphRequest',
                detail: { text: mockSelection }
            })
        );
        
        // Cleanup
        window.getSelection = originalGetSelection;
        dispatchEventSpy.mockRestore();
    });
    
    test('empty selection handling', () => {
        const button = document.querySelector('[data-kg-toolbar-button="graph"]');
        
        // Mock empty selection
        const originalGetSelection = window.getSelection;
        window.getSelection = jest.fn().mockReturnValue({
            toString: () => ''
        });
        
        // Mock console.log
        const consoleSpy = jest.spyOn(console, 'log');
        
        // Click button
        button.click();
        
        // Verify warning logged
        expect(consoleSpy).toHaveBeenCalledWith('No text selected');
        
        // Cleanup
        window.getSelection = originalGetSelection;
        consoleSpy.mockRestore();
    });
    
    test('multiple toolbar instances', () => {
        // Create second toolbar
        const container2 = document.createElement('div');
        container2.innerHTML = `
            <div class="not-kg-prose fixed">
                <ul class="toolbar-container"></ul>
            </div>
        `;
        document.body.appendChild(container2);
        
        // Verify only one button per toolbar
        const buttons = document.querySelectorAll('[data-kg-toolbar-button="graph"]');
        expect(buttons.length).toBe(2);
        
        // Cleanup
        document.body.removeChild(container2);
    });
    
    test('style injection', () => {
        const styleEl = document.querySelector('style');
        const styles = styleEl.textContent;
        
        // Verify transition styles
        expect(styles).toContain('transition: transform 600ms ease-out');
        expect(styles).toContain('settings-menu-container');
    });
});

describe('HabitatGraphViewer', () => {
    let graphViewer;
    
    beforeEach(() => {
        // Initialize graph viewer
        eval(document.querySelector('script[src*="graph_viewer.js"]').textContent);
        graphViewer = window.habitatGraphViewer;
    });
    
    test('event listener setup', () => {
        const addEventListenerSpy = jest.spyOn(document, 'addEventListener');
        
        // Verify event listener
        expect(addEventListenerSpy).toHaveBeenCalledWith(
            'habitatGraphRequest',
            expect.any(Function)
        );
        
        addEventListenerSpy.mockRestore();
    });
    
    test('graph display', async () => {
        const mockData = {
            graph_image: 'base64_encoded_image',
            nodes: [
                { id: 1, hazard_type: 'drought' },
                { id: 2, hazard_type: 'wildfire' }
            ],
            edges: [
                { source: 1, target: 2, type: 'INTERACTS_WITH' }
            ]
        };
        
        // Mock fetch
        global.fetch = jest.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(mockData)
        });
        
        // Process text
        await graphViewer.processAndDisplayGraph('test text');
        
        // Verify container creation
        const container = document.querySelector('.habitat-graph-container');
        expect(container).toBeTruthy();
        
        // Verify image
        const img = container.querySelector('img');
        expect(img.src).toContain('base64_encoded_image');
        
        // Cleanup
        container.remove();
        global.fetch.mockRestore();
    });
    
    test('error handling', async () => {
        // Mock fetch error
        global.fetch = jest.fn().mockRejectedValue(new Error('API Error'));
        
        // Mock console.error
        const consoleSpy = jest.spyOn(console, 'error');
        
        // Process text
        await graphViewer.processAndDisplayGraph('test text');
        
        // Verify error logging
        expect(consoleSpy).toHaveBeenCalledWith(
            'Error processing graph:',
            expect.any(Error)
        );
        
        // Cleanup
        consoleSpy.mockRestore();
        global.fetch.mockRestore();
    });
});
