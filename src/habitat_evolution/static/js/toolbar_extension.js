(function() {
    // Clean up any existing observers to prevent duplicates
    if (window.habitatToolbarObserver) {
        window.habitatToolbarObserver.disconnect();
    }
    
    // Add custom styles for smooth transition
    const styleEl = document.createElement('style');
    styleEl.textContent = `
        .settings-menu-container {
            transition: transform 600ms ease-out !important;
        }
        .settings-menu-expanded .settings-menu-container {
            transition: transform 600ms ease-out !important;
        }
        .habitat-graph-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
    `;
    document.head.appendChild(styleEl);
    
    // Load graph viewer script
    const script = document.createElement('script');
    script.src = '/static/js/graph_viewer.js';
    document.head.appendChild(script);
    
    function addGraphButton(toolbar) {
        // Remove any existing graph buttons first
        const existingButton = toolbar.querySelector('[data-kg-toolbar-button="graph"]');
        if (existingButton) {
            existingButton.remove();
        }
        
        const li = document.createElement('li');
        li.className = 'group relative m-0 flex p-0 first:m-0';
        li.setAttribute('data-kg-toolbar-button', 'graph');
        
        const button = document.createElement('button');
        button.className = 'my-1 flex h-8 w-9 cursor-pointer items-center justify-center rounded-md transition hover:bg-grey-200/80 dark:bg-grey-950 dark:hover:bg-grey-900 bg-white';
        button.setAttribute('aria-label', 'Graph Selection');
        button.setAttribute('data-kg-active', 'false');
        button.type = 'button';
        
        button.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="size-6 overflow-visible transition text-black dark:text-white">
                <circle cx="12" cy="6" r="2.5" fill="currentColor"/>
                <circle cx="6" cy="17" r="2.5" fill="currentColor"/>
                <circle cx="18" cy="17" r="2.5" fill="currentColor"/>
                <path stroke="currentColor" stroke-width="1.5" stroke-linecap="round" d="M12 8.5L7.5 15M12 8.5l4.5 6.5M7.5 17h10"/>
            </svg>
        `;

        const tooltip = document.createElement('div');
        tooltip.className = 'invisible absolute -top-8 left-1/2 z-[1000] flex -translate-x-1/2 items-center gap-1 whitespace-nowrap rounded-md bg-black py-1 font-sans text-2xs font-medium text-white group-hover:visible dark:bg-grey-900 px-[1rem]';
        
        const tooltipText = document.createElement('span');
        tooltipText.textContent = 'Graph Selection';
        tooltip.appendChild(tooltipText);
        
        button.onclick = () => {
            const selection = window.getSelection();
            const selectedText = selection.toString();
            
            if (selectedText) {
                // Dispatch event to graph viewer
                const event = new CustomEvent('habitatGraphRequest', {
                    detail: { text: selectedText }
                });
                document.dispatchEvent(event);
            } else {
                console.log('No text selected');
            }
        };
        
        li.appendChild(button);
        li.appendChild(tooltip);
        
        const lastChild = toolbar.lastElementChild;
        if (!lastChild.classList.contains('w-px')) {
            const divider = document.createElement('li');
            divider.className = 'm-0 w-px self-stretch bg-grey-300/80 dark:bg-grey-900';
            toolbar.appendChild(divider);
        }
        toolbar.appendChild(li);
    }
    
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
    
    // Store observer reference for cleanup
    window.habitatToolbarObserver = observer;
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    const existingToolbar = document.querySelector('.not-kg-prose.fixed ul');
    if (existingToolbar) {
        addGraphButton(existingToolbar);
    }
    
    console.log('Habitat toolbar initialized');
})();
