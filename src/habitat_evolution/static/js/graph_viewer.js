class HabitatGraphViewer {
    constructor() {
        this.apiEndpoint = 'http://localhost:8000/api/process-text';
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.addEventListener('habitatGraphRequest', async (event) => {
            const text = event.detail.text;
            await this.processAndDisplayGraph(text);
        });
    }

    async processAndDisplayGraph(text) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayGraph(data);
        } catch (error) {
            console.error('Error processing graph:', error);
        }
    }

    displayGraph(data) {
        // Create graph container
        const container = document.createElement('div');
        container.className = 'habitat-graph-container';
        container.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
        `;

        // Add close button
        const closeButton = document.createElement('button');
        closeButton.textContent = '×';
        closeButton.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            border: none;
            background: none;
            font-size: 24px;
            cursor: pointer;
        `;
        closeButton.onclick = () => container.remove();

        // Display graph image
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${data.graph_image}`;
        img.style.cssText = 'max-width: 100%; height: auto;';

        // Add elements to container
        container.appendChild(closeButton);
        container.appendChild(img);

        // Add container to document
        document.body.appendChild(container);
    }
}

// Initialize graph viewer
window.habitatGraphViewer = new HabitatGraphViewer();
