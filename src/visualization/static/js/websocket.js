class WebSocketManager {
    constructor() {
        this.connect();
        this.onMessage = null;
    }

    connect() {
        const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
        this.ws = new WebSocket(`ws://localhost:8765/ws/${clientId}`);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (this.onMessage) {
                this.onMessage(message);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected. Reconnecting...');
            setTimeout(() => this.connect(), 1000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    send(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
}
