document.addEventListener('DOMContentLoaded', function() {
    const balanceEl = document.getElementById('balance');
    const availableBalanceEl = document.getElementById('available-balance');
    const winrateEl = document.getElementById('winrate');
    const tableBody = document.querySelector('#orders-table tbody');
    const emptyState = document.getElementById('empty-state');

    // --- Core UI Update Function ---
    function updateUI(data) {
        console.log('updateUI called with data:', data);
        
        // Check if data is valid
        if (!data || typeof data !== 'object') {
            console.warn('Invalid data received:', data);
            return;
        }

        // Update stat cards with new format
        const balance = data.balance ? parseFloat(data.balance).toFixed(2) : '0.00';
        const available_balance = data.available_balance ? parseFloat(data.available_balance).toFixed(2) : '0.00';
        const winrate = data.winrate ? parseFloat(data.winrate).toFixed(1) : '0.0';
        
        console.log('Updating UI with:', { balance, available_balance, winrate });
        
        balanceEl.textContent = `$${balance}`;
        availableBalanceEl.textContent = `$${available_balance}`;
        winrateEl.textContent = `${winrate}%`;

        // Update table
        tableBody.innerHTML = ''; // Clear existing rows
        
        if (!data.orders || data.orders.length === 0) {
            console.log('No orders to display, showing empty state');
            // Show empty state
            emptyState.style.display = 'block';
            document.querySelector('.orders-table').style.display = 'none';
        } else {
            console.log('Displaying', data.orders.length, 'orders');
            // Hide empty state and show table
            emptyState.style.display = 'none';
            document.querySelector('.orders-table').style.display = 'table';
            
            data.orders.forEach((order, index) => {
                console.log('Processing order', index, ':', order);
                const row = tableBody.insertRow();
                for (let i = 0; i < 8; i++) {
                    const cell = row.insertCell();
                    let cellData = order[i];

                    // Format numbers for PnL cells
                    if (i === 6 || i === 7) {
                        cellData = parseFloat(cellData).toFixed(2);
                         if (i === 7) cellData += '%'; // Add percent sign
                    }

                    cell.textContent = cellData;

                    // Apply styling based on content
                    const pnlPercent = parseFloat(order[7]);
                    if (i === 6 || i === 7) { // PnL ($) and PnL (%)
                        if (pnlPercent > 0) {
                            cell.className = 'pnl-positive';
                        } else if (pnlPercent < 0) {
                            cell.className = 'pnl-negative';
                        }
                    } else if (i === 2) { // Side
                        if (String(cellData).toLowerCase() === 'long') {
                            cell.className = 'side-long';
                        } else if (String(cellData).toLowerCase() === 'short') {
                            cell.className = 'side-short';
                        }
                    }
                }
            });
        }
    }

    // --- Initial Data Load ---
    const initialData = JSON.parse(document.getElementById('initial-data').textContent);
    console.log('Initial data loaded:', initialData);
    updateUI(initialData);

    let fallbackInterval;
    let websocketConnected = false;

    // --- WebSocket Connection ---
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const dashboardSocket = new WebSocket(
        protocol + '://' + window.location.host + '/ws/dashboard/'
    );

    dashboardSocket.onopen = function(e) {
        console.log("WebSocket connection established.");
        websocketConnected = true;
        // Clear fallback polling if WebSocket is connected
        if (fallbackInterval) {
            clearInterval(fallbackInterval);
            fallbackInterval = null;
        }
    };

    dashboardSocket.onmessage = function(e) {
        try {
            const data = JSON.parse(e.data);
            console.log('WebSocket data received:', data);
            // Check if data has message property, otherwise use data directly
            const messageData = data.message || data;
            updateUI(messageData);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error, 'Raw data:', e.data);
        }
    };

    dashboardSocket.onclose = function(e) {
        console.error('Dashboard socket closed unexpectedly. Code:', e.code, 'Reason:', e.reason);
        websocketConnected = false;
        // Start fallback polling when WebSocket disconnects
        startFallbackPolling();
    };

    dashboardSocket.onerror = function(err) {
        console.error('WebSocket encountered error:', err);
        websocketConnected = false;
        dashboardSocket.close();
    };

    // --- Fallback REST API Polling ---
    function startFallbackPolling() {
        if (fallbackInterval) return; // Already polling
        
        console.log('Starting fallback REST API polling...');
        fallbackInterval = setInterval(async () => {
            if (websocketConnected) {
                clearInterval(fallbackInterval);
                fallbackInterval = null;
                return;
            }
            
            try {
                const response = await fetch('/api/dashboard-data/');
                if (response.ok) {
                    const data = await response.json();
                    console.log('REST API data received:', data);
                    updateUI(data);
                }
            } catch (error) {
                console.error('Error fetching data via REST API:', error);
            }
        }, 3000); // Poll every 3 seconds
    }

    // Start fallback polling after a delay if WebSocket doesn't connect
    setTimeout(() => {
        if (!websocketConnected) {
            startFallbackPolling();
        }
    }, 2000);
});