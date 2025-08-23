document.addEventListener('DOMContentLoaded', function() {
    const balanceEl = document.getElementById('balance');
    const availableBalanceEl = document.getElementById('available-balance');
    const winrateEl = document.getElementById('winrate');
    const tableBody = document.querySelector('#orders-table tbody');

    // --- Core UI Update Function ---
    function updateUI(data) {
        // Check if data is valid
        if (!data || typeof data !== 'object') {
            console.warn('Invalid data received:', data);
            return;
        }

        // Update info bar
        const balance = data.balance ? data.balance.toFixed(2) : '0.00';
        const available_balance = data.available_balance ? data.available_balance.toFixed(2) : '0.00';
        balanceEl.innerHTML = `<strong>Balance:</strong> $${balance}`;
        availableBalanceEl.innerHTML = `<strong>Available:</strong> $${available_balance}`;
        winrateEl.innerHTML = `<strong>Global Winrate:</strong> ${data.winrate || 0}%`;

        // Update table
        tableBody.innerHTML = ''; // Clear existing rows
        if (!data.orders || data.orders.length === 0) {
            const row = tableBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 8;
            cell.textContent = 'No open orders.';
            cell.style.textAlign = 'center';
        } else {
            data.orders.forEach(order => {
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
    updateUI(initialData);

    // --- WebSocket Connection ---
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const dashboardSocket = new WebSocket(
        protocol + '://' + window.location.host + '/ws/dashboard/'
    );

    dashboardSocket.onopen = function(e) {
        console.log("WebSocket connection established.");
    };

    dashboardSocket.onmessage = function(e) {
        try {
            const data = JSON.parse(e.data);
            // Check if data has message property, otherwise use data directly
            const messageData = data.message || data;
            updateUI(messageData);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error, 'Raw data:', e.data);
        }
    };

    dashboardSocket.onclose = function(e) {
        console.error('Dashboard socket closed unexpectedly. Attempting to reconnect...');
        // Optional: Implement a reconnection logic here
        setTimeout(function() {
            // Reconnect logic
        }, 1000);
    };

    dashboardSocket.onerror = function(err) {
        console.error('WebSocket encountered error: ', err.message, 'Closing socket');
        dashboardSocket.close();
    };
});