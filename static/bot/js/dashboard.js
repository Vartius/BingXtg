document.addEventListener('DOMContentLoaded', function() {
    const balanceEl = document.getElementById('balance');
    const availableBalanceEl = document.getElementById('available-balance');
    const winrateEl = document.getElementById('winrate');
    const totalTradesEl = document.getElementById('total-trades');
    const winsEl = document.getElementById('wins');
    const lossesEl = document.getElementById('losses');
    const profitEl = document.getElementById('profit');
    const roiEl = document.getElementById('roi');
    const tableBody = document.querySelector('#orders-table tbody');
    const emptyState = document.getElementById('empty-state');
    const simulationModeEl = document.getElementById('simulation-mode');

    // --- Core UI Update Function ---
    function updateUI(data) {
        console.log('updateUI called with data:', data);
        
        // Check if data is valid
        if (!data || typeof data !== 'object') {
            console.warn('Invalid data received:', data);
            return;
        }

        // Show/hide simulation mode indicator
        if (data.is_simulation === true) {
            simulationModeEl.style.display = 'block';
        } else {
            simulationModeEl.style.display = 'none';
        }

        // Update stat cards with new format
        const balance = data.balance ? parseFloat(data.balance).toFixed(2) : '0.00';
        const available_balance = data.available_balance ? parseFloat(data.available_balance).toFixed(2) : '0.00';
        const winrate = data.winrate ? parseFloat(data.winrate).toFixed(1) : '0.0';
        const total_trades = data.total_trades ? parseInt(data.total_trades) : 0;
        const wins = data.wins ? parseInt(data.wins) : 0;
        const losses = data.losses ? parseInt(data.losses) : 0;
        const profit = data.profit ? parseFloat(data.profit).toFixed(2) : '0.00';
        const roi = data.roi ? parseFloat(data.roi).toFixed(1) : '0.0';
        
        console.log('Updating UI with:', { balance, available_balance, winrate, total_trades, wins, losses, profit, roi });
        
        balanceEl.textContent = `$${balance}`;
        availableBalanceEl.textContent = `$${available_balance}`;
        winrateEl.textContent = `${winrate}%`;
        totalTradesEl.textContent = total_trades;
        winsEl.textContent = wins;
        lossesEl.textContent = losses;
        
        // Color-code profit based on positive/negative
        profitEl.textContent = `$${profit}`;
        if (parseFloat(profit) > 0) {
            profitEl.style.color = 'var(--ctp-green)';
        } else if (parseFloat(profit) < 0) {
            profitEl.style.color = 'var(--ctp-red)';
        } else {
            profitEl.style.color = 'var(--ctp-teal)';
        }
        
        // Color-code ROI based on positive/negative
        roiEl.textContent = `${roi}%`;
        if (parseFloat(roi) > 0) {
            roiEl.style.color = 'var(--ctp-green)';
        } else if (parseFloat(roi) < 0) {
            roiEl.style.color = 'var(--ctp-red)';
        } else {
            roiEl.style.color = 'var(--ctp-teal)';
        }

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
                
                // Handle both array and object formats
                let orderData;
                if (Array.isArray(order)) {
                    // Array format: [trade_id, channel_id, coin, direction, targets, leverage, sl, margin, entry_price, current_price, pnl, pnl_percent, status]
                    orderData = order;
                } else {
                    // Object format: convert to array for backwards compatibility
                    orderData = [
                        order.trade_id,
                        order.channel_id,
                        order.coin,
                        order.direction,
                        order.targets || '-',
                        order.leverage ? `${parseFloat(order.leverage).toFixed(0)}x` : '-',
                        order.sl ? parseFloat(order.sl).toFixed(4) : '-',
                        parseFloat(order.margin).toFixed(2),
                        parseFloat(order.entry_price).toFixed(4),
                        parseFloat(order.current_price).toFixed(4),
                        parseFloat(order.pnl).toFixed(2),
                        parseFloat(order.pnl_percent).toFixed(2),
                        order.status || 'OPEN'
                    ];
                }
                
                for (let i = 0; i < 13; i++) {
                    const cell = row.insertCell();
                    let cellData = orderData[i];

                    // Format numbers for PnL cells (indices 10 and 11)
                    if (i === 10 || i === 11) {
                        cellData = parseFloat(cellData).toFixed(2);
                        if (i === 11) {
                            cellData += '%'; // Just add percent sign, don't multiply by 100
                        }
                    }

                    // Handle status column (index 12)
                    if (i === 12) {
                        const statusBadge = document.createElement('span');
                        statusBadge.className = `status-badge status-${cellData.toLowerCase()}`;
                        statusBadge.textContent = cellData;
                        cell.appendChild(statusBadge);
                    } else {
                        cell.textContent = cellData;
                    }

                    // Apply styling based on content
                    const pnlPercent = parseFloat(orderData[11]); // PnL % is at index 11
                    if (i === 10 || i === 11) { // PnL ($) and PnL (%)
                        if (pnlPercent > 0) {
                            cell.className = 'pnl-positive';
                        } else if (pnlPercent < 0) {
                            cell.className = 'pnl-negative';
                        }
                    } else if (i === 3) { // Side (direction column at index 3)
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
    const initialDataElement = document.getElementById('initial-data');
    if (initialDataElement) {
        try {
            const initialData = JSON.parse(initialDataElement.textContent);
            console.log('Initial data loaded:', initialData);
            updateUI(initialData);
        } catch (error) {
            console.error('Error parsing initial data:', error);
            const defaultData = {
                orders: [],
                balance: 0.0,
                available_balance: 0.0,
                winrate: 0.0,
                total_trades: 0,
                wins: 0,
                losses: 0,
                profit: 0.0,
                roi: 0.0,
                is_simulation: false
            };
            updateUI(defaultData);
        }
    } else {
        console.warn('Initial data element not found, using default data');
        const defaultData = {
            orders: [],
            balance: 0.0,
            available_balance: 0.0,
            winrate: 0.0,
            total_trades: 0,
            wins: 0,
            losses: 0,
            profit: 0.0,
            roi: 0.0,
            is_simulation: false
        };
        updateUI(defaultData);
    }

    let fallbackInterval;
    let websocketConnected = false;

    // --- WebSocket Connection ---
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const dashboardSocket = new WebSocket(
        protocol + '://' + window.location.host + '/ws/dashboard/'
    );

    // Show connection status indicator
    const statusEl = document.getElementById('connection-status');
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = 'Connecting...';
        statusEl.className = 'connection-status status-disconnected';
    }

    dashboardSocket.onopen = function(e) {
        console.log("WebSocket connection established.");
        websocketConnected = true;
        // Clear fallback polling if WebSocket is connected
        if (fallbackInterval) {
            clearInterval(fallbackInterval);
            fallbackInterval = null;
        }
        
        // Show connection status if there's a status element
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = 'Connected';
            statusEl.className = 'connection-status status-connected';
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
        
        // Show connection status if there's a status element
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = 'Disconnected (using polling)';
            statusEl.className = 'connection-status status-disconnected';
        }
        
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