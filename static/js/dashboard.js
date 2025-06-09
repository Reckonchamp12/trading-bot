/**
 * Walford Capitals Trading Bot - Dashboard JavaScript
 * Main dashboard functionality and UI interactions
 */

// Global variables
let dashboardData = {
    portfolio: null,
    positions: [],
    trades: [],
    signals: [],
    performance: []
};

let refreshInterval = null;
let chartInstances = {};

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startAutoRefresh();
});

/**
 * Initialize dashboard components
 */
function initializeDashboard() {
    console.log('Initializing Walford Capitals Dashboard...');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Load initial data
    loadDashboardData();
    
    // Setup real-time updates simulation
    setupRealTimeUpdates();
    
    // Initialize keyboard shortcuts
    setupKeyboardShortcuts();
    
    console.log('Dashboard initialized successfully');
}

/**
 * Setup event listeners for interactive elements
 */
function setupEventListeners() {
    // Portfolio refresh buttons
    document.querySelectorAll('[data-action="refresh"]').forEach(button => {
        button.addEventListener('click', function() {
            refreshDashboardData();
        });
    });
    
    // Quick action buttons
    document.querySelectorAll('[data-action="quick-trade"]').forEach(button => {
        button.addEventListener('click', function() {
            openQuickTradeModal();
        });
    });
    
    // Signal generation buttons
    document.querySelectorAll('[data-action="generate-signal"]').forEach(button => {
        button.addEventListener('click', function() {
            generateTradingSignal();
        });
    });
    
    // Modal form validations
    setupFormValidations();
    
    // Window resize handler
    window.addEventListener('resize', debounce(handleWindowResize, 300));
    
    // Page visibility change handler
    document.addEventListener('visibilitychange', handleVisibilityChange);
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Load dashboard data from server
 */
async function loadDashboardData() {
    try {
        showLoadingState();
        
        // Simulate API calls (in real implementation, these would be actual API endpoints)
        const promises = [
            fetchPortfolioData(),
            fetchPositionsData(),
            fetchTradesData(),
            fetchSignalsData(),
            fetchPerformanceData()
        ];
        
        const [portfolio, positions, trades, signals, performance] = await Promise.all(promises);
        
        dashboardData = {
            portfolio,
            positions,
            trades,
            signals,
            performance
        };
        
        updateDashboardUI();
        hideLoadingState();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showErrorMessage('Failed to load dashboard data. Please try again.');
        hideLoadingState();
    }
}

/**
 * Fetch portfolio data
 */
async function fetchPortfolioData() {
    try {
        const response = await fetch('/api/portfolio_performance');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.warn('Portfolio data unavailable, using fallback');
        return {
            current_value: 100000,
            initial_value: 100000,
            total_return: 0,
            performance: []
        };
    }
}

/**
 * Fetch positions data
 */
async function fetchPositionsData() {
    // In real implementation, this would fetch from /api/positions
    return [];
}

/**
 * Fetch trades data
 */
async function fetchTradesData() {
    // In real implementation, this would fetch from /api/trades
    return [];
}

/**
 * Fetch signals data
 */
async function fetchSignalsData() {
    // In real implementation, this would fetch from /api/signals
    return [];
}

/**
 * Fetch performance data
 */
async function fetchPerformanceData() {
    // In real implementation, this would fetch from /api/performance
    return [];
}

/**
 * Update dashboard UI with loaded data
 */
function updateDashboardUI() {
    updatePortfolioSummary();
    updatePositionsTable();
    updateTradesTable();
    updateSignalsPanel();
    updatePerformanceMetrics();
}

/**
 * Update portfolio summary cards
 */
function updatePortfolioSummary() {
    const portfolio = dashboardData.portfolio;
    if (!portfolio) return;
    
    // Update portfolio value
    const valueElement = document.querySelector('[data-metric="portfolio-value"]');
    if (valueElement) {
        valueElement.textContent = formatCurrency(portfolio.current_value);
    }
    
    // Update total return
    const returnElement = document.querySelector('[data-metric="total-return"]');
    if (returnElement) {
        const returnValue = portfolio.total_return || 0;
        returnElement.textContent = formatPercentage(returnValue);
        returnElement.className = `${returnValue >= 0 ? 'text-success' : 'text-danger'}`;
    }
}

/**
 * Update positions table
 */
function updatePositionsTable() {
    const positions = dashboardData.positions;
    const tableBody = document.querySelector('#positions-table tbody');
    
    if (!tableBody) return;
    
    if (positions.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-muted py-4">
                    <i class="fas fa-layer-group fa-2x mb-2"></i>
                    <div>No open positions</div>
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = positions.map(position => `
        <tr>
            <td><strong>${position.symbol}</strong></td>
            <td>${formatNumber(position.quantity)}</td>
            <td>${formatCurrency(position.avg_price)}</td>
            <td>${formatCurrency(position.current_price)}</td>
            <td class="${position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                ${formatCurrency(position.unrealized_pnl)}
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="viewPosition('${position.symbol}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-danger" onclick="closePosition('${position.symbol}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

/**
 * Update trades table
 */
function updateTradesTable() {
    const trades = dashboardData.trades;
    const tableBody = document.querySelector('#trades-table tbody');
    
    if (!tableBody) return;
    
    if (trades.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center text-muted py-4">
                    <i class="fas fa-exchange-alt fa-2x mb-2"></i>
                    <div>No recent trades</div>
                </td>
            </tr>
        `;
        return;
    }
    
    tableBody.innerHTML = trades.slice(0, 10).map(trade => `
        <tr>
            <td>${formatDateTime(trade.executed_at)}</td>
            <td><strong>${trade.symbol}</strong></td>
            <td>
                <span class="badge bg-${trade.side === 'buy' ? 'success' : 'danger'}">
                    ${trade.side.toUpperCase()}
                </span>
            </td>
            <td>${formatNumber(trade.quantity)}</td>
            <td>${formatCurrency(trade.price)}</td>
            <td class="${trade.pnl >= 0 ? 'text-success' : 'text-danger'}">
                ${formatCurrency(trade.pnl)}
            </td>
            <td>
                <span class="badge bg-${getStatusColor(trade.status)}">
                    ${trade.status}
                </span>
            </td>
        </tr>
    `).join('');
}

/**
 * Update signals panel
 */
function updateSignalsPanel() {
    const signals = dashboardData.signals;
    const signalsContainer = document.querySelector('#signals-container');
    
    if (!signalsContainer) return;
    
    if (signals.length === 0) {
        signalsContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-signal fa-3x text-muted mb-3"></i>
                <p class="text-muted">No active signals</p>
            </div>
        `;
        return;
    }
    
    signalsContainer.innerHTML = signals.slice(0, 5).map(signal => `
        <div class="d-flex align-items-center p-3 border-bottom">
            <div class="me-3">
                <i class="fas fa-arrow-${signal.signal_type === 'buy' ? 'up text-success' : signal.signal_type === 'sell' ? 'down text-danger' : 'right text-warning'} fa-lg"></i>
            </div>
            <div class="flex-fill">
                <div class="d-flex justify-content-between">
                    <strong>${signal.symbol}</strong>
                    <span class="badge bg-${signal.signal_type === 'buy' ? 'success' : signal.signal_type === 'sell' ? 'danger' : 'warning'}">
                        ${signal.signal_type.toUpperCase()}
                    </span>
                </div>
                <small class="text-muted">
                    Confidence: ${formatPercentage(signal.confidence)}
                    <span class="ms-2">${formatTime(signal.created_at)}</span>
                </small>
            </div>
        </div>
    `).join('');
}

/**
 * Update performance metrics
 */
function updatePerformanceMetrics() {
    const performance = dashboardData.performance;
    if (!performance || performance.length === 0) return;
    
    // Update performance indicators
    const indicators = {
        'sharpe-ratio': performance.sharpe_ratio || 0,
        'max-drawdown': performance.max_drawdown || 0,
        'win-rate': performance.win_rate || 0,
        'total-trades': performance.total_trades || 0
    };
    
    Object.entries(indicators).forEach(([key, value]) => {
        const element = document.querySelector(`[data-metric="${key}"]`);
        if (element) {
            if (key === 'sharpe-ratio') {
                element.textContent = formatNumber(value, 3);
            } else if (key.includes('rate') || key.includes('drawdown')) {
                element.textContent = formatPercentage(value);
            } else {
                element.textContent = formatNumber(value);
            }
        }
    });
}

/**
 * Start auto-refresh for real-time updates
 */
function startAutoRefresh() {
    // Refresh every 30 seconds when page is visible
    refreshInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            refreshDashboardData(true); // Silent refresh
        }
    }, 30000);
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

/**
 * Refresh dashboard data
 */
async function refreshDashboardData(silent = false) {
    if (!silent) {
        showRefreshIndicator();
    }
    
    try {
        await loadDashboardData();
        
        if (!silent) {
            showSuccessMessage('Dashboard updated successfully');
        }
        
        // Update last refresh time
        updateLastRefreshTime();
        
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
        if (!silent) {
            showErrorMessage('Failed to refresh dashboard data');
        }
    } finally {
        if (!silent) {
            hideRefreshIndicator();
        }
    }
}

/**
 * Setup real-time updates simulation
 */
function setupRealTimeUpdates() {
    // Simulate price updates every 5 seconds
    setInterval(() => {
        updatePriceDisplays();
    }, 5000);
    
    // Simulate status updates
    setInterval(() => {
        updateStatusIndicators();
    }, 10000);
}

/**
 * Update price displays with simulated real-time data
 */
function updatePriceDisplays() {
    const priceElements = document.querySelectorAll('[data-price]');
    
    priceElements.forEach(element => {
        const currentPrice = parseFloat(element.textContent.replace(/[^0-9.-]+/g, ''));
        if (currentPrice && currentPrice > 0) {
            // Simulate small price movements (-2% to +2%)
            const change = (Math.random() - 0.5) * 0.04;
            const newPrice = currentPrice * (1 + change);
            
            element.textContent = formatCurrency(newPrice);
            
            // Add visual feedback for price changes
            element.classList.remove('price-up', 'price-down');
            if (change > 0) {
                element.classList.add('price-up');
            } else if (change < 0) {
                element.classList.add('price-down');
            }
            
            // Remove classes after animation
            setTimeout(() => {
                element.classList.remove('price-up', 'price-down');
            }, 1000);
        }
    });
}

/**
 * Update status indicators
 */
function updateStatusIndicators() {
    const statusElements = document.querySelectorAll('[data-status]');
    
    statusElements.forEach(element => {
        const isOnline = Math.random() > 0.1; // 90% uptime simulation
        element.className = isOnline ? 'status-online' : 'status-offline';
        element.textContent = isOnline ? 'Online' : 'Offline';
    });
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + R: Refresh dashboard
        if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
            event.preventDefault();
            refreshDashboardData();
        }
        
        // Ctrl/Cmd + T: Quick trade
        if ((event.ctrlKey || event.metaKey) && event.key === 't') {
            event.preventDefault();
            openQuickTradeModal();
        }
        
        // Ctrl/Cmd + S: Generate signal
        if ((event.ctrlKey || event.metaKey) && event.key === 's') {
            event.preventDefault();
            generateTradingSignal();
        }
        
        // Escape: Close modals
        if (event.key === 'Escape') {
            closeAllModals();
        }
    });
}

/**
 * Setup form validations
 */
function setupFormValidations() {
    const forms = document.querySelectorAll('.needs-validation');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

/**
 * Handle window resize
 */
function handleWindowResize() {
    // Recalculate chart dimensions
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
    
    // Adjust table responsiveness
    adjustTableResponsiveness();
}

/**
 * Handle page visibility changes
 */
function handleVisibilityChange() {
    if (document.visibilityState === 'visible') {
        // Resume auto-refresh when page becomes visible
        if (!refreshInterval) {
            startAutoRefresh();
        }
        // Refresh data after being away
        refreshDashboardData(true);
    } else {
        // Pause auto-refresh when page is hidden
        stopAutoRefresh();
    }
}

/**
 * Adjust table responsiveness
 */
function adjustTableResponsiveness() {
    const tables = document.querySelectorAll('.table-responsive table');
    
    tables.forEach(table => {
        const container = table.closest('.table-responsive');
        if (container.scrollWidth > container.clientWidth) {
            table.classList.add('table-sm');
        } else {
            table.classList.remove('table-sm');
        }
    });
}

/**
 * Open quick trade modal
 */
function openQuickTradeModal() {
    const modal = document.getElementById('quickTradeModal');
    if (modal) {
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    } else {
        // If modal doesn't exist, redirect to portfolio page
        window.location.href = '/portfolio';
    }
}

/**
 * Generate trading signal
 */
async function generateTradingSignal() {
    try {
        showLoadingMessage('Generating signal...');
        
        // In real implementation, this would call the API
        const symbol = 'AAPL'; // Default symbol
        const response = await fetch(`/api/generate_signal/${symbol}`);
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                showSuccessMessage(`Signal generated: ${data.signal.type.toUpperCase()} ${symbol}`);
                refreshDashboardData(true);
            } else {
                showErrorMessage('Failed to generate signal: ' + data.message);
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
        
    } catch (error) {
        console.error('Error generating signal:', error);
        showErrorMessage('Failed to generate signal. Please try again.');
    } finally {
        hideLoadingMessage();
    }
}

/**
 * Close all open modals
 */
function closeAllModals() {
    const modals = document.querySelectorAll('.modal.show');
    modals.forEach(modal => {
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal) {
            bsModal.hide();
        }
    });
}

/**
 * Show loading state
 */
function showLoadingState() {
    const loadingElements = document.querySelectorAll('[data-loading]');
    loadingElements.forEach(element => {
        element.style.display = 'block';
    });
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const loadingElements = document.querySelectorAll('[data-loading]');
    loadingElements.forEach(element => {
        element.style.display = 'none';
    });
}

/**
 * Show refresh indicator
 */
function showRefreshIndicator() {
    const indicators = document.querySelectorAll('[data-refresh-indicator]');
    indicators.forEach(indicator => {
        indicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    });
}

/**
 * Hide refresh indicator
 */
function hideRefreshIndicator() {
    const indicators = document.querySelectorAll('[data-refresh-indicator]');
    indicators.forEach(indicator => {
        indicator.innerHTML = '<i class="fas fa-sync"></i>';
    });
}

/**
 * Update last refresh time
 */
function updateLastRefreshTime() {
    const timeElements = document.querySelectorAll('[data-last-refresh]');
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    timeElements.forEach(element => {
        element.textContent = timeString;
    });
}

/**
 * Show success message
 */
function showSuccessMessage(message) {
    showToast(message, 'success');
}

/**
 * Show error message
 */
function showErrorMessage(message) {
    showToast(message, 'danger');
}

/**
 * Show loading message
 */
function showLoadingMessage(message) {
    showToast(message, 'info', 0); // No auto-hide
}

/**
 * Hide loading message
 */
function hideLoadingMessage() {
    hideToast();
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', autoHide = 3000) {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast-notification');
    existingToasts.forEach(toast => toast.remove());
    
    const toastHTML = `
        <div class="toast-notification alert alert-${type} alert-dismissible position-fixed" 
             style="top: 80px; right: 20px; z-index: 1060; min-width: 300px;">
            <i class="fas fa-${getToastIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', toastHTML);
    
    // Auto-hide if specified
    if (autoHide > 0) {
        setTimeout(() => {
            hideToast();
        }, autoHide);
    }
}

/**
 * Hide toast notification
 */
function hideToast() {
    const toasts = document.querySelectorAll('.toast-notification');
    toasts.forEach(toast => {
        toast.classList.add('fade');
        setTimeout(() => toast.remove(), 150);
    });
}

/**
 * Get toast icon based on type
 */
function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        danger: 'exclamation-triangle',
        warning: 'exclamation-circle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Get status color class
 */
function getStatusColor(status) {
    const colors = {
        executed: 'success',
        pending: 'warning',
        cancelled: 'secondary',
        failed: 'danger'
    };
    return colors[status] || 'secondary';
}

/**
 * Format currency values
 */
function formatCurrency(value) {
    if (value == null || isNaN(value)) return '$0.00';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format percentage values
 */
function formatPercentage(value) {
    if (value == null || isNaN(value)) return '0.00%';
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format number values
 */
function formatNumber(value, decimals = 2) {
    if (value == null || isNaN(value)) return '0';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

/**
 * Format date and time
 */
function formatDateTime(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Format time only
 */
function formatTime(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Debounce function to limit function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Cleanup when page unloads
 */
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
    
    // Cleanup chart instances
    Object.values(chartInstances).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
});

// Export functions for use in other scripts
window.DashboardJS = {
    refreshData: refreshDashboardData,
    generateSignal: generateTradingSignal,
    showMessage: showSuccessMessage,
    showError: showErrorMessage,
    formatCurrency,
    formatPercentage,
    formatNumber
};
