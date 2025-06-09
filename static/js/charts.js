/**
 * Walford Capitals Trading Bot - Charts JavaScript
 * Financial chart implementations using Chart.js
 */

// Chart configuration defaults
Chart.defaults.color = '#adb5bd';
Chart.defaults.borderColor = '#404449';
Chart.defaults.backgroundColor = 'rgba(13, 110, 253, 0.1)';

// Global chart instances
window.chartInstances = window.chartInstances || {};

/**
 * Initialize all charts on the page
 */
function initializeCharts() {
    console.log('Initializing financial charts...');
    
    // Portfolio performance chart
    initializePortfolioChart();
    
    // Asset allocation chart
    initializeAllocationChart();
    
    // Performance comparison chart
    initializeComparisonChart();
    
    // Trading volume chart
    initializeVolumeChart();
    
    // P&L distribution chart
    initializePnLChart();
    
    console.log('Charts initialized successfully');
}

/**
 * Portfolio Performance Line Chart
 */
function initializePortfolioChart() {
    const canvas = document.getElementById('portfolioChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.chartInstances.portfolioChart) {
        window.chartInstances.portfolioChart.destroy();
    }
    
    window.chartInstances.portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointBackgroundColor: '#28a745',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }, {
                label: 'Benchmark (S&P 500)',
                data: [],
                borderColor: '#6c757d',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(33, 37, 41, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#404449',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = context.parsed.y;
                            return `${label}: ${formatCurrency(value)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM dd'
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            }
        }
    });
    
    // Load initial data
    loadPortfolioChartData();
}

/**
 * Asset Allocation Doughnut Chart
 */
function initializeAllocationChart() {
    const canvas = document.getElementById('allocationChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (window.chartInstances.allocationChart) {
        window.chartInstances.allocationChart.destroy();
    }
    
    window.chartInstances.allocationChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#0d6efd', '#198754', '#dc3545', '#ffc107',
                    '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'
                ],
                borderColor: '#2d3135',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        generateLabels: function(chart) {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, index) => {
                                    const value = data.datasets[0].data[index];
                                    const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    
                                    return {
                                        text: `${label} (${percentage}%)`,
                                        fillStyle: data.datasets[0].backgroundColor[index],
                                        strokeStyle: data.datasets[0].borderColor,
                                        lineWidth: data.datasets[0].borderWidth,
                                        hidden: false,
                                        index: index
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(33, 37, 41, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#404449',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${formatCurrency(value)} (${percentage}%)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 1000
            }
        }
    });
    
    loadAllocationChartData();
}

/**
 * Performance Comparison Bar Chart
 */
function initializeComparisonChart() {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (window.chartInstances.comparisonChart) {
        window.chartInstances.comparisonChart.destroy();
    }
    
    window.chartInstances.comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['1D', '1W', '1M', '3M', '6M', '1Y'],
            datasets: [{
                label: 'Portfolio',
                data: [],
                backgroundColor: '#0d6efd',
                borderColor: '#0d6efd',
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false
            }, {
                label: 'S&P 500',
                data: [],
                backgroundColor: '#6c757d',
                borderColor: '#6c757d',
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(33, 37, 41, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#404449',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = context.parsed.y;
                            return `${label}: ${formatPercentage(value / 100)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
    
    loadComparisonChartData();
}

/**
 * Trading Volume Chart
 */
function initializeVolumeChart() {
    const canvas = document.getElementById('volumeChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (window.chartInstances.volumeChart) {
        window.chartInstances.volumeChart.destroy();
    }
    
    window.chartInstances.volumeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Buy Volume',
                data: [],
                backgroundColor: 'rgba(40, 167, 69, 0.7)',
                borderColor: '#28a745',
                borderWidth: 1
            }, {
                label: 'Sell Volume',
                data: [],
                backgroundColor: 'rgba(220, 53, 69, 0.7)',
                borderColor: '#dc3545',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(33, 37, 41, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#404449',
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                }
            },
            animation: {
                duration: 800,
                easing: 'easeInOutQuart'
            }
        }
    });
    
    loadVolumeChartData();
}

/**
 * P&L Distribution Chart
 */
function initializePnLChart() {
    const canvas = document.getElementById('pnlChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (window.chartInstances.pnlChart) {
        window.chartInstances.pnlChart.destroy();
    }
    
    window.chartInstances.pnlChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Winning Trades',
                data: [],
                backgroundColor: 'rgba(40, 167, 69, 0.7)',
                borderColor: '#28a745',
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Losing Trades',
                data: [],
                backgroundColor: 'rgba(220, 53, 69, 0.7)',
                borderColor: '#dc3545',
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(33, 37, 41, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#404449',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        title: function(context) {
                            return 'Trade #' + (context[0].dataIndex + 1);
                        },
                        label: function(context) {
                            const x = context.parsed.x;
                            const y = context.parsed.y;
                            return `Duration: ${x} days, P&L: ${formatCurrency(y)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Trade Duration (days)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'P&L ($)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutBounce'
            }
        }
    });
    
    loadPnLChartData();
}

/**
 * Load portfolio chart data
 */
async function loadPortfolioChartData() {
    const chart = window.chartInstances.portfolioChart;
    if (!chart) return;
    
    try {
        const response = await fetch('/api/portfolio_performance?period=30d');
        const data = await response.json();
        
        if (data.performance && data.performance.length > 0) {
            const labels = data.performance.map(point => new Date(point.date));
            const portfolioValues = data.performance.map(point => point.value);
            
            // Generate benchmark data (simulated S&P 500)
            const benchmarkValues = generateBenchmarkData(portfolioValues);
            
            chart.data.labels = labels;
            chart.data.datasets[0].data = portfolioValues;
            chart.data.datasets[1].data = benchmarkValues;
            chart.update('active');
        }
    } catch (error) {
        console.error('Error loading portfolio chart data:', error);
        // Load sample data as fallback
        loadSamplePortfolioData(chart);
    }
}

/**
 * Load allocation chart data
 */
async function loadAllocationChartData() {
    const chart = window.chartInstances.allocationChart;
    if (!chart) return;
    
    try {
        // In real implementation, this would fetch actual position data
        const sampleData = {
            labels: ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy'],
            values: [45000, 25000, 15000, 10000, 5000]
        };
        
        chart.data.labels = sampleData.labels;
        chart.data.datasets[0].data = sampleData.values;
        chart.update('active');
        
    } catch (error) {
        console.error('Error loading allocation chart data:', error);
    }
}

/**
 * Load comparison chart data
 */
async function loadComparisonChartData() {
    const chart = window.chartInstances.comparisonChart;
    if (!chart) return;
    
    try {
        // Sample performance comparison data
        const portfolioReturns = [0.5, 2.1, 5.3, 12.7, 18.9, 24.5];
        const benchmarkReturns = [0.3, 1.8, 4.2, 9.1, 15.2, 19.8];
        
        chart.data.datasets[0].data = portfolioReturns;
        chart.data.datasets[1].data = benchmarkReturns;
        chart.update('active');
        
    } catch (error) {
        console.error('Error loading comparison chart data:', error);
    }
}

/**
 * Load volume chart data
 */
async function loadVolumeChartData() {
    const chart = window.chartInstances.volumeChart;
    if (!chart) return;
    
    try {
        // Sample volume data
        const dates = generateDateLabels(7);
        const buyVolumes = [15000, 22000, 18000, 25000, 19000, 21000, 17000];
        const sellVolumes = [12000, 18000, 20000, 16000, 22000, 14000, 19000];
        
        chart.data.labels = dates;
        chart.data.datasets[0].data = buyVolumes;
        chart.data.datasets[1].data = sellVolumes;
        chart.update('active');
        
    } catch (error) {
        console.error('Error loading volume chart data:', error);
    }
}

/**
 * Load P&L chart data
 */
async function loadPnLChartData() {
    const chart = window.chartInstances.pnlChart;
    if (!chart) return;
    
    try {
        // Sample P&L data
        const winningTrades = [
            {x: 2, y: 150}, {x: 5, y: 300}, {x: 1, y: 75},
            {x: 8, y: 450}, {x: 3, y: 200}, {x: 6, y: 350}
        ];
        
        const losingTrades = [
            {x: 1, y: -80}, {x: 4, y: -200}, {x: 2, y: -120},
            {x: 7, y: -180}, {x: 3, y: -150}
        ];
        
        chart.data.datasets[0].data = winningTrades;
        chart.data.datasets[1].data = losingTrades;
        chart.update('active');
        
    } catch (error) {
        console.error('Error loading P&L chart data:', error);
    }
}

/**
 * Load sample portfolio data as fallback
 */
function loadSamplePortfolioData(chart) {
    const now = new Date();
    const labels = [];
    const values = [];
    let currentValue = 100000;
    
    for (let i = 29; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        labels.push(date);
        
        // Simulate portfolio growth with some volatility
        const change = (Math.random() - 0.48) * 0.02; // Slight upward bias
        currentValue *= (1 + change);
        values.push(currentValue);
    }
    
    const benchmarkValues = generateBenchmarkData(values);
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[1].data = benchmarkValues;
    chart.update('active');
}

/**
 * Generate benchmark data based on portfolio values
 */
function generateBenchmarkData(portfolioValues) {
    if (!portfolioValues.length) return [];
    
    const initialValue = portfolioValues[0];
    return portfolioValues.map((value, index) => {
        // Simulate S&P 500 with lower volatility and steady growth
        const portfolioReturn = (value - initialValue) / initialValue;
        const benchmarkReturn = portfolioReturn * 0.8 + (Math.random() - 0.5) * 0.01;
        return initialValue * (1 + benchmarkReturn);
    });
}

/**
 * Generate date labels for charts
 */
function generateDateLabels(days) {
    const labels = [];
    const now = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    
    return labels;
}

/**
 * Update chart data with real-time information
 */
function updateChartData(chartId, newData) {
    const chart = window.chartInstances[chartId];
    if (!chart) return;
    
    try {
        if (newData.labels) {
            chart.data.labels = newData.labels;
        }
        
        if (newData.datasets) {
            newData.datasets.forEach((dataset, index) => {
                if (chart.data.datasets[index]) {
                    chart.data.datasets[index].data = dataset.data;
                }
            });
        }
        
        chart.update('active');
    } catch (error) {
        console.error(`Error updating chart ${chartId}:`, error);
    }
}

/**
 * Resize all charts
 */
function resizeCharts() {
    Object.values(window.chartInstances).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
}

/**
 * Destroy all charts
 */
function destroyCharts() {
    Object.keys(window.chartInstances).forEach(key => {
        const chart = window.chartInstances[key];
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
        delete window.chartInstances[key];
    });
}

/**
 * Export chart as image
 */
function exportChart(chartId, filename = 'chart.png') {
    const chart = window.chartInstances[chartId];
    if (!chart) return;
    
    try {
        const url = chart.toBase64Image();
        const link = document.createElement('a');
        link.download = filename;
        link.href = url;
        link.click();
    } catch (error) {
        console.error(`Error exporting chart ${chartId}:`, error);
    }
}

/**
 * Format currency for chart displays
 */
function formatCurrency(value) {
    if (value == null || isNaN(value)) return '$0.00';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format percentage for chart displays
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
 * Format number for chart displays
 */
function formatNumber(value, decimals = 0) {
    if (value == null || isNaN(value)) return '0';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Delay initialization to ensure Chart.js is loaded
    setTimeout(initializeCharts, 100);
});

// Handle window resize
window.addEventListener('resize', debounce(resizeCharts, 300));

// Cleanup on page unload
window.addEventListener('beforeunload', destroyCharts);

// Debounce function
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

// Export functions for external use
window.ChartsJS = {
    initialize: initializeCharts,
    update: updateChartData,
    resize: resizeCharts,
    destroy: destroyCharts,
    export: exportChart,
    instances: window.chartInstances
};
