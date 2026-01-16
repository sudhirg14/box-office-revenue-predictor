// JavaScript for Movie Box Office Predictor

// Global variables
let genreChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set default values
    setDefaultValues();
    
    // Load statistics
    loadStatistics();
    
    // Setup form submission
    setupFormSubmission();
    
    // Setup smooth scrolling
    setupSmoothScrolling();
}

function setDefaultValues() {
    // Set current year as default
    const currentYear = new Date().getFullYear();
    document.getElementById('releaseYear').value = currentYear;
    
    // Set some reasonable defaults
    document.getElementById('budget').value = 50;
    document.getElementById('runtime').value = 120;
    document.getElementById('criticRating').value = 6.5;
    document.getElementById('audienceRating').value = 6.5;
    document.getElementById('reviewSentiment').value = 0.2;
    document.getElementById('reviewVolume').value = 10000;
    document.getElementById('starPower').value = 0.5;
    document.getElementById('socialBuzz').value = 50000;
    document.getElementById('marketingSpend').value = 20;
}

function setupFormSubmission() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });
}

function makePrediction() {
    // Show loading spinner
    showLoadingSpinner();
    
    // Hide previous results
    hidePredictionResult();
    
    // Get form data
    const formData = getFormData();
    
    // Validate form data
    if (!validateFormData(formData)) {
        hideLoadingSpinner();
        return;
    }
    
    // Make API request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingSpinner();
        
        if (data.error) {
            showError(data.error);
        } else {
            showPredictionResult(data);
        }
    })
    .catch(error => {
        hideLoadingSpinner();
        showError('An error occurred while making the prediction: ' + error.message);
    });
}

function getFormData() {
    return {
        genre: document.getElementById('genre').value,
        budget_million: parseFloat(document.getElementById('budget').value),
        release_year: parseInt(document.getElementById('releaseYear').value),
        runtime_min: parseInt(document.getElementById('runtime').value),
        critic_rating: parseFloat(document.getElementById('criticRating').value),
        audience_rating: parseFloat(document.getElementById('audienceRating').value),
        review_sentiment: parseFloat(document.getElementById('reviewSentiment').value),
        review_volume: parseInt(document.getElementById('reviewVolume').value),
        star_power: parseFloat(document.getElementById('starPower').value),
        social_media_buzz: parseInt(document.getElementById('socialBuzz').value),
        marketing_spend_million: parseFloat(document.getElementById('marketingSpend').value)
    };
}

function validateFormData(data) {
    // Check if all required fields are filled
    const requiredFields = ['genre', 'budget_million', 'release_year', 'runtime_min', 
                          'critic_rating', 'audience_rating', 'review_sentiment', 
                          'review_volume', 'star_power', 'social_media_buzz', 'marketing_spend_million'];
    
    for (const field of requiredFields) {
        if (data[field] === '' || data[field] === null || data[field] === undefined) {
            showError(`Please fill in all required fields. Missing: ${field}`);
            return false;
        }
    }
    
    // Validate ranges
    if (data.critic_rating < 1 || data.critic_rating > 10) {
        showError('Critic rating must be between 1 and 10');
        return false;
    }
    
    if (data.audience_rating < 1 || data.audience_rating > 10) {
        showError('Audience rating must be between 1 and 10');
        return false;
    }
    
    if (data.review_sentiment < -1 || data.review_sentiment > 1) {
        showError('Review sentiment must be between -1 and 1');
        return false;
    }
    
    if (data.star_power < 0 || data.star_power > 1) {
        showError('Star power must be between 0 and 1');
        return false;
    }
    
    return true;
}

function showPredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    const predictedValue = document.getElementById('predictedValue');
    const confidence = document.getElementById('confidence');
    const predictionTime = document.getElementById('predictionTime');
    
    // Update values
    predictedValue.textContent = `$${data.prediction}M`;
    confidence.textContent = `${data.confidence}%`;
    predictionTime.textContent = new Date(data.timestamp).toLocaleString();
    
    // Show result with animation
    resultDiv.style.display = 'block';
    resultDiv.classList.add('fade-in-up');
    
    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hidePredictionResult() {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.style.display = 'none';
    resultDiv.classList.remove('fade-in-up');
}

function showLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'block';
}

function hideLoadingSpinner() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    // Create error alert
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
    errorAlert.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert after the form
    const form = document.getElementById('predictionForm');
    form.parentNode.insertBefore(errorAlert, form.nextSibling);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (errorAlert.parentNode) {
            errorAlert.remove();
        }
    }, 5000);
}

function loadStatistics() {
    fetch('/api/stats')
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error loading statistics:', data.error);
            return;
        }
        
        updateStatisticsDisplay(data);
        createGenreChart(data.genres);
    })
    .catch(error => {
        console.error('Error loading statistics:', error);
    });
}

function updateStatisticsDisplay(data) {
    document.getElementById('totalMovies').textContent = data.total_movies.toLocaleString();
    document.getElementById('avgBoxOffice').textContent = `$${data.avg_box_office}M`;
    document.getElementById('maxBoxOffice').textContent = `$${data.max_box_office}M`;
    document.getElementById('yearRange').textContent = `${data.year_range.min}-${data.year_range.max}`;
}

function createGenreChart(genres) {
    const ctx = document.getElementById('genreChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (genreChart) {
        genreChart.destroy();
    }
    
    const genreNames = Object.keys(genres);
    const genreCounts = Object.values(genres);
    
    // Define colors for each genre
    const colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
    ];
    
    genreChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: genreNames,
            datasets: [{
                data: genreCounts,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function setupSmoothScrolling() {
    // Handle navigation clicks
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility functions
function formatNumber(num) {
    return num.toLocaleString();
}

function formatCurrency(num) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(num * 1000000);
}

// Add some interactive features
document.addEventListener('DOMContentLoaded', function() {
    // Add hover effects to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Add input validation feedback
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateInput(this);
        });
        
        input.addEventListener('input', function() {
            clearValidation(this);
        });
    });
});

function validateInput(input) {
    const value = input.value;
    const type = input.type;
    const name = input.name;
    
    // Clear previous validation
    clearValidation(input);
    
    // Validate based on input type and name
    let isValid = true;
    let message = '';
    
    if (type === 'number') {
        const numValue = parseFloat(value);
        
        if (isNaN(numValue)) {
            isValid = false;
            message = 'Please enter a valid number';
        } else {
            // Check ranges based on field name
            switch (name) {
                case 'budget_million':
                    if (numValue < 1 || numValue > 500) {
                        isValid = false;
                        message = 'Budget must be between 1 and 500 million';
                    }
                    break;
                case 'critic_rating':
                case 'audience_rating':
                    if (numValue < 1 || numValue > 10) {
                        isValid = false;
                        message = 'Rating must be between 1 and 10';
                    }
                    break;
                case 'review_sentiment':
                    if (numValue < -1 || numValue > 1) {
                        isValid = false;
                        message = 'Sentiment must be between -1 and 1';
                    }
                    break;
                case 'star_power':
                    if (numValue < 0 || numValue > 1) {
                        isValid = false;
                        message = 'Star power must be between 0 and 1';
                    }
                    break;
            }
        }
    }
    
    // Apply validation styling
    if (isValid) {
        input.classList.add('is-valid');
    } else {
        input.classList.add('is-invalid');
        
        // Show error message
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = message;
        input.parentNode.appendChild(feedback);
    }
}

function clearValidation(input) {
    input.classList.remove('is-valid', 'is-invalid');
    
    // Remove feedback messages
    const feedback = input.parentNode.querySelector('.valid-feedback, .invalid-feedback');
    if (feedback) {
        feedback.remove();
    }
}

