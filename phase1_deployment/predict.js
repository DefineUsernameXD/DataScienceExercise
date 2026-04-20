// Add event listeners to update slider display values in real time
document.getElementById('average_playtime').addEventListener('input', function() {
    document.getElementById('playtime_val').innerText = this.value + ' min';
});

document.getElementById('achievements').addEventListener('input', function() {
    document.getElementById('achievements_val').innerText = this.value;
});

document.getElementById('release_year').addEventListener('input', function() {
    document.getElementById('year_val').innerText = this.value;
});

// Model variables
let modelIntercept = 0;
let modelCoefs = [];
let modelFeatureNames = [];
let modelLoaded = false;

// Load the model.json on startup
window.onload = async function() {
    try {
        const response = await fetch('model.json');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        
        // ML.js logic: Extract the linear regression parameters
        modelIntercept = data.intercept;
        modelCoefs = data.coef;
        modelFeatureNames = data.feature_names;
        modelLoaded = true;
        
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load model.json. Ensure you are running this on a web server (e.g. GitHub Pages or local http server) rather than viewing local file.", error);
    }
};

function predictPrice() {
    if (!modelLoaded) {
        alert("The model hasn't loaded yet, or it failed to load. Please ensure you view this via a web server (http://) and that model.json exists in the same directory.");
        return;
    }

    // Get input values (Numeric)
    const playtime = parseFloat(document.getElementById('average_playtime').value);
    const achievements = parseFloat(document.getElementById('achievements').value);
    const releaseYear = parseFloat(document.getElementById('release_year').value);
    
    // Get checkboxes (Binary features mapping)
    const inputDict = {
        "average_playtime": playtime,
        "achievements": achievements,
        "release_year": releaseYear,
        "self_published": document.getElementById('self_published').checked ? 1 : 0,
        "english": document.getElementById('english').checked ? 1 : 0,
        "is_windows": document.getElementById('is_windows').checked ? 1 : 0,
        "is_mac": document.getElementById('is_mac').checked ? 1 : 0,
        "is_multiplayer": document.getElementById('is_multiplayer').checked ? 1 : 0,
        "is_indie": document.getElementById('is_indie').checked ? 1 : 0,
        "is_action": document.getElementById('is_action').checked ? 1 : 0,
        "is_early_access": document.getElementById('is_early_access').checked ? 1 : 0
    };

    // Evaluate the Linear Regression prediction formula (Native ML.js array multiplication equivalent)
    let predictedPrice = modelIntercept;
    for (let i = 0; i < modelCoefs.length; i++) {
        let featureName = modelFeatureNames[i];
        
        // Handle forced Windows inclusion
        if (featureName === "is_windows") {
            predictedPrice += modelCoefs[i] * 1;
        } else {
            let featureValue = inputDict[featureName];
            predictedPrice += modelCoefs[i] * featureValue;
        }
    }

    // Ensure price doesn't go below minimum (Free / $0.00)
    if (predictedPrice < 0) {
        predictedPrice = 0;
    }

    // Update the UI
    const priceDisplay = document.getElementById('price_result');
    const resultContainer = document.getElementById('result-container');
    
    // Format as currency
    priceDisplay.innerText = "$" + predictedPrice.toFixed(2);
    
    // Show the container
    resultContainer.classList.remove('hidden');
    
    // Tiny micro-animation for the result
    priceDisplay.style.transform = 'scale(0.9)';
    priceDisplay.style.opacity = '0';
    setTimeout(() => {
        priceDisplay.style.transition = 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        priceDisplay.style.transform = 'scale(1)';
        priceDisplay.style.opacity = '1';
    }, 50);
}
