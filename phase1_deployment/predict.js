// Add event listeners to update slider display values in real time
document.getElementById('average_playtime').addEventListener('input', function() {
    document.getElementById('playtime_val').innerText = this.value + ' min';
});

document.getElementById('achievements').addEventListener('input', function() {
    document.getElementById('achievements_val').innerText = this.value;
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

    // Get input values
    const playtime = parseFloat(document.getElementById('average_playtime').value);
    const achievements = parseFloat(document.getElementById('achievements').value);
    const isMultiplayer = document.getElementById('is_multiplayer').checked ? 1 : 0;
    const isIndie = document.getElementById('is_indie').checked ? 1 : 0;
    const isAction = document.getElementById('is_action').checked ? 1 : 0;

    // The order of feature_names is ["average_playtime", "achievements", "is_multiplayer", "is_indie", "is_action"]
    // Let's create an input array mapped to the feature names
    const inputDict = {
        "average_playtime": playtime,
        "achievements": achievements,
        "is_multiplayer": isMultiplayer,
        "is_indie": isIndie,
        "is_action": isAction
    };

    // Evaluate the Linear Regression prediction formula
    let predictedPrice = modelIntercept;
    for (let i = 0; i < modelCoefs.length; i++) {
        let featureName = modelFeatureNames[i];
        let featureValue = inputDict[featureName];
        predictedPrice += modelCoefs[i] * featureValue;
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
