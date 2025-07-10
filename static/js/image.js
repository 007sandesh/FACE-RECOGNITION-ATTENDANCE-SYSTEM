function processImageData(pixels, width, height) {
    const integralMatrix = Array(height + 1).fill(0).map(() => Array(width + 1).fill(0));
    const squaredIntegralMatrix = Array(height + 1).fill(0).map(() => Array(width + 1).fill(0));

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const grayValue = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
            const normalizedValue = grayValue / 255;

            const above = integralMatrix[y][x + 1];
            const left = integralMatrix[y + 1][x];
            const aboveLeft = integralMatrix[y][x];

            const aboveSq = squaredIntegralMatrix[y][x + 1];
            const leftSq = squaredIntegralMatrix[y + 1][x];
            const aboveLeftSq = squaredIntegralMatrix[y][x];

            integralMatrix[y + 1][x + 1] = normalizedValue + above + left - aboveLeft;
            squaredIntegralMatrix[y + 1][x + 1] = (normalizedValue * normalizedValue) + aboveSq + leftSq - aboveLeftSq;
        }
    }

    return {
        integralMatrix,
        squaredIntegralMatrix
    };
}

// User-specific face detection data
let userFaceData = {
    trainingMode: false,
    userSpecificThresholds: {},
    userFaceFeatures: {},
    adaptiveThresholds: true,
    confidenceMultipliers: {}
};

function findFaces(integralImage, squaredIntegralImage, width, height, stepSize = 2, maxScale = 3, scaleStep = 1) {
    let rawDetections = [];
    
    // Improved cascade stages for better rejection of false positives
    const stages = [
        { index: 10, threshold: -0.5 },
        { index: 50, threshold: -0.2 },
        { index: 100, threshold: 0 },
        { index: 200, threshold: 0.1 },
        { index: 500, threshold: 0.2 }
    ];

    const detectFace = (integralImage, squaredIntegralImage, posX, posY, scaleFactor) => {
        let sumAlphaH = 0;
        let stageIndex = 0;
        let featureResponses = [];

        for (let i = 0; i < stumps.length; i++) {
            let stump = stumps[i];
            let response = stump.feature.applyFeature(integralImage, squaredIntegralImage, posX, posY, scaleFactor);
            featureResponses.push(response);

            let scaledThreshold = stump.threshold * scaleFactor * scaleFactor;
            
            // Apply user-specific threshold adjustments if in training mode or if we have learned data
            if (userFaceData.adaptiveThresholds && userFaceData.userSpecificThresholds[i]) {
                scaledThreshold = userFaceData.userSpecificThresholds[i] * scaleFactor * scaleFactor;
            }
            
            let h = (stump.polarity * response <= stump.polarity * scaledThreshold) ? 1 : -1;
            
            // Apply confidence multiplier if available
            let weight = stump.amountOfSay;
            if (userFaceData.confidenceMultipliers[i]) {
                weight *= userFaceData.confidenceMultipliers[i];
            }
            
            sumAlphaH += weight * h;

            // Check if we've reached a stage boundary
            if (stageIndex < stages.length && i + 1 >= stages[stageIndex].index) {
                // If we don't meet the stage threshold, reject this window early
                if (sumAlphaH <= stages[stageIndex].threshold) {
                    return;
                }
                stageIndex++;
            }
        }

        // Only add detections with positive confidence
        if (sumAlphaH > 0) {
            const detection = {
                x: posX,
                y: posY,
                scaleFactor: scaleFactor,
                confidency: sumAlphaH,
                featureResponses: featureResponses
            };
            
            // If in training mode, store feature responses for learning
            if (userFaceData.trainingMode) {
                storeTrainingData(detection);
            }
            
            rawDetections.push(detection);
        }
    }

    let scaleFactor = 1;
    while (scaleFactor <= maxScale) {
        const scaledWindowSize = Math.floor(WINDOW_SIZE * scaleFactor);
        const step = Math.max(1, Math.floor(stepSize * scaleFactor));
        
        for (let startY = 0; startY < height - scaledWindowSize; startY += step) {
            for (let startX = 0; startX < width - scaledWindowSize; startX += step) {
                detectFace(integralImage, squaredIntegralImage, startX, startY, scaleFactor);
            }
        }

        scaleFactor += scaleStep;
    }
    
    // Apply improved filtering to reduce false positives
    return filterStrongestFaces(rawDetections, WINDOW_SIZE * 0.6);
}

// Store training data from detected faces
function storeTrainingData(detection) {
    // We'll store feature responses to learn user-specific patterns
    if (!userFaceData.userFaceFeatures.responses) {
        userFaceData.userFaceFeatures.responses = [];
    }
    
    userFaceData.userFaceFeatures.responses.push(detection.featureResponses);
    
    // If we have enough samples, update the user-specific thresholds
    if (userFaceData.userFaceFeatures.responses.length >= 5) {
        updateUserSpecificThresholds();
    }
}

// Update user-specific thresholds based on collected training data
function updateUserSpecificThresholds() {
    if (!userFaceData.userFaceFeatures.responses || 
        userFaceData.userFaceFeatures.responses.length < 5) {
        return;
    }
    
    const responses = userFaceData.userFaceFeatures.responses;
    
    // For each feature, calculate the average response
    for (let i = 0; i < stumps.length; i++) {
        let sum = 0;
        let count = 0;
        
        for (let j = 0; j < responses.length; j++) {
            if (responses[j][i] !== undefined) {
                sum += responses[j][i];
                count++;
            }
        }
        
        if (count > 0) {
            // Adjust the threshold based on the average response
            const avgResponse = sum / count;
            
            // Store the new threshold (with some margin)
            if (stumps[i].polarity > 0) {
                userFaceData.userSpecificThresholds[i] = avgResponse * 1.1; // 10% margin
            } else {
                userFaceData.userSpecificThresholds[i] = avgResponse * 0.9; // 10% margin
            }
            
            // Adjust confidence multiplier based on consistency
            let variance = 0;
            for (let j = 0; j < responses.length; j++) {
                if (responses[j][i] !== undefined) {
                    variance += Math.pow(responses[j][i] - avgResponse, 2);
                }
            }
            
            variance = variance / count;
            const stdDev = Math.sqrt(variance);
            
            // Higher consistency (lower stdDev) means higher confidence
            const consistency = 1 / (1 + stdDev);
            userFaceData.confidenceMultipliers[i] = 1 + consistency;
        }
    }
    
    console.log("Updated user-specific thresholds based on " + responses.length + " training samples");
}

// Enable or disable training mode
function setTrainingMode(enabled) {
    userFaceData.trainingMode = enabled;
    
    if (!enabled) {
        // If disabling training mode, make sure we update thresholds
        updateUserSpecificThresholds();
    } else {
        // If enabling, reset the training data
        userFaceData.userFaceFeatures.responses = [];
    }
    
    return userFaceData.trainingMode;
}

// Reset all user-specific data
function resetUserData() {
    userFaceData.userSpecificThresholds = {};
    userFaceData.userFaceFeatures = {};
    userFaceData.confidenceMultipliers = {};
}

function filterStrongestFaces(detections, windowSize = WINDOW_SIZE / 2) {
    if (detections.length === 0) return [];
    
    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidency - a.confidency);
    
    const filtered = [];
    const used = new Array(detections.length).fill(false);
    
    // Minimum confidence threshold to further reduce false positives
    const minConfidence = 50;

    // Take the highest confidence detections first
    for (let i = 0; i < detections.length; i++) {
        if (used[i]) continue;
        
        const current = detections[i];
        
        // Skip low confidence detections
        if (current.confidency < minConfidence) continue;
        
        filtered.push(current);
        used[i] = true;
        
        // Mark overlapping detections as used
        for (let j = i + 1; j < detections.length; j++) {
            if (used[j]) continue;
            
            const other = detections[j];
            const overlapThreshold = windowSize * (current.scaleFactor + other.scaleFactor) / 2;
            
            if (Math.abs(current.x - other.x) < overlapThreshold && 
                Math.abs(current.y - other.y) < overlapThreshold) {
                used[j] = true;
            }
        }
    }

    // Limit to top 3 faces to improve performance and reduce false positives
    return filtered.slice(0, 3);
}