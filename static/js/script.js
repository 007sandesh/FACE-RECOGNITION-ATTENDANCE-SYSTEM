const WINDOW_SIZE = 24;
let canvas, context, startButton, fileButton, fileInput, detectionThreshold, video, imageUploader;
let detectionCanvas, detectionContext;
let haarFeatures = [];
let stumps = [];
let feedToInputDiff;

let userDetectionThreshold = 300;
let cameraEnabled = false;
let uploadedImage = null;

// Add training mode integration
document.addEventListener("DOMContentLoaded", function() {
    // Check if training mode toggle exists
    const trainingModeToggle = document.getElementById('training-mode-toggle');
    
    if (trainingModeToggle) {
        // Initialize training mode if the function exists
        if (typeof setTrainingMode === 'function') {
            setTrainingMode(trainingModeToggle.checked);
            
            // Add event listener for toggle changes
            trainingModeToggle.addEventListener('change', function() {
                setTrainingMode(this.checked);
                console.log("Training mode " + (this.checked ? "enabled" : "disabled"));
            });
        }
    }
    
    // Add training data persistence
    window.addEventListener('beforeunload', function() {
        // If we have training data and the function exists, try to save it
        if (typeof userFaceData !== 'undefined' && 
            userFaceData.userSpecificThresholds && 
            Object.keys(userFaceData.userSpecificThresholds).length > 0) {
            
            // We could save this to localStorage for persistence
            try {
                localStorage.setItem('userFaceTrainingData', JSON.stringify(userFaceData));
                console.log("Training data saved to localStorage");
            } catch (e) {
                console.error("Failed to save training data:", e);
            }
        }
    });
    
    // Load any previously saved training data
    if (typeof userFaceData !== 'undefined') {
        try {
            const savedData = localStorage.getItem('userFaceTrainingData');
            if (savedData) {
                const parsedData = JSON.parse(savedData);
                
                // Merge saved data with current data
                if (parsedData.userSpecificThresholds) {
                    userFaceData.userSpecificThresholds = parsedData.userSpecificThresholds;
                }
                
                if (parsedData.confidenceMultipliers) {
                    userFaceData.confidenceMultipliers = parsedData.confidenceMultipliers;
                }
                
                console.log("Loaded saved training data");
            }
        } catch (e) {
            console.error("Failed to load training data:", e);
        }
    }
});

document.addEventListener("DOMContentLoaded", () => {
    stumpsJSON.forEach(stump => {
        let feature = new HaarFeature(stump.feature.Type, stump.feature.Width, stump.feature.Height, stump.feature.PosX, stump.feature.PosY);
        stumps.push(new Stump(feature, stump.threshold, stump.error, stump.polarity, stump.amountOfSay));
    });

    canvas = document.getElementById("camCanvas");
    detectionCanvas = document.getElementById("detectionCanvas");
    context = canvas.getContext("2d", {
        willReadFrequently: true
    });
    detectionContext = detectionCanvas.getContext("2d", {
        willReadFrequently: true
    });
    startButton = document.getElementById("startButton");
    fileButton = document.getElementById("fileButton");
    fileInput = document.getElementById("imageUploader");
    detectionThreshold = document.getElementById("detectionThreshold");
    video = document.createElement("video");
    imageUploader = document.getElementById('imageUploader');

    canvas.width = 1280;
    detectionCanvas.width = 240;
    canvas.height = 720;
    detectionCanvas.height = 135;
    context.imageSmoothingEnabled = true;
    detectionContext.imageSmoothingEnabled = true;
    detectionCanvas.hidden = true;

    feedToInputDiff = canvas.width / detectionCanvas.width;
    drawFrame();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera not supported");
        console.error("Camera not supported");
        return;
    }

    initControls();

    startButton.addEventListener("click", () => {
        if (cameraEnabled) {
            if (video.srcObject) {
                let tracks = video.srcObject.getTracks();
                tracks.forEach(track => track.stop());
            }
            cameraEnabled = false;
            console.log("Camera stopped.");
        } else {
            navigator.mediaDevices
                .getUserMedia({
                    video: {
                        width: {
                            ideal: 1280
                        },
                        height: {
                            ideal: 720
                        }
                    },
                    audio: false
                })
                .then((stream) => {
                    video.srcObject = stream;
                    video.play();
                    cameraEnabled = true;
                    console.log("Camera started.");
                })
                .catch((err) => {
                    alert("Cam error: " + err.message);
                    console.error("error:", err);
                });
        }
    });

    fileButton.addEventListener("click", () => {
        fileInput.click();
    });

    imageUploader.addEventListener('change', (event) => {
        if (cameraEnabled) {
            console.log("Camera active, image ignored.");
            return;
        }

        const file = event.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            const img = new Image();

            img.onload = () => {
                uploadedImage = img;
                imageX = 0;
                imageY = 0;
                imageScale = 1;
                console.log("Image uploaded");
                URL.revokeObjectURL(url);
            };

            img.onerror = () => {
                console.error("Image error");
            };

            img.src = url;
        }
    });

    detectionThreshold.oninput = function() {
        userDetectionThreshold = this.value;
    }
});

function drawFrame() {
    context.clearRect(0, 0, canvas.width, canvas.height);

    if (cameraEnabled) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else if (uploadedImage) {
        drawImageInteractive(uploadedImage, context, canvas);
        context.font = "40px Arial";
        context.fillStyle = "white";
        context.strokeStyle = "black";
        context.lineWidth = 1;
        context.fillText("Drag to move", 10, 50);
        context.strokeText("Drag to move", 10, 50);
        context.fillText("Scroll to zoom", 10, 100);
        context.strokeText("Scroll to zoom", 10, 100);
    }

    runFaceDetection();
    requestAnimationFrame(drawFrame);
}

function drawImageInteractive(img) {
    const scaledWidth = img.width * imageScale;
    const scaledHeight = img.height * imageScale;

    context.drawImage(img, imageX, imageY, scaledWidth, scaledHeight);
}

function runFaceDetection() {
    detectionContext.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    detectionContext.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, detectionCanvas.width, detectionCanvas.height);

    const imageData = detectionContext.getImageData(0, 0, detectionCanvas.width, detectionCanvas.height);
    const pixels = imageData.data;

    const processedImage = processImageData(pixels, detectionCanvas.width, detectionCanvas.height);
    const integralImage = processedImage.integralMatrix;
    const squaredIntegralImage = processedImage.squaredIntegralMatrix;
    const detectedFaces = findFaces(integralImage, squaredIntegralImage, detectionCanvas.width, detectionCanvas.height, 1.5, 5, 0.25);

    detectedFaces.forEach(pos => {
        if (pos.confidency > userDetectionThreshold) {
            context.beginPath();
            context.rect(
                pos.x * feedToInputDiff,
                pos.y * feedToInputDiff,
                WINDOW_SIZE * pos.scaleFactor * feedToInputDiff,
                WINDOW_SIZE * pos.scaleFactor * feedToInputDiff
            );
            context.strokeStyle = "lime";
            context.lineWidth = 5;
            context.stroke();
        }
    });
}