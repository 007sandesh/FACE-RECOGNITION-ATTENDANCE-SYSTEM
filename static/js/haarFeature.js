class Stump {
    constructor(feature, threshold, error, polarity, amountOfSay) {
        this.feature = feature;
        this.threshold = threshold;
        this.error = error;
        this.polarity = polarity;
        this.amountOfSay = amountOfSay;
    }
}

const FeatureType = {
    EdgeHorizontal: 0,
    EdgeVertical: 1,
    ThreeHorizontal: 2,
    ThreeVertical: 3,
    FourBottomToTop: 4,
    FourTopToBottom: 5, // I ended up not using this one
};

class HaarFeature {
    constructor(type, width, height, posX, posY) {
        this.type = type;
        this.width = width;
        this.height = height;
        this.posX = posX;
        this.posY = posY;
    }

    applyFeature(integralImage, squaredIntegralImage, posX, posY, scaleFactor) {
        const combinedX = ~~(this.posX * scaleFactor) + posX;
        const combinedY = ~~(this.posY * scaleFactor) + posY;
        const scaledWidth = Math.round(this.width * scaleFactor);
        const scaledHeight = Math.round(this.height * scaleFactor);
        const sum = this.getRegionSum(integralImage, combinedX, combinedY, scaledWidth, scaledHeight);
        const squaredSum = this.getRegionSum(squaredIntegralImage, combinedX, combinedY, scaledWidth, scaledHeight);

        const area = scaledWidth * scaledHeight;
        const mean = sum / area;
        const variance = (squaredSum / area) - (mean * mean);
        const stdDev = Math.sqrt(Math.max(variance, 1e-6));

        const {
            white,
            black
        } = this.calculateRegions(integralImage, combinedX, combinedY, scaledWidth, scaledHeight);
        const rawFeatureValue = black - white;

        return variance > 0 ? rawFeatureValue / stdDev : rawFeatureValue;
    }

    calculateRegions(integralImage, combinedX, combinedY, scaledWidth, scaledHeight) {
        const totalSum = this.getRegionSum(integralImage, combinedX, combinedY, scaledWidth, scaledHeight);

        let blackSum;
        switch (this.type) {
            case FeatureType.EdgeHorizontal:
                blackSum = this.getRegionSum(integralImage, combinedX + Math.floor(scaledWidth / 2), combinedY, Math.floor(scaledWidth / 2), scaledHeight);
                break;
            case FeatureType.EdgeVertical:
                blackSum = this.getRegionSum(integralImage, combinedX, combinedY + Math.floor(scaledHeight / 2), scaledWidth, Math.floor(scaledHeight / 2));
                break;
            case FeatureType.ThreeHorizontal:
                blackSum = this.getRegionSum(integralImage, combinedX + Math.floor(scaledWidth / 3), combinedY, Math.floor(scaledWidth / 3), scaledHeight);
                break;
            case FeatureType.ThreeVertical:
                blackSum = this.getRegionSum(integralImage, combinedX, combinedY + Math.floor(scaledHeight / 3), scaledWidth, Math.floor(scaledHeight / 3));
                break;
            case FeatureType.FourBottomToTop:
                blackSum = this.getRegionSum(integralImage, combinedX + Math.floor(scaledWidth / 2), combinedY, Math.floor(scaledWidth / 2), Math.floor(scaledHeight / 2)) +
                    this.getRegionSum(integralImage, combinedX, combinedY + Math.floor(scaledHeight / 2), Math.floor(scaledWidth / 2), Math.floor(scaledHeight / 2));
                break;
            case FeatureType.FourTopToBottom:
                blackSum = this.getRegionSum(integralImage, combinedX, combinedY, Math.floor(scaledWidth / 2), Math.floor(scaledHeight / 2)) +
                    this.getRegionSum(integralImage, combinedX + Math.floor(scaledWidth / 2), combinedY + Math.floor(scaledHeight / 2), Math.floor(scaledWidth / 2), Math.floor(scaledHeight / 2));
                break;
            default:
                blackSum = 0;
        }

        const whiteSum = totalSum - blackSum;

        // Adjusts weight for three-rectangle features, felt intuitive, but I think this is not needed
        if (this.type == FeatureType.ThreeHorizontal || this.type == FeatureType.ThreeVertical) {
            blackSum *= 2;
        }

        return {
            white: whiteSum,
            black: blackSum
        };
    }

    getRegionSum(matrix, x, y, w, h) {
        return matrix[y + h][x + w] -
            (x > 0 ? matrix[y + h][x] : 0) -
            (y > 0 ? matrix[y][x + w] : 0) +
            (x > 0 && y > 0 ? matrix[y][x] : 0);
    }
}