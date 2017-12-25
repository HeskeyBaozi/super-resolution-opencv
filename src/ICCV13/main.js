const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const { getYChannelMatAsync } = require('../utils/helpers');


async function trainPairs(dir) {
    const fileNames = fs.readdirSync(dir);
    for (const name of fileNames) {
        console.log(`[Process] ${name}`);
        const inputMat = await cv.imreadAsync(path.resolve(dir, name));
        await trainOne(inputMat, 1.6);
    }
}

// run!
trainPairs(path.resolve(__dirname, '../../', './train'));

async function trainOne(inputMat, sigma) {
    cv.imshowWait('training', inputMat);
    const Y = await getYChannelMatAsync(inputMat);
    const { features, centers } = await GenerateFeatureFromHR(Y, sigma);
}

async function GenerateFeatureFromHR(HR_Mat, sigma) {
    const patchSize = 7;
    const patchSizeHalf = (patchSize - 1) / 2;

    const LR = await GenerateLR(HR_Mat, sigma);
    cv.imshowWait('GS, down 1/3', LR);

    const features = [];
    const centers = [];

    // r => 3, 4, ...,
    for (let r = patchSizeHalf; r < LR.rows - patchSizeHalf; r++) {
        for (let c = patchSizeHalf; c < LR.cols - patchSizeHalf; c++) {
            const patch_LR = LR.getRegion(new cv.Rect(c - patchSizeHalf, r - patchSizeHalf, patchSize, patchSize));

            const vector_LR = [];
            for (let i = 0; i < 7; i++) {
                for (let j = 0; j < 7; j++) {
                    if (i === 0 && j === 0 || i === 0 && j === 6 || i === 6 && j === 0 || i === 6 && j === 6) {
                        continue;
                    }
                    vector_LR.push(patch_LR.at(i, j));
                }
            }
            const sum = vector_LR.reduce((sum, next) => sum + next, 0);
            const mean = sum / vector_LR.length;
            const vector_mean = vector_LR.map(val => val - mean);
            features.push(vector_mean);
            centers.push({ r, c });

        }
    }

    return {
        features,
        centers
    };
}

/**
 * Il = (Ih ⊗ G) ↓s,
 * @param HR_Mat
 * @param sigma
 * @return {Promise<*>}
 */
async function GenerateLR(HR_Mat, sigma) {
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
    const blur = await HR_Mat.gaussianBlurAsync(new cv.Size(kernelSize, kernelSize), sigma);
    return await blur.resizeAsync(Math.floor(blur.rows / 3), Math.floor(blur.cols / 3));
}

