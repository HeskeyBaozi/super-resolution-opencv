const cv = require('opencv4nodejs');

async function GenerateLR(HR_raw, sigma) {
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
    const blur = await HR_raw.gaussianBlurAsync(new cv.Size(kernelSize, kernelSize), sigma);

    // resize
    const afterResize = blur;
    return afterResize;
}

module.exports = {
    GenerateLR
};