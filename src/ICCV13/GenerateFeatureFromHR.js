const { GenerateLR } = require('./GenerateLR');

async function GenerateFeatureFromHR(HR_raw, sf, sigma) {
    const patchSize = 7;
    const patchSizeHalf = (patchSize - 1) / 2;

    const LR = await GenerateLR(HR_raw, sigma);

    for (let r = patchSizeHalf + 1; r <= LR.rows - patchSizeHalf; r++) {
        for (let c = patchSizeHalf + 1; c <= LR.cols - patchSizeHalf; c++) {
            const patch_LR = 56
        }
    }
}