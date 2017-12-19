const cv = require('opencv4nodejs');
const path = require('path');

const { PSNR } = require('./PSNR');
const { SSIM } = require('./SSIM');

async function run() {
    const mat = await cv.imreadAsync(path.resolve(__dirname, './set14/monarch.bmp'));
    const b = (await mat.splitChannelsAsync());
    const psnr = await SSIM(mat, mat);
    console.log(psnr);
    cv.imshowWait('b', b[0]);
}

run();