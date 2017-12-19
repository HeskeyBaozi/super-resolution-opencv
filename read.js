const cv = require('opencv4nodejs');
const path = require('path');

const { PSNR } = require('./PSNR');
const { SSIM } = require('./SSIM');
const { bicubic, bicubicBGR } = require('./bicubic');

async function run() {
    const mat = await cv.imreadAsync(path.resolve(__dirname, './set14/monarch.bmp'));
    const b = (await mat.splitChannelsAsync());
    const output = await bicubicBGR(mat, 800, 1000);

    cv.imshowWait('bic', output);
    //await SSIM(mat, mat);
    // console.log('mat', mat);
    // console.log('b', b);
}

run();