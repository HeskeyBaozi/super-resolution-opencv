/**
 * 用基于双三次插值的图像超分辨率算法进行实验
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const fs = require('fs');
const path = require('path');

const cv = require('opencv4nodejs');
const { bicubicBGR } = require('./bicubic/bicubic');
const { PSNR } = require('./estimation/PSNR');
const { SSIM } = require('./estimation/SSIM');

async function runEx4() {
    const fileNames = fs.readdirSync(path.resolve(__dirname, '../set14'));
    for (const name of fileNames) {
        console.log('running ', name);
        await processOnePicture(name);
    }
}

async function processOnePicture(filename) {
    const I_HR = await cv.imreadAsync(path.resolve(__dirname, '../set14/', filename));
    const { rows: input_rows, cols: input_cols } = I_HR;

    //const I_LR = await I_HR.resizeAsync(Math.floor(input_rows / 3), Math.floor(input_cols / 3));
    //const I_BI = await I_LR.resizeAsync(input_rows, input_cols);
    const I_LR = await bicubicBGR(I_HR, Math.floor(input_rows / 3), Math.floor(input_cols / 3));
    const I_BI = await bicubicBGR(I_LR, input_rows, input_cols);
    const psnr = await PSNR(I_HR, I_BI);
    const ssim = await SSIM(I_HR, I_BI);

    console.log(`[${filename}]: PSNR = ${psnr}, SSIM = ${ssim}`);
}

runEx4();