/**
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const { getYChannelMatAsync } = require('./helpers');
const cv = require('opencv4nodejs');

/**
 * 将BGR通道的彩色图像双三次插值
 * @param input_img 输入的BGR图像
 * @param height
 * @param width
 * @return {Promise<cv.Mat>}
 */
async function bicubicBGR(input_img, height, width) {
    const channels = await input_img.splitChannelsAsync();

    const bicubiced = await Promise.all(channels.map(channel => scale(channel, height, width)));

    return new cv.Mat(bicubiced);
}


/**
 * 双三次差值图像缩放
 * @param input
 * @param newH
 * @param newW
 * @return {Promise<cv.Mat>}
 */
async function scale(input, newH, newW) {
    const output = new cv.Mat(newH, newW, cv.CV_8UC1, 0);
    for (let i = 0; i < output.rows; i++) {
        for (let j = 0; j < output.cols; j++) {
            const srcRow = i / (output.rows / input.rows);
            const srcCol = j / (output.cols / input.cols);

            const floorRow = Math.floor(srcRow);
            const floorCol = Math.floor(srcCol);

            const v = srcRow - floorRow;
            const u = srcCol - floorCol;

            let sum = 0;
            for (let m = -1; m <= 2; m++) {
                for (let n = -1; n <= 2; n++) {
                    let inputValue = 0;
                    if (0 <= floorRow + m && floorRow + m < input.rows && 0 <= floorCol + n && floorCol + n < input.cols) {
                        inputValue = input.at(floorRow + m, floorCol + n);
                    }
                    sum += inputValue * W(m - v) * W(n - u);
                }
            }

            sum = Math.min(255, sum);
            sum = Math.max(0, sum);

            output.set(i, j, sum);
        }
    }
    return output;
}

/**
 * 权重函数
 * @param x
 * @return {number}
 */
function W(x) {
    const absX = Math.abs(x);
    let result = 0;

    if (2 < absX) {
        result = 0;
    } else {
        const absX2 = Math.pow(absX, 2);
        const absX3 = Math.pow(absX, 3);
        if (0 <= absX && absX <= 1) {
            result = 1.5 * absX3 - 2.5 * absX2 + 1;
        } else if (1 < absX && absX <= 2) {
            result = -0.5 * absX3 + 2.5 * absX2 - 4 * absX + 2;
        }
    }
    return result;
}


module.exports = {
    scale,
    bicubicBGR
};