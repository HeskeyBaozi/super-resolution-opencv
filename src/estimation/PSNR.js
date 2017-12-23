/**
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const { getYChannelMatAsync } = require('../utils/helpers');

/**
 * 峰值信噪比
 * @param input_img1 cv.Mat 第一个图像
 * @param input_img2 cv.Mat 第二个图像
 * @return Promise<number> 两幅图的峰值信噪比
 */
async function PSNR(input_img1, input_img2) {

    // 将BGR通道转换到Y通道
    const [X, Y] = [
        await getYChannelMatAsync(input_img1),
        await getYChannelMatAsync(input_img2)
    ];

    // 计算均方差
    const mse = await MSE(X, Y);

    // 灰度级别 - 1
    const MAXI = 255;

    return 20 * Math.log10(MAXI / (Math.sqrt(mse)));
}

/**
 * 计算两幅图片矩阵之间的均方差
 * @param X 原图 X
 * @param Y 噪声近似图像 Y
 * @return {Promise<number>} 均方差
 */
async function MSE(X, Y) {
    const { rows, cols } = X;
    let sum = 0;

    // 遍历整个图像
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sum += Math.pow(X.at(i, j) - Y.at(i, j), 2);
        }
    }
    return sum / (rows * cols);
}


module.exports = {
    PSNR, MSE
};