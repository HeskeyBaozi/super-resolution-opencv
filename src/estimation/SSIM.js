/**
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const { getYChannelMatAsync } = require('../utils/helpers');

/**
 * 结构相似性指标
 * @param input_img1 cv.Mat 第一个图像
 * @param input_img2 cv.Mat 第二个图像
 * @return Promise<number> 两幅图的结构相似性指标
 */
async function SSIM(input_img1, input_img2) {

    // 将BGR通道转换到Y通道
    const [X, Y] = [
        await getYChannelMatAsync(input_img1),
        await getYChannelMatAsync(input_img2)
    ];

    const meanX = mean(X); // X均值
    const meanY = mean(Y); // Y均值
    const varX = variance(X, meanX); // X方差
    const varY = variance(Y, meanY); // Y方差
    const covXY = covariance(X, meanX, Y, meanY); // X和Y的协方差
    const k1 = 0.01;
    const k2 = 0.03;
    const L = 256 - 1;
    const c1 = Math.pow(k1 * L, 2); // 维持稳定的常数C1
    const c2 = Math.pow(k2 * L, 2); // 维持稳定的常数C2

    const top1 = 2 * meanX * meanY + c1;
    const top2 = 2 * covXY + c2;
    const bottom1 = Math.pow(meanX, 2) + Math.pow(meanY, 2) + c1;
    const bottom2 = varX + varY + c2;

    return (top1 * top2) / (bottom1 * bottom2);
}

/**
 * 获取一个图像的平均值
 * @param mat
 * @return {number}
 */
function mean(mat) {
    const { rows, cols } = mat;
    let sum = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sum += mat.at(i, j);
        }
    }
    return sum / (rows * cols);
}

/**
 * 计算一个图像的方差
 * @param mat 图像
 * @param meanOfMat 图像均值
 * @return {number} 方差
 */
function variance(mat, meanOfMat) {
    const { rows, cols } = mat;
    let sum = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sum += Math.pow((mat.at(i, j) - meanOfMat), 2);
        }
    }
    return sum / (rows * cols);
}

/**
 * 计算两幅图像的协方差
 * @param X
 * @param meanX
 * @param Y
 * @param meanY
 * @return {number}
 */
function covariance(X, meanX, Y, meanY) {
    const { rows, cols } = X;
    let sum = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sum += (X.at(i, j) - meanX) * (Y.at(i, j) - meanY);
        }
    }
    return sum / (rows * cols);
}

module.exports = {
    SSIM
};