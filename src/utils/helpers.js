/**
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const cv = require('opencv4nodejs');

/**
 * 异步获取图像的Y通道图像
 * @param bgrMat cv.Mat BGR彩色空间的图像
 * @return {Promise<cv.Mat>} 由Y通道产生的图像
 */
async function getYChannelMatAsync(bgrMat) {

    // 若只有一个颜色通道，直接返回
    if (bgrMat.channels === 1) return bgrMat;

    // 否则转为Y通道
    const YCbCrMat = await bgrMat.cvtColorAsync(cv.COLOR_BGR2YCrCb);
    const channels = await YCbCrMat.splitChannels();
    const Y = channels[0];
    return Y;
}

module.exports = {
    getYChannelMatAsync
};