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

    const bicubiced = await Promise.all(channels.map(channel => bicubic(channel, height, width)));

    return new cv.Mat(bicubiced);
}

/**
 * 双三次差值图像缩放
 * @param input_img 输入图像BGR格式
 * @param height
 * @param width
 * @return {Promise<void>}
 */
async function bicubic(input_img, height, width) {
    const output = new cv.Mat(height, width, cv.CV_8UC1);
    const { rows: input_rows, cols: input_cols } = input_img;
    const { rows, cols } = output;
    let count = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const r_need_computing = i * (input_rows / rows);
            const c_need_computing = j * (input_cols / cols);

            const floorX = Math.floor(r_need_computing);
            const floorY = Math.floor(c_need_computing);


            if (0 < floorX && floorX < input_rows - 2 && 0 < floorY && floorY < input_cols - 2) {
                // 中间内容的点
                count++;
                const region = input_img.getRegion(new cv.Rect(floorY - 1, floorX - 1, 4, 4));
                const value = getValueOfRegion(region, r_need_computing - floorX + 1, c_need_computing - floorY + 1);
                output.set(i, j, value);
            } else if (0 <= floorX && floorX < input_rows - 3 && 0 <= floorY && floorY < input_cols - 3) {
                // 左边、左上、上边的点
                count++;
                const region = input_img.getRegion(new cv.Rect(floorY, floorX, 4, 4));
                const value = getValueOfRegion(region, r_need_computing - floorX, c_need_computing - floorY);
                output.set(i, j, value);
            } else if (2 < floorX && floorX < input_rows && 2 < floorY && floorY < input_cols) {
                // 右边、右下、下边的点
                count++;
                const region = input_img.getRegion(new cv.Rect(floorY - 3, floorX - 3, 4, 4));
                const value = getValueOfRegion(region, r_need_computing - floorX + 3, c_need_computing - floorY + 3);
                output.set(i, j, value);
            } else if (0 <= floorY && floorY <= 2) {
                // 左下的点
                count++;
                const region = input_img.getRegion(new cv.Rect(0, input_rows - 4, 4, 4));
                const value = getValueOfRegion(region, r_need_computing - input_rows + 4, c_need_computing);
                output.set(i, j, value);
            } else if (input_cols - 3 <= floorY && floorY < input_cols) {
                // 右上的点
                count++;
                const region = input_img.getRegion(new cv.Rect(input_cols - 4, 0, 4, 4));
                const value = getValueOfRegion(region, r_need_computing, c_need_computing - input_cols + 4);
                output.set(i, j, value);
            }


        }
    }

    console.log('count', count, '/', rows * cols);
    return output;
}

/**
 * 权重函数
 * @param x
 * @param a
 * @return {number}
 */
function W(x, a = -0.5) {
    const absX = Math.abs(x);
    let result = 0;

    if (2 < absX) {
        result = 0;
    } else {
        const absX2 = Math.pow(absX, 2);
        const absX3 = Math.pow(absX, 3);
        if (0 <= absX && absX <= 1) {
            result = (a + 2) * absX3 - (a + 3) * absX2 + 1;
        } else if (1 < absX && absX <= 2) {
            result = a * absX3 - 5 * a * absX2 + 8 * a * absX - 4 * a;
        }
    }
    return result;
}

/**
 * 获取一个邻近点区域（4*4）的插值
 * @param region 邻近点趋于
 * @param x 有小数的行
 * @param y 有小数的列
 * @param a
 * @return {number} 插值
 */
function getValueOfRegion(region, x, y, a = -0.5) {
    const { rows, cols } = region;
    let sum = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            sum += region.at(i, j) * W(x - i, a) * W(y - j, a);
        }
    }
    return sum;
}

module.exports = {
    bicubic,
    bicubicBGR
};