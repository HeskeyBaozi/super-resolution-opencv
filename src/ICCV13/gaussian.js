const math = require('mathjs');
const cv = require('opencv4nodejs');

/**
 * 高斯函数的二维形式
 * @param x
 * @param y
 * @param sigma
 * @return {number}
 */
function G(sigma, x, y) {
  const c = 1 / (2 * math.pi * sigma * sigma);
  const upper = -(x * x + y * y) / (2 * sigma * sigma);
  return c * math.exp(upper);
}

/**
 * 得到高斯矩阵
 * @param size { number }
 * @param sigma
 */
function getGMatrix(size, sigma) {
  const matrix = math.zeros(size, size).map((value, index) => G(sigma, ...index));
  let sum = 0;
  matrix.forEach(value => {
    sum += value;
  });

  return matrix.map(value => value / sum);
}

/**
 * 高斯滤波
 * @param input 输入图像
 * @param size 高斯核大小
 * @param sigma 标准差
 * @return {cv.Mat}
 */
function gaussianBlur(input, size, sigma) {
  const filter = getGMatrix(size, sigma).toArray();
  const half = math.floor(size / 2);
  const output = new cv.Mat(input.rows, input.cols, cv.CV_8UC1, 0);
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.cols; j++) {

      let sum = 0;
      for (let m = -half; m <= half; m++) {
        for (let n = -half; n <= half; n++) {
          let inputValue = 0;

          if (0 < i + m && i + m < input.rows && 0 < j + n && j + n < input.cols) {
            inputValue = input.at(i + m, j + n);
          }

          sum += inputValue * filter[m + half][n + half];

        }
      }
      sum = math.max(0, sum);
      sum = math.min(255, sum);
      output.set(i, j, sum);
    }
  }
  return output;
}

module.exports = {
  gaussianBlur
};