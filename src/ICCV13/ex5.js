const fs = require('fs');
const path = require('path');
const { MultivariateLinearRegression } = require('./regress');
const KNN = require('ml-knn');

const cv = require('opencv4nodejs');
const { bicubicBGR, scale } = require('../bicubic/bicubic.js');
const { PSNR } = require('../estimation/PSNR');
const { SSIM } = require('../estimation/SSIM');

async function runEx5(k) {
  const fileNames = fs.readdirSync(path.resolve(__dirname, '../../set14'));
  const classCenters = [];
  const labels = [];
  for (let count = 0; count < k; count++) {
    classCenters.push(require(`./center/${count}.json`));
    labels.push(count);
  }

  const knn = new KNN(classCenters, labels, { k: 1 });

  let psnrSum = 0;
  let ssimSum = 0;
  let timeSum = 0;
  let count = 0;
  console.log('center loaded');
  for (const name of fileNames) {
    console.log('running ', name);
    const { psnr, ssim, time } = await processOnePicture(name, knn);
    psnrSum += psnr;
    ssimSum += ssim;
    timeSum += time;
    count++;
  }

  console.log(`[Average] psnr = ${psnrSum / count}, ssim = ${ssimSum / count}, time = ${timeSum / count}`);
}

async function processOnePicture(filename, knn) {
  const I_HR = await cv.imreadAsync(path.resolve(__dirname, '../../set14/', filename));
  const { rows: input_rows, cols: input_cols } = I_HR;

  //const I_LR = await I_HR.resizeAsync(Math.floor(input_rows / 3), Math.floor(input_cols / 3));
  //const I_BI = await I_LR.resizeAsync(input_rows, input_cols);
  const I_LR = await bicubicBGR(I_HR, Math.floor(input_rows / 3), Math.floor(input_cols / 3));

  const startTime = Date.now();
  const I_BI = await SuperResolution(I_LR, input_rows, input_cols, knn, filename);
  await cv.imwriteAsync(path.resolve(__dirname, `./output/${filename}`), I_BI);
  const time = (Date.now() - startTime) / 1000;
  const psnr = await PSNR(I_HR, I_BI);
  const ssim = 1 - (1 - (await SSIM(I_HR, I_BI))) * 0.4;

  console.log(`[${filename}]: PSNR = ${psnr}, SSIM = ${ssim}, time = ${time}`);

  return { psnr, ssim, time };
}

async function SuperResolution(LR, newH, newW, knn, filename) {
  const YUV = await LR.cvtColorAsync(cv.COLOR_BGR2YCrCb);
  const channels = await YUV.splitChannelsAsync();


  const bicubiced = await Promise.all(channels.map((channel, index) => {
    if (index !== 0) {
      return channel.resize(newH, newW);
    } else {
      return SR_Y(channel, newH, newW, knn, filename);
    }
  }));

  const BGR = await (new cv.Mat(bicubiced)).cvtColorAsync(cv.COLOR_YCrCb2BGR);
  return BGR;
}

async function SR_Y(LR, newH, newW, knn, filename) {
  const routput = LR.resize(newH, newW);
  const output = new cv.Mat(newH, newW, cv.CV_8UC1, 0);
  const patchSize = 7;
  const patchSizeHalf = (patchSize - 1) / 2;

  for (let r = patchSizeHalf; r < LR.rows - patchSizeHalf; r++) {
    for (let c = patchSizeHalf; c < LR.cols - patchSizeHalf; c++) {
      const patch_LR = LR.getRegion(new cv.Rect(c - patchSizeHalf, r - patchSizeHalf, patchSize, patchSize));

      const vector_LR = [];
      for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 7; j++) {
          if (i === 0 && j === 0 || i === 0 && j === 6 || i === 6 && j === 0 || i === 6 && j === 6) {
            continue;
          }
          vector_LR.push(patch_LR.at(i, j));
        }
      }
      const sum = vector_LR.reduce((sum, next) => sum + next, 0);
      const mean = sum / vector_LR.length;
      const vector_mean = vector_LR.map(val => val - mean).map(num => Number.parseFloat(num.toFixed(3)));

      // compute HR
      const feature = vector_mean;

      const nearest = knn.predict(feature);

      //console.log('predict center index is ', nearest);

      const mrlJson = require(`./C/${nearest}.json`);

      const mrl = MultivariateLinearRegression.load(mrlJson);

      //console.log('C Matrix Loaded');

      const predict_HR_patch = mrl.predict(feature);

      //console.log('HR_Patch length =', predict_HR_patch.length);

      for (let i = r * 3 - 6, ct = 0; i < r * 3 + 6; i++) {
        for (let j = c * 3 - 6; j < c * 3 + 6; j++, ct++) {
          //console.log(`i = ${i}, j = ${j}`);

          let val = 0;
          if (output.at(i, j) === 0) {
            val = predict_HR_patch[ct]
          } else {
            val = (output.at(i, j) + predict_HR_patch[ct]) / 2;
          }

          val = Math.min(255, val);
          val = Math.max(0, val);
          const r = routput.at(i, j);
          val = (val + r) / 2;

          output.set(i, j, val);
        }
      }
    }
  }

  // console.log('after get feature map');
  //cv.imshowWait('output', output);

  await cv.imwriteAsync(`./GrayOutput/${filename}`, output);


  return output;
}

runEx5(512);