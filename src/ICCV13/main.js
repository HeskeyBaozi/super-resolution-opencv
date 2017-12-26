const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const { getYChannelMatAsync } = require('../utils/helpers');
const K_means = require('node-kmeans');
const MLR = require('ml-regression-multivariate-linear');
const mathjs = require('mathjs');


async function trainPairs(dir) {

    const originNames = fs.readdirSync(dir).map(name => name.replace(/\..*$/, ''));

    // train
    for (const name of originNames) {
        console.log(`[Train] ${name}`);
        const inputMat = await cv.imreadAsync(path.resolve(dir, name + '.jpg'));
        await trainOne(inputMat, 1.6, name);
    }

    // k-means
    let features = [];
    let centers = [];
    let names = [];
    if (!fs.existsSync('features.json')) {
        for (const name of originNames) {
            if (name.startsWith('BSD')) {
                const dbPath = path.resolve(__dirname, `./db/${name}.json`);
                console.log('[Read] ' + name);
                const { LR_features, HR_centers, HR_names } = require(dbPath);
                features.push(...LR_features);
                centers.push(...HR_centers);
                names.push(...HR_names);

                if (features.length >= 400000) {
                    console.log('data length is >= 40w, break!');
                    break;
                }
            }
        }

        fs.writeFileSync('features.json', JSON.stringify(features));
        fs.writeFileSync('centers.json', JSON.stringify(centers));
        fs.writeFileSync('names.json', JSON.stringify(names));
        console.log('writeFile!');
    } else {
        features = require('./features.json');
        centers = require('./centers.json');
        names = require('./names.json');
    }

    console.log('get feature, length = ', features.length);
    if (false) {
        const result = await KMeans(features, { k: 256 });
        console.log('kmeans computed');
        for (let classCase = 0; classCase < 256; classCase++) {
            if (!fs.existsSync(`./V/${classCase}.json`)) {
                const { centroid, cluster, clusterInd } = result[classCase];
                const V_Matrix = new cv.Mat(45, cluster.length, cv.CV_32FC1);
                const M_Matrix = new cv.Mat(144, cluster.length, cv.CV_32FC1);

                // get V Matrix
                for (let r = 0; r < V_Matrix.rows; r++) {
                    for (let c = 0; c < V_Matrix.cols; c++) {
                        V_Matrix.set(r, c, Number.parseFloat(cluster[c][r].toFixed(3)));
                    }
                }

                let lastName = null;
                let HR = null;

                for (let r = 0; r < M_Matrix.rows; r++) {
                    for (let c = 0; c < M_Matrix.cols; c++) {
                        const index = clusterInd[c];
                        const center = centers[index];
                        const name = names[index];

                        if (name !== lastName) {
                            HR = await cv.imreadAsync(path.resolve(__dirname, '../../train/' + name + '.jpg'));
                            HR = await getYChannelMatAsync(HR);
                            lastName = name;
                        }
                        //cv.imshowWait(name, HR);

                        const vector = [];

                        for (let i = center.r - 6; i < center.r + 6; i++) {
                            for (let j = center.c - 6; j < center.c + 6; j++) {
                                vector.push(HR.at(i, j));
                            }
                        }

                        //console.log('W vector length =', vector.length);
                        M_Matrix.set(r, c, vector[r]);
                    }
                }

                console.log(V_Matrix);
                console.log(M_Matrix);

                await cv.imwriteAsync(`./V/${classCase}.bmp`, V_Matrix);
                await cv.imwriteAsync(`./M/${classCase}.bmp`, M_Matrix);
                fs.writeFileSync(`./V/${classCase}.json`, JSON.stringify(V_Matrix.getDataAsArray()));
                fs.writeFileSync(`./M/${classCase}.json`, JSON.stringify(M_Matrix.getDataAsArray()));
                fs.writeFileSync(`./center/${classCase}.json`, JSON.stringify(centroid));

                console.log('write Matrix File!' + classCase);
            }
        }
    }

    if (!fs.existsSync('./C/255.json')) {
        for (let count = 0; count < 256; count++) {
            console.log('run C', count);
            const V = require(`./V/${count}.json`);
            const M = require(`./M/${count}.json`);

            const X = mathjs.transpose(V);
            const Y = mathjs.transpose(M);

            const mlr = new MLR(X, Y);

            fs.writeFileSync(`./C/${count}.json`, JSON.stringify(mlr.toJSON()));
        }
    }

    return;
}

// run!
trainPairs(path.resolve(__dirname, '../../', './train'));

async function trainOne(inputMat, sigma, name) {
    const dbPath = path.resolve(__dirname, `./db/${name}.json`);
    if (!fs.existsSync(dbPath)) {
        const Y = await getYChannelMatAsync(inputMat);
        const generated = await GenerateFeatureFromHR(Y, sigma, name);
        fs.writeFileSync(dbPath, JSON.stringify(generated));
    }
}

async function GenerateFeatureFromHR(HR_Mat, sigma, name) {
    const patchSize = 7;
    const patchSizeHalf = (patchSize - 1) / 2;

    const LR = await GenerateLR(HR_Mat, sigma);
    //cv.imshowWait('GS, down 1/3', LR);

    const LR_features = [];
    const HR_centers = [];
    const HR_names = [];

    // r => 3, 4, ...,
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
            LR_features.push(vector_mean);
            // zoom out 3
            HR_centers.push({ r: r * 3, c: c * 3 });
            HR_names.push(name);

        }
    }

    return {
        LR_features,
        HR_centers,
        HR_names
    };
}

/**
 * Il = (Ih ⊗ G) ↓s
 * 高斯模糊之后下采样
 * @param HR_Mat
 * @param sigma
 * @return {Promise<*>}
 */
async function GenerateLR(HR_Mat, sigma) {
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
    const blur = await HR_Mat.gaussianBlurAsync(new cv.Size(kernelSize, kernelSize), sigma);
    return await blur.resizeAsync(Math.floor(blur.rows / 3), Math.floor(blur.cols / 3));
}

/**
 * the clustering algorithm k-means
 * K Means 聚类算法
 * @param vectors
 * @param options
 * @return {Promise<Object>}
 */
async function KMeans(vectors, options) {
    return new Promise((resolve, reject) => {
        K_means.clusterize(vectors, options, (error, res) => {
            if (error) {
                reject(error);
            } else {
                resolve(res);
            }
        });
    });
}