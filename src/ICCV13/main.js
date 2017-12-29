/**
 * 训练字典，即训练映射LR feature 到 HR patch 的矩阵
 * @author 何志宇<hezhiyu233@foxmail.com>
 */

const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const { getYChannelMatAsync } = require('../utils/helpers');
const K_means = require('node-kmeans');
const { MultivariateLinearRegression } = require('./regress');


// run!
trainPairs(path.resolve(__dirname, '../../', './train'));

/**
 * 在训练完所有的高清图像之后，我们得到了长达百万的feature
 */
async function readFeatureFromDataBase(originNames) {
    let features = [];
    let centers = [];
    let names = [];
    if (fs.existsSync('features.json')) {
        features = require('./features.json');
        centers = require('./centers.json');
        names = require('./names.json');
    } else {
        for (const name of originNames) {
            if (name.startsWith('BSD')) {
                const dbPath = path.resolve(__dirname, `./db/${name}.json`);
                console.log('[Read] ' + name);
                const { LR_features, HR_centers, HR_names } = require(dbPath);
                features.push(...LR_features);
                centers.push(...HR_centers);
                names.push(...HR_names);

                /**
                 * 由于Javascript Heap Memory限制，超过了某一个长度的数组会导致警报，强制退出程序
                 * 这里限制了feature在limit数量以内
                 */
                const limit = 500000;
                if (features.length >= limit) {
                    console.log(`data length is >= ${limit}, break!`);
                    break;
                }
            }
        }

        /**
         * 用.json文件存储处理结果
         */
        fs.writeFileSync('features.json', JSON.stringify(features));
        fs.writeFileSync('centers.json', JSON.stringify(centers));
        fs.writeFileSync('names.json', JSON.stringify(names));
        console.log('writeFile!');
    }

    return {
        features,
        centers,
        names
    };
}


/**
 * 训练 <LR Feature, HR patch> 这一个元组队
 * @param dir 训练集合的路径 指向 train 文件夹
 * @return {Promise<void>}
 */
async function trainPairs(dir) {

    /**
     * 读取的名字数组，长度为训练集的数量
     * ['BSD_2092', 'BSD_3096', ...]
     */
    const originNames = fs.readdirSync(dir).map(name => name.replace(/\..*$/, ''));

    // train
    for (const name of originNames) {
        console.log(`[Train] ${name}`);
        const inputMat = await cv.imreadAsync(path.resolve(dir, name + '.jpg'));

        /**
         * 这里调用训练函数, sigma = 1.6 用于高斯模糊处理
         */
        await trainOne(inputMat, 1.6, name);
    }


    /**
     * 如果没有训练聚类，就训练聚类并存储
     */
    const k = 512;
    if (!fs.existsSync(`./V/${k - 1}.json`) || !fs.existsSync(`./C/${k - 1}.json`)) {
        const { features, centers, names } = await readFeatureFromDataBase(originNames);


        console.log(`[Running] Kmeans, k = ${k}`);

        let result = await KMeans(features, { k });

        while (result.some(({ cluster }) => cluster.length === 0)) {
            console.log('Run again');
            result = null;
            result = await KMeans(features, { k });
        }

        console.log('Run VM');
        for (let count = 0; count < k; count++) {
            const { centroid, cluster, clusterInd } = result[count];

            // 对于每一个聚类

            // @types Array<Vector45>
            const LR_Patch_Array = cluster;


            // @types Array<Vector144>
            const HR_Patch_Array = [];

            let HR = null;
            let lastName = null;

            for (let i = 0; i < clusterInd.length; i++) {
                const index = clusterInd[i];
                const center = centers[index];
                const name = names[index];

                if (name !== lastName) {
                    HR = await cv.imreadAsync(path.resolve(__dirname, '../../train/' + name + '.jpg'));
                    HR = await getYChannelMatAsync(HR);
                    lastName = name;
                }

                const vector_HR = []; // Vector<144>
                for (let i = center.r - 6; i < center.r + 6; i++) {
                    for (let j = center.c - 6; j < center.c + 6; j++) {
                        vector_HR.push(HR.at(i, j));
                    }
                }
                HR_Patch_Array.push(vector_HR);
            }

            // console.log('LR length = ', LR_Patch_Array.length);
            // console.log('HR length = ', HR_Patch_Array.length);
            fs.writeFileSync(`./V/${count}.json`, JSON.stringify(LR_Patch_Array));
            fs.writeFileSync(`./M/${count}.json`, JSON.stringify(HR_Patch_Array));
            fs.writeFileSync(`./center/${count}.json`, JSON.stringify(centroid));
            if (LR_Patch_Array.length === 0) {
                console.log(`[Write ${count}]`);
            }
        }
    }


    // 如果没有系数矩阵，就训练
    if (!fs.existsSync(`./C/${k - 1}.json`)) {
        for (let count = 0; count < k; count++) {
            console.log('[Train C Matrix]', count);
            const V = require(`./V/${count}.json`);
            const M = require(`./M/${count}.json`);


            const mlr = new MultivariateLinearRegression(V, M);

            fs.writeFileSync(`./C/${count}.json`, JSON.stringify(mlr.toJSON()));
        }
    }

    return;
}


/**
 * 对于一个图像的训练函数
 * @param inputMat 要训练的高清图像
 * @param sigma 高斯模糊参数
 * @param name 图像文件名
 * @return {Promise<void>}
 */
async function trainOne(inputMat, sigma, name) {

    /**
     * 用来存储生成结果的路径
     * @type {*|string}
     */
    const dbPath = path.resolve(__dirname, `./db/${name}.json`);
    if (!fs.existsSync(dbPath)) {

        /**
         * 只处理Y通道
         */
        const Y = await getYChannelMatAsync(inputMat);
        const generated = await GenerateFeatureFromHR(Y, sigma, name);
        fs.writeFileSync(dbPath, JSON.stringify(generated));
    }
}

/**
 * 从高清图生成feature的函数
 * @param HR_Mat 高清图矩阵
 * @param sigma 高斯模糊参数
 * @param name 文件名字
 * @return {Promise<{LR_features: Array, HR_centers: Array, HR_names: Array}>}
 */
async function GenerateFeatureFromHR(HR_Mat, sigma, name) {
    const patchSize = 7; // patch 大小
    const patchSizeHalf = (patchSize - 1) / 2; // patch 大小的一般，用于后面迭代计算

    /**
     * 生成低分辨率图像
     * @type {cv.Mat}
     */
    const LR = await GenerateLR(HR_Mat, sigma);
    //cv.imshowWait('GS, down 1/3', LR);

    const LR_features = []; // 存储图像特征
    const HR_centers = []; // 存储该特征对应的高清图中心点
    const HR_names = []; // 存储该图像名字

    // 这里遍历了低分辨率图像的中心内容
    // r => 3, 4, ...,
    for (let r = patchSizeHalf; r < LR.rows - patchSizeHalf; r++) {
        for (let c = patchSizeHalf; c < LR.cols - patchSizeHalf; c++) {

            /**
             * 一个LR Patch来自于该分辨率图像的一个7*7切割
             */
            const patch_LR = LR.getRegion(new cv.Rect(c - patchSizeHalf, r - patchSizeHalf, patchSize, patchSize));

            /**
             * feature是一个长度为45的数组，45 = 7 * 7 - 4
             * @type {Array}
             */
            const vector_LR = [];
            for (let i = 0; i < 7; i++) {
                for (let j = 0; j < 7; j++) {
                    /**
                     * 这里除掉了LR Patch的四个角
                     */
                    if (i === 0 && j === 0 || i === 0 && j === 6 || i === 6 && j === 0 || i === 6 && j === 6) {
                        continue;
                    }
                    vector_LR.push(patch_LR.at(i, j));
                }
            }
            const sum = vector_LR.reduce((sum, next) => sum + next, 0);

            /**
             * vector的平均数
             * @type {number}
             */
            const mean = sum / vector_LR.length;

            /**
             * vector_mean就是一个feature，由于减去了平均数，所以有正有负
             * @type {number[]}
             */
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