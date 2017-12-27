const Matrix = require('ml-matrix');

/**
 * 线性回归基本类型
 */

class MultivariateLinearRegression {

    /**
     * 输入两个映射数据集合
     * @param x
     * @param y
     * @param options
     */
    constructor(x, y, options = {}) {
        const {
            intercept = true
        } = options;
        if (x === true) {
            this.weights = y.weights;
            this.inputs = y.inputs;
            this.outputs = y.outputs;
            this.intercept = y.intercept;
        } else {
            if (intercept) {
                x = new Matrix(x);
                x.addColumn(new Array(x.length).fill(1));
            }
            this.weights = new Matrix.SVD(x, { autoTranspose: true }).solve(y).to2DArray();
            this.inputs = x[0].length;
            this.outputs = y[0].length;
            if (intercept) this.inputs--;
            this.intercept = intercept;
        }
    }

    /**
     * 预测对应的向量
     * @param x
     * @return {any[]}
     */
    predict(x) {
        if (Array.isArray(x)) {
            if (typeof x[0] === 'number') {
                return this._predict(x);
            } else if (Array.isArray(x[0])) {
                const y = new Array(x.length);
                for (let i = 0; i < x.length; i++) {
                    y[i] = this._predict(x[i]);
                }
                return y;
            }
        }
    }

    _predict(x) {
        const result = new Array(this.outputs);
        if (this.intercept) {
            for (let i = 0; i < this.outputs; i++) {
                result[i] = this.weights[this.inputs][i];
            }
        } else {
            result.fill(0);
        }
        for (let i = 0; i < this.inputs; i++) {
            for (let j = 0; j < this.outputs; j++) {
                result[j] += this.weights[i][j] * x[i];
            }
        }
        return result;
    }

    /**
     * 存储加载函数
     * @return {{name: string, weights: *, inputs: *, outputs: *, intercept: *}}
     */
    toJSON() {
        return {
            name: 'multivariateLinearRegression',
            weights: this.weights,
            inputs: this.inputs,
            outputs: this.outputs,
            intercept: this.intercept
        };
    }

    static load(model) {
        return new MultivariateLinearRegression(true, model);
    }
}

module.exports = {
    MultivariateLinearRegression
};



