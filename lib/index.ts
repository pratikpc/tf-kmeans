import * as tf from "@tensorflow/tfjs-core";

export default class KMeans {
    public k: number = 2;
    public maxIter: number = 200;
    public distanceFunction = KMeans.EuclideanDistance;
    public centroids!: tf.Tensor;

    public constructor({ k = 2, maxIter = 10, distanceFunction = KMeans.EuclideanDistance } = {}) {
        this.k = k;
        this.maxIter = maxIter;
        this.distanceFunction = distanceFunction;
    }

    public static EuclideanDistance(values: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => values.squaredDifference(centroids).sum(1).sqrt());
    }
    private GenerateIndices(rows: number) {
        const indices: number[] = [];
        indices.length = rows;
        for (let i = 0; i < indices.length; ++i)
            indices[i] = i;
        return indices;
    }
    private NewCentroid(values: tf.Tensor, assignments: tf.Tensor, cluster: number, rows: number) {
        return tf.tidy(() => {
            // Make All Values Of Array to be of Same Size as Our Cluster
            let selectedIndices: number[] = [];
            selectedIndices.length = rows;
            selectedIndices = selectedIndices.fill(cluster);
            const selectedIndicesT = tf.tensor(selectedIndices);
            
            let where = tf.equal(assignments, selectedIndicesT).asType("int32");
            where = where.reshape([where.shape[0], 1]);
            const count = where.sum();

            const newCentroid = values.mul(where).sum(0).div(count)
            return newCentroid;
        })
    }
    private NewCentroids(values: tf.Tensor, assignments: tf.Tensor) {
        return tf.tidy(() => {
            const rows = values.shape[0];
            const centroids: tf.Tensor[] = [];
            for (let cluster = 0; cluster < this.k; ++cluster) {
                centroids.push(this.NewCentroid(values, assignments, cluster, rows));
            }
            return tf.stack(centroids);
        });
    }
    private AssignCluster(value: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => this.distanceFunction(value, centroids).argMin(0));
    }
    private AssignClusters(values: tf.Tensor, centroids: tf.Tensor) {
        return tf.tidy(() => {
            const rows = values.shape[0];
            const minIndexes: tf.Tensor[] = [];
            for (const index of this.GenerateIndices(rows)) {
                const value = values.gather(index);
                minIndexes.push(this.AssignCluster(value, centroids));
                value.dispose();
            }
            return tf.stack(minIndexes);
        });
    }
    private RandomSample(vals: tf.Tensor) {
        return tf.tidy(() => {
            const rows = vals.shape[0];
            if (rows < this.k)
                throw new Error("Rows are Less than K");

            const indicesRaw = tf.util.createShuffledIndices(rows).slice(0, this.k);
            const indices: number[] = [];
            indicesRaw.forEach((index: number) => indices.push(index))
            // Extract Random Indices
            return tf.gatherND(vals, tf.tensor(indices, [this.k, 1], "int32"))
        })
    }
    private CheckCentroidSimmilarity(newCentroids: tf.Tensor, centroids: tf.Tensor, vals: tf.Tensor) {
        return tf.tidy(() => newCentroids
            .equal(centroids)
            .asType("int32")
            .sum(1)
            .div(vals.shape[1]!)
            .sum()
            .equal(this.k)
            .dataSync()[0]
        );
    }
    public Train(values: tf.Tensor) {
        this.centroids = this.RandomSample(values);
        let assignments = tf.tensor([]);

        let iter = 0;
        while (true) {
            assignments.dispose();
            assignments = this.AssignClusters(values, this.centroids);
            const newCentroids = this.NewCentroids(values, assignments);
            const same = this.CheckCentroidSimmilarity(newCentroids, this.centroids, values);
            if (same || iter >= this.maxIter) {
                newCentroids.dispose();
                break;
            }
            this.centroids.dispose();
            this.centroids = newCentroids;
            ++iter;
        }

        return assignments;
    }
    public Predict(y: tf.Tensor) {
        return tf.tidy(() => {
            if (y.shape[1] == null)
                y = y.reshape([1, y.shape[0]]);
            return this.AssignClusters(y, this.centroids);
        });
    }
    public Centroids() {
        return this.centroids;
    }
    public Dispose() {
        this.centroids.dispose();
    }
}