// process.env.DEBUG = 'node-vault'; // switch on debug mode
const KMeans = require("tf-kmeans");
const tf = require("@tensorflow/tfjs");
tf.tidy(() => {
	const kmeans = new KMeans.default({k:2, maxIter: 30, distanceFunction: KMeans.default.EuclideanDistance});
	console.log("Assigned To ");
	kmeans.Train(tf.tensor([[2, 2, 2], [5, 5, 5], [3, 3, 3], [4, 4, 4], [7, 8, 7]])).print();
	console.log("Centroids Used are ");
	kmeans.Centroids().print();
	console.log("Prediction for Given Value is");
    kmeans.Predict(tf.tensor([2, 3, 2])).print();
    console.log(tf.memory());
})