const KMeans = require("tf-kmeans");
const tf = require("@tensorflow/tfjs");
function SyncTest() {
	tf.tidy(() => {
		const kmeans = new KMeans.default({
			k: 2,
			maxIter: 30,
			distanceFunction: KMeans.default.EuclideanDistance
		});
		const dataset = tf.tensor([[2, 2, 2], [5, 5, 5], [3, 3, 3], [4, 4, 4], [7, 8, 7]]);
		const predictions = kmeans.Train(
			dataset
		);

		console.log("Assigned To ", predictions.arraySync());
		console.log("Centroids Used are ", kmeans.Centroids().arraySync());
		console.log("Prediction for Given Value is");
		kmeans.Predict(tf.tensor([2, 3, 2])).print();
		console.log("Amount of Memory Used is ", tf.memory());
		// Use this In case kmeans not executed in Tidy Function
		kmeans.Dispose();
		predictions.dispose();
		dataset.dispose();
	});
}

async function AsyncTest() {
	const kmeans = new KMeans.default({
		k: 3,
		maxIter: 30,
		distanceFunction: KMeans.default.EuclideanDistance
	});
	const dataset = tf.tensor([[2, 2, 2], [5, 5, 5], [3, 3, 3], [4, 4, 4], [7, 8, 7]]);

	console.log("\n\nAsync Test");
	const predictions = await kmeans.TrainAsync(
		dataset,
		// Called At End of Every Iteration
		async(iter, centroid, preds)=>{
			console.log("===");
			console.log("Iteration Count", iter);
			console.log("Centroid ", await centroid.array());
			console.log("Prediction ", await preds.array());
			console.log("===");
			// You could instead use TFVIS for Plotting Here
		}
	);
	console.log("Assigned To ", await predictions.array());
	console.log("Centroids Used are ", await kmeans.Centroids().array());
	console.log("Prediction for Given Value is");
	kmeans.Predict(tf.tensor([2, 3, 2])).print();
	console.log("Amount of Memory Used is ", tf.memory());

	kmeans.Dispose();
	predictions.dispose();
	dataset.dispose();
}

SyncTest();

AsyncTest().then(() => console.log("Hi"));