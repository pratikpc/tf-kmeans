# [TF-KMeans](https://github.com/pratikpc/TF-KMeans)

## Description

A Simple JavaScript Library to make it easy for people to use KMeans algorithms with Tensorflow JS.

The library was born out of another project in which except KMeans, our code completely depended on TF.JS

As such, moving to TF.JS helped standardise our code base substantially and reduce dependency on other libraries

## [Sample Code](samples/index.js)

~~~javascript
	const KMeans = require("tf-kmeans");
	const tf = require("@tensorflow/tfjs");
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
~~~

You can use the Asynchronous TrainAsync if you want to use an asynchronous callback function

~~~javascript
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
		// This function is Asynchronous
		async(iter, centroid, preds)=>{
			console.log("===");
			console.log("Iteration Count", iter);
			console.log("Centroid ", await centroid.array());
			console.log("Prediction ", await preds.array());
			console.log("===");
			// You could instead use TFVIS for Plotting Here
		}
	);
~~~

## Functions

1. ***`Constructor`***

    Takes 3 Optional parameters
    1. k:-                Number of Clusters
    2. maxIter:-          Max Iterations
    3. distanceFunction:- The Distance function Used
	                      Currently only Eucledian Distance Provided

2. ***`Train`***

    Takes Dataset as Parameter

	Performs Training on This Dataset

	Sync callback function is _optional_

3. ***`TrainAsync`***

    Takes Dataset as Parameter

	Performs Training on This Dataset

	Also takes _async_ callback function called at the end of every iteration

4. ***`Centroids`***

	Returns the Centroids found for the dataset on which KMeans was Trained

5. ***`Predict`***

	Performs Predictions on the data Provided as Input

## PEER DEPENDENCIES
1. [TensorFlow.JS](https://www.tensorflow.org/js "tfjs")

## Typings
As the code is originally written in TypeScript, Type Support is provided out of the box

## Contact Me
You could contact me [via LinkedIn](https://www.linkedin.com/in/pratik-chowdhury-889bb2183/ "via LinkedIn")
You could file issues or add features via Pull Requests on GitHub
