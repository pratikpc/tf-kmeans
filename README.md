# [TF-KMeans](https://github.com/pratikpc/TF-KMeans)

## Sample Code

~~~
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
~~~

## TODO
1. Add Additional Distance Algorithms
2. Perform Optimisations Wherever Possible

#### PEER DEPENDENCIES
1. [TensorFlow.JS](https://www.tensorflow.org/js "tfjs")


#### Contact Us
You could contact me [via LinkedIn](https://www.linkedin.com/in/pratik-chowdhury-889bb2183/ "via LinkedIn")
You could file issues or add features via Pull Requests on GitHub