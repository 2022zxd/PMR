For testing the MR predictor, the code representation model is applied on the input programs, 
and the resulting code vectors(mark MR labels to create datasets) are then fed into the classification model, yielding applicable MRs as the prediction results.

##The main steps are：
*first, use the trained or pre-trained code representation model for transforming a set of programs to their vector representations, 
which accordingly form the basis for the data preparation phase. 
*second, feed the code vector into the classifier, yielding applicable MRs as the prediction results.

you can run singleMR.py to obtain the evaluation results of each MRs, and run overall.py to obtain the overall evaluation results.

you can run like is:

	python singleMR.py -i "/home/zxd/Desktop/PMR/data/vector/python/UniXcoder-python.csv" -ns 10 

After running, the corresponding evaluation results will be given.

The classifier that runs by default is SVM and the language is python. 
When you want to run other operations(classifier or language java/C++), you need to make corresponding modifications to the code content.
