import numpy as np

class MyNaiveBayes:
    #gather some basic information about the dataset
    #to caluclate the summary statsistica of classes and features
    #including the prior
    #need ot collect the mean, variance and prior to the class
    def fit(self, X, y):
        self.init_statistics(X, y)
        self.calc_statistics(X, y)

    #main function that predicts the classification
    def predict(self, X):
        #for each sample x in the dataset X
        y_hat = [self.prob_of_class(x) for x in X]
        return np.array(y_hat)

    #iterates through all calsses and caluclates a new posterior
    #for a sample
    def prob_of_class(self, x):
        #store all the posterior information
        posteriors = list()
        
        #iterate through all classes
        for c in range(self.num_classes):
            #get mean variance and prior
            Mean = self.mean[c]
            Variance = self.variance[c]
            Prior = np.log(self.priors[c])
            #caluclate the new postierior and add to the list
            posterior = np.sum(np.log(self.gaussian_density(x, Mean, Variance)))
            posterior = Prior + posterior
            posteriors.append(posterior)
        #return the index of the highest probability in the poststeriors set
        return np.argmax(posteriors)
    
    #creates the gausisan distribution for a class
    #given the class, mean, and variance
    def gaussian_density(self, x, Mean, Variance):
        const = 1 / np.sqrt(Variance * 2 * np.pi)
        probability = np.exp(-0.5 * (np.square((x - Mean)/Variance)))
        return const * probability

    def init_statistics(self, X, y):
        # get the number of samples and features
        # samples are contained in rows and features are
        #contained in columns
        self.num_samples, self.num_features = X.shape

        #find the number of unique classes that exist in the data
        self.num_classes = len(np.unique(y))
    
        #create three matriceis to store all the nececcary information
        #mentioned above
        self.mean = np.zeros((self.num_classes, self.num_features))
        self.variance = np.zeros((self.num_classes, self.num_features))
        self.priors = np.zeros(self.num_classes)

    def calc_statistics(self, X, y):
        #iterate through the classes and compute the statistics to
        #update the mean, variance and prior of each class
        for c in range(self.num_classes):
            #create subset for class c
            subX = X[y == c]
            #calulate the stats and update the information
            self.mean[c, :] = np.mean(subX, axis=0)
            self.variance[c, :] = np.var(subX, axis=0)
            self.priors[c] = subX.shape[0] / self.num_samples