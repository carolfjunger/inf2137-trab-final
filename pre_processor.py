from sklearn.model_selection import train_test_split

class PreProcessor:

    def pre_process(self, dataset, percentual_teste, seed=7):
        """ Take care of pre process """

        X_train, X_test, Y_train, Y_test = self.__preparar_holdout(dataset,
                                                                  percentual_teste,
                                                                  seed)
        
        return (X_train, X_test, Y_train, Y_test)

    def __preparar_holdout(self, df, percentual_test, seed):
        """ Divide train/test date with holdout.
        Target is the last colunm
        percentual_test is test percentage
        """
        data = df.values
        X = data[:, 0:-1]
        Y = data[:, -1]
        return train_test_split(X, Y, test_size=percentual_test, random_state=seed)