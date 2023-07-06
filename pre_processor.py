from sklearn.model_selection import train_test_split

class PreProcessor:

    def pre_process(self, dataset, percentual_teste, seed=7):
        """_summary_

        Args:
            dataset (Dataframe): data frame to run proccess
            percentual_teste (float): is test percentage
            seed (int, optional): random seed. Defaults to 7.

        Returns:
            tuple: contains the list of split 
        """

        X_train, X_test, Y_train, Y_test = self.__preparar_holdout(dataset,
                                                                  percentual_teste,
                                                                  seed)
        
        return (X_train, X_test, Y_train, Y_test)

    def __preparar_holdout(self, df, percentual_test, seed):
        """Divide train/test date with holdout.

        Args:
            df (Dataframe):
            percentual_test (float): is test percentage
            seed (int): random seed

        Returns:
            list: train and test splited lists
        """
        
        data = df.values
        X = data[:, 0:-1]
        Y = data[:, -1]
        return train_test_split(X, Y, test_size=percentual_test, random_state=seed)