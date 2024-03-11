import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm

class Preprocessor(object):
    """
    Preprocessing application log data before giving it as an input to the deep learning model.
    """
    def __init__(self, context_length, timeout, NO_LOG=-1337):
        """
        Parameters
        ----------
        context_length : int
            Number of logs in context

        timeout : float
            Maximum time between context logs and the actual log (in seconds)
        
        NO_LOG : int, default=-1337
            ID of NO_LOG log, when there is no adequate context log.
        """
        self.context_length = context_length
        self.timeout = np.timedelta64(timeout, 's')
        self.NO_LOG = NO_LOG

        self.REQUIRED_COLUMNS = {'timestamp', 'thread', 'body', 'method_name'}
    


    def sequence(self, data, verbose=None):
        """
        Transform pandas DataFrame into input sequences of log_anomaly_detection model.

        Parameters
        ----------
        data : pd.DataFrame
                Dataframe to preprocess.

        verbose : boolean, default=False
            If True, prints progress in transforming input to sequences.

        Returns
        -------
        context : torch.Tensor of shape=(n_samples, context_length)
            Context logs for each log in logs.

        logs : torch.Tensor of shape=(n_samples,)
            logs in data.

        labels : torch.Tensor of shape=(n_samples,)
            Labels will be None if data does not contain any 'labels' column.

        mapping : dict()
            Map each unique logs to IDs(i.e. numbers).
        """

        ###################################################################
        #                    Transformation & Checks                      #
        ###################################################################
        
        # make new column 'logs' from 'body' and 'method_name'
        data['logs'] = data['method_name'] + ' ' + data['body']
        
        # if data contains label columns bring it in as 'labels', else labels=None
        if 'Label' in data.columns:
            labels = data['Label']
            labels = labels.map({'abnormal':0, 'normal':1})
            labels = np.asarray(labels)
        
        # transform timestamp from str to datetime
        data['timestamp'] = [datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f') for time in data['timestamp']]

        # Check if data contains required columns
        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                "data must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        # Check if labels is same shape as data
        if labels.ndim and (labels.shape[0] != data.shape[0]):
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )
        
        ###################################################################
        #                           Map events                            #
        ###################################################################

        # mapping to be returned in this function
        mapping = {i:log for i, log in enumerate(np.unique(data['logs'].values))}

        mapping[len(mapping)] = self.NO_LOG
        mapping_inverse = {v:k for k,v in mapping.items()}

        # apply mapping to logs
        data['logs'] = data['logs'].map(mapping_inverse)

        ###################################################################
        #                        Initialize returns                       #
        ###################################################################

        # Set logs
        logs = torch.Tensor(data['logs'].values).to(torch.long)

        # initialize context
        context = torch.full(
            size       = (data.shape[0], self.context_length),
            fill_value = mapping_inverse[self.NO_LOG]
        ).to(torch.long)

        # Set labels if given
        if labels.ndim:
            labels = torch.Tensor(labels).to(torch.long)
        else:
            labels = None
        
        ################################################################
        #                        Create context                        #
        ################################################################

        # Sort data by timestamp
        data = data.sort_values(by='timestamp')
    
        # Group by thread
        thread_grouped = data.groupby('thread')
        
        # Add verbosity
        if verbose: 
            thread_grouped = tqdm(thread_grouped, desc='Loading')

        # Group by thread
        for thread, logs_ in thread_grouped:
            indices = logs_.index.values
            timestamps = logs_['timestamp'].values
            logs_ = logs_['logs'].values

            # initialize thread_context
            thread_context = np.full(
                (logs_.shape[0], self.context_length),
                mapping_inverse[self.NO_LOG],
                dtype = int
            )

            # fill thread_context (from back part)
            for i in range(self.context_length):
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                timeout_mask = time_diff > self.timeout

                thread_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_LOG],
                    logs_[:-i-1]
                )

            thread_context = torch.Tensor(thread_context).to(torch.long)
            context[indices] = thread_context

        return context, logs, labels, mapping

    ########################################################################
    #                     Preprocess different formats                     #
    ########################################################################

    def csv(self, path, nrows=None, verbose=False):
        """Preprocess data from csv file.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            Returns
            -------
            logs : torch.Tensor of shape=(n_samples,)

            context : torch.Tensor of shape=(n_samples, context_length)

            labels : torch.Tensor of shape=(n_samples,)

            mapping : dict()
              
            """
        # Read data from csv file into pandas dataframe
        data = pd.read_csv(path, nrows=nrows)

        # Transform to sequences and return
        return self.sequence(data, verbose=verbose)
    
