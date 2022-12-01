import numpy as np
from tqdm import tqdm

import geopandas
import pickle

import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from dataset import prepare_data


global global_val

class ParisDatasetLoader(object):
    """Based on
    https://pytorch-geometric-temporal.readthedocs.io/en/latest/
    _modules/torch_geometric_temporal/dataset/pems_bay.html
    """

    def __init__(self, data, travel_times, input_features, output_feature):
        super(ParisDatasetLoader, self).__init__()
        self.df = data
        self.travel_times = travel_times
        self.input_features = input_features
        self.output_feature = output_feature
        self._read_data()

    def _read_data(self):
        A = self.travel_times        
        X = np.zeros((len(self.df.paris_id.unique()),
                      len(self.input_features),
                      self.df.time_idx.max()-self.df.time_idx.min()+1))
        list_miss_det = []
        list_miss_time = []

        for i, det_id in tqdm(enumerate(np.sort(self.df.paris_id.unique()))):
            temp = self.df[self.df.paris_id==det_id]
            for k, time_instance in enumerate(range(self.df.time_idx.min(), self.df.time_idx.max())):
                try:
                    temp_t = temp[temp.time_idx==time_instance]
                    for j, feature in enumerate(self.input_features):
        # print(                print(temp_t.loc[:, feature])
                        X[i, j, k] = list(temp_t.loc[:, feature])[0]
                except IndexError:
#                     list_miss_det.append(det_id)
#                     list_miss_time.append(time_instance)
                    for j, feature in enumerate(self.input_features):
                        
                        X[i, j, k] = 0
            
        X = X.astype(np.float32)
        
        
        Y = np.zeros((len(self.df.paris_id.unique()),
                                len(self.output_feature),
                                self.df.time_idx.max()-self.df.time_idx.min()+1))
        
        

                               
        for i, det_id in tqdm(enumerate(np.sort(self.df.paris_id.unique()))):
            temp = self.df[self.df.paris_id==det_id]
            for k, time_instance in enumerate(range(self.df.time_idx.min(), self.df.time_idx.max())):                
                try:
                    temp_t = temp[temp.time_idx==time_instance]
                    for j, feature in enumerate(self.output_feature):
        #                 print(temp_t.loc[:, feature])
                        Y[i, j, k] = list(temp_t.loc[:, feature])[0]
                except IndexError:
                    time_rem = time_instance%7
                    
                    for j, feature in enumerate(self.output_feature):
                        ##### inputing missing values as the weekday average
                        global global_val 
#                         print(temp[temp.time_idx%7==time_rem])
                        Y[i, j, k] = np.nanmean(temp[temp.time_idx%7==time_rem][feature])#[0]
        
        Y = Y.astype(np.float32)
        print(Y)
        
        global_val = Y


        # Normalise as in DCRNN paper (via Z-Score Method)
        means_X = np.mean(X, axis=(0, 2))
        X = X - means_X.reshape(1, -1, 1)
        stds_X = np.std(X, axis=(0, 2))
        X = X / stds_X.reshape(1, -1, 1)
        
        self.means_Y = np.nanmean(Y, axis=(0, 2))
        Y = Y - self.means_Y.reshape(1, -1, 1)
        self.stds_Y = np.nanstd(Y, axis=(0, 2))
        Y = Y / self.stds_Y.reshape(1, -1, 1)
        
        
        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int=12, num_timesteps_out: int=12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [(i, i + (num_timesteps_in + num_timesteps_out)) for i
                   in range(self.X.shape[2] - (
                    num_timesteps_in + num_timesteps_out) + 1)]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i: i + num_timesteps_in]).numpy())
            target.append((self.Y[:, :, i + num_timesteps_in: j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(self, num_timesteps_in: int=12, num_timesteps_out: int=12)  -> StaticGraphTemporalSignal:
        """Returns data iterator for Paris Dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PEMS-BAY traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
        return dataset, self.means_Y, self.stds_Y


def main_loader(input_window, prediction_horizon, time_threshold=10):
    # time_threshold = 10

    filename = "../data/paris_detectors.matched.geojson"
    file = open(filename)
    streets_det = geopandas.read_file(file)

    streets_det.to_crs(epsg=3857, inplace=True) 

    data = "../data/fusion/traffic_data_2019.csv"
    X_formatted, det_ids = prepare_data(data, ["trunk"])

    streets_det_undup = streets_det.drop_duplicates(subset=['pp_iu_ac']).reset_index(drop=True)
    streets_det_undup.pp_iu_ac  = streets_det_undup.pp_iu_ac.astype(int)
    streets_det_undup = streets_det_undup[streets_det_undup.pp_iu_ac.isin(X_formatted.paris_id)]
    streets_det_undup.reset_index(drop=True, inplace=True)

    # df = pd.read_csv("../data/static_test.csv")

    input_features = list(X_formatted.columns)
    input_features.remove('q')
    input_features.remove('k')
    input_features.remove('paris_id')                       
    input_features.remove("time_idx")
    output_feature = ['q']

    df = X_formatted

    travel_times = np.genfromtxt('../output/travel_times.csv', delimiter=',')

    ### First parameter is the cutoff travel-time within which it shall be considered
    ### that there exists one edge    
    for i in range(len(travel_times)):
        for j in range(len(travel_times)):
            if i==j:
                travel_times[i, j] = 0
            elif travel_times[i, j]> time_threshold:
                travel_times[i, j] = 0
            else:
                travel_times[i, j] = 1

    travel_times_new = np.zeros((len(df.paris_id.unique()), len(df.paris_id.unique())))

    for i, det_df_s in enumerate(df.paris_id.unique()):
        s = np.where(np.array(streets_det_undup.pp_iu_ac)==det_df_s)
        for j, det_df_t in enumerate(df.paris_id.unique()):
            t = np.where(np.array(streets_det_undup.pp_iu_ac)==det_df_t)
            travel_times_new[i, j] = travel_times[s, t]

    paris_loader = ParisDatasetLoader(df, travel_times_new, input_features, output_feature)
    paris_data, y_scaler_mean, y_scaler_std = paris_loader.get_dataset(num_timesteps_in = input_window, 
                                        num_timesteps_out=prediction_horizon)


    with open( "../output/paris_data_test_"+str(input_window)+"_"+str(prediction_horizon)+"_"+str(time_threshold)+".pkl", 'wb') as pickle_file:
        pickle.dump([paris_data, y_scaler_mean, y_scaler_std] , pickle_file)

if __name__ == "__main__":
    for lb, ph in [(3,1), (6,1), (9, 1), (6, 3), (6, 6), (6, 9), (12, 1),(1,1)]:
        main_loader(input_window=lb, prediction_horizon=ph)