# SimilaritiesTS
Python package for clustering latent representation of multivariate multiple time series. Version 0.2.0

It takes DataFrame of very long time series (especially designed for minute ticks of stocks/commodities, especially log returns of those) and creates a Multivariate Multiple time-series of given chunk length (with non overlapping and sliding window methods).
Dataset created in this way, can be subject of additional transofrmation, for example - reduce dimensions with HybridVAE and later cluster it around centroids by k-means.
HybridVAE is attention-base variational autoencoder Neural Network.

## Future work will include:
- Additional clustering methods
- HybridVAE generation methods
- Pipeline class (in transformers style)
- PyPi publishing

________________
## Credits
Part of the inspiration for package came from:
Hybrid Variational Autoencoder for Timeseries Forecasting: https://arxiv.org/pdf/2303.07048, credits to Borui Cai, Shuiqiao Yang, Longxiang Gao, Yong Xiang
Implementational changes: LeakyReLU instead of ReLu for conv transpose layers, customized number of layers, upsampling in Decoder.

HybridVAE from this package was based on the afromentioned paper.

________________
## How it works

Below is an example, assume the package is already added to the system path.

```
import similarities_ts as sts

df = sts.SampleMethods.random_dataset(9, ['random_oscillator','random_oscillator','random_oscillator','brownian_motion','brownian_motion','brownian_motion','noise','noise','noise'], uniform_range=(0,1), num_time_steps=830*2880, initial_value=100, scale=0.5, transform='standardize')
sample_dataset = sts.MultivariateDataset('non_overlapping_window', df, 2880)

CONV_FILTERS = [64,128]
EXAMPLE_PARAMS = {'dataset': SAMPLE_DATASET, 
                 'batch_size': 32,
                 'optimizer': 'Adam',
                 'latent_dim': 1000, 
                 'conv_filters': CONV_FILTERS,
                 'conv_kernel_size': [3,3], 
                 'conv_strides': [2,2],
                 'attention_heads':8,
                 'hidden_dim': 400, 
                 'conv_transpose_filters': [128,9], 
                 'conv_transpose_kernel_sizes': [2,2], 
                 'conv_transpose_strides': [2,2], 
                 'upsample': SAMPLE_DATASET.tensors.shape[-1]/(len(CONV_FILTERS)*2)}

r = sts.Reducer(**EXAMPLE_PARAMS)
r.fit(epochs=10, metrics=['mae','mse'], schedule=True)
latent = r.latent_rep()
clusters = sts.Clusters(latent, True)
kmeans = clusters.get(10,20)
desc = ClustersDescription(kmeans, latent)

for a in range(len(desc)):
  print(desc[a]) 


```
