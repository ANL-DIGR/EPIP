# EPIP
Evaluation of Precipitation Instrument Performance

THis project uses basic machine learning methods to determine a precipitation best estimate.
Work is also being done to determine the performance of differents instruments across different rain rates.

parse_data.py - downloads data from multiple ARM precipitation instruments at SGP, corrects the data using 
   PyDSD, and applies QC filters.
   
precip_clustering.py was the initial program to test out different clustering techniques.  

vis_kmeans.py can be used for visualizing the clustering at each time step

cluster_binning.py is similar to precip_clustering, but starts to visualize the data to get at
    instrument weights.
