# anonymous_code
supplementary code, anonymized for NIPS submission #8290

Dependencies:
  Users are recommended to create conda env from the .yml file attached
  This code is compatible with PyTorch 1.0.0 and latest sklearn, numpy, matplotlib

To deploy training code:
  Custom experiment: 
  python -u vaegp.py [result_folder_prefix] [method] [dataset] [cuda_device] [seed_number] [batch_size] 2>&1 > [log_file]

  Default (Revisited SSGP with 32 samples on Gas-10k dataset, batch size=1000): 
  python -u vaegp.py

Methods available:
  'full': Full GP 
  'ssgp_x': SSGP with x samples
  'vaegp_x': Revisited SSGP with x samples

Dataset:
  'abalone': Abalone dataset
  'gas-10': Sample 10k data points from Gas Sensor dataset
  'gas-500': Sample 500k data points from Gas Sensor dataset

Remark:
  Gas dataset is not available on this repo because it exceeds the maximum file size
  Please download the .csv files from ... and run utility.py to prepare a single .npy file containing all data points

Evaluation code:
  Gather log files and run performance_plot.py (with appropriate settings -- see inside the code) to generate RMSE vs. Iter plots.
  Gather saved model files (.pth) and run embedding_plot.py (with appropriate settings -- see inside the code) to visualize data re-orientation.
