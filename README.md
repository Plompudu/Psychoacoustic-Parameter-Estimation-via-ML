## Project

This project trains a deep neural network to estimate psychoacoustic parameters directly from raw audio.

The model learns to predict five time-varying psychoacoustic parameters **per 1-second segment (48 kHz)**:

- **loudness**  - 500 frames per segment
- **sharpness**  - 500 frames per segment
- **roughness** - 10 frames per segment
- **TNR** - 2 frames per segment
- **SII** - 1 value per segment

The model is build like this:
![model](plot%20paper/DL-model.png)

Reference values are computed with **mosqito** on real audio. The reference signals and their labels are aligned onto a shared 2 ms time grid (500 frames per segment), so each parameter keeps its native time resolution across the 1 s window.
Training uses the checkpointed, resumable training loop in `main.py` and writes checkpoints to `DL_model/epochs`.

The repo also contains the full data-preprocessing pipeline (segmentation, 70/15/15 split, mosqito reference computation, 2 ms-grid alignment, 1 s median pooling, training-set statistics) and the plotting scripts behind the paper figures (data distributions, reference-vs-prediction, loudness runtime benchmark, K-weighting, equal-loudness contours, and inference-time sweeps across CPU/iGPU/GPU).

## What the Code does / How to use it
1. **Manually add raw audio files to data/raw_sound_files**
2. **run the data_preprocessing scripts (in order)**
   1. split raw audio files into <=1s long segments per stereo channel
      - input: data/raw_sound_files 
      - output: data/raw_sound_files_1s (each stereo file yields a separate left and right mono segment)
   2. split data into training, test and validation set (70:15:15, randomized selection, no fixed seed)
      - input: data/raw_sound_files_1s 
      - output: data/test_set , data/training_set , data/validation_set
   3. calculate reference values for psychoacoustic parameters
      - input_1: data/raw_sound_files 
      - output_1: data/reference_values/partial_df_1/XXX.csv (loudness, sharpness, roughness, TNR)
      - input_2: data/raw_sound_files_1s 
      - output_2: data/reference_values/partial_df_2/XXX.csv (SII)
   4. merge them into a single references.csv: build df2 on a per-file basis by placing each 1s SII value on the 2ms grid at time_index = i*500 (i = segment index) with NaN for all other frames, over the same frame count as the matching df1 file; then fix df1 by moving each value to its correct place on the 2ms grid (0 = 0ms, 1 = 2ms, 2 = 4ms, ...), keeping its native resolution (loudness/sharpness every 2ms, roughness every 100ms, TNR every 500ms) with NaN in between; save both grids as intermediate df1.csv / df2.csv and merge them per file+time_index
      - input: data/reference_values/partial_df_1, data/reference_values/partial_df_2
      - output: data/reference_values/references.csv (intermediate: df1.csv, df2.csv)
   5. median 1s-chunk: median of each parameter per time_index pooled across all source files, only for the first 1000ms; values keep their grid positions, they are NOT compacted
      - input: data/reference_values/references.csv
      - output: data/reference_values/median_1s_chunk.csv
   6. compute variance/statistics over the training-set segments (count, mean, std, min, quartiles, max per parameter)
      - input: data/reference_values
      - output: data/reference_values/variance.csv
3. **run the main.py script**
   1. train the model
      - output: DL_model/epochs
      - output: DL_model/losses
   2. run comparison
      - output: DL_model/comparison
4. **run the plot paper scripts (each in its own subfolder of "plot paper")**
   1. data_distribution: histograms of all parameter values plus mean/median 1000ms chunk plots, chunk CSVs and a combined spread plot with percentile bands
      - input: data/reference_values/references.csv
      - output: plot paper/data_distribution (histograms.png, mean/median_1000ms_chunk.csv + .png, average_1000ms_chunk.png, chunk_spread_1000ms.png)

      ![chunk_spread_1000ms](plot%20paper/data_distribution/chunk_spread_1000ms.png)

   2. reference_vs_prediction: reference vs predicted parameters per test segment (uses per default epoch 547)
      - input: data/test_set, data/reference_values/references.csv, DL_model/epochs
      - output: plot paper/reference_vs_prediction (reference_vs_prediction_combined.png, reference_vs_prediction_per_parameter.png)

      ![reference_vs_prediction_combined](plot%20paper/reference_vs_prediction/reference_vs_prediction_combined.png)
      ![reference_vs_prediction_per_parameter](plot%20paper/reference_vs_prediction/reference_vs_prediction_per_parameter.png)

   3. runtime_loudness: runtime benchmark of dB, LUFS and mosqito loudness_zwtv on synthetic white noise, plotted as log-scale boxplot
      - input: none (self-generated white noise)
      - output: plot paper/runtime_loudness/runtime comparison.png

      ![runtime comparison](plot%20paper/runtime_loudness/runtime%20comparison.png)

   4. LUFS K-Filter: K-weighting frequency response (ITU-R BS.1770-5)
      - input: none (hardcoded filter coefficients)
      - output: plot paper/LUFS K-Filter/k_weighting.pdf + .png

      ![k_weighting](plot%20paper/LUFS%20K-Filter/k_weighting.png)

   5. equal_loudness_contour: ISO 226:2003 equal-loudness contours (10-100 phon)
      - input: none (hardcoded ISO 226:2003 coefficients)
      - output: plot paper/equal_loudness_contour/equal_loudness_contours.pdf + .png

      ![equal_loudness_contours](plot%20paper/equal_loudness_contour/equal_loudness_contours.png)

   6. inference_time_DL_model: sweep inference time vs head width / # head stages on all detected devices (CPU, iGPU, external GPU); writes one combined + one per-device benchmark CSV, then auto-runs "plot results.py"
      - input: none (synthetic input, model built programmatically)
      - output: plot paper/inference_time_DL_model/VRAM vs Inferencetime.csv (+ one CSV per device), benchmark plots (PNGs)

      ![Time heatmap stacked](plot%20paper/inference_time_DL_model/Time%20heatmap_stacked.png)

   7. plot results.py: standalone plotting of the CSVs from 4.6
      - input: plot paper/inference_time_DL_model/VRAM vs Inferencetime.csv
      - output: plot paper/inference_time_DL_model (benchmark plots)
5. **run the live demo (Demo_new/demo_new.py)**
   - loads the trained checkpoint `Demo_new/epoch_0483.pt` (falls back to `DL_model/epochs/epoch_0483.pt`) and seeds each parameter head's temporal bias from `data/reference_values/median_1s_chunk.csv`, so the model starts from the dataset's median 1 s curve
   - captures audio in real time from a `sounddevice` input (mic or virtual cable like VoiceMeeter; defaults to device index 1) and runs on the fastest available device (CUDA GPU → DirectML → CPU)
   - processes a sliding 1 s window (48 kHz) with a 20 ms hop: each window is fed through the model once and all five psychoacoustic parameters are estimated
   - plots live in a dark-themed 5-panel window (loudness, sharpness, roughness, TNR, SII): the faint grey cloud is the raw per-window prediction, the colored line is the current filter output
   - switch the output filter on the fly with keys **1-5** (1=direct keep-newest, 2=direct overwrite, 3=impulse 35 ms, 4=fast 125 ms, 5=slow 1000 ms)
   - auto-saves every finished 20 s window as a PNG and prints per-step timing averages (file load / model inference / plot update)
   - input: live audio from a sounddevice input device + `data/reference_values/median_1s_chunk.csv` + checkpoint
   - output: live plot, `Demo_new/example images/segment_Xs-Ys.png`, timing summary

   Example live-demo output (one 20 s window each):
   ![segment_0.0s-20.0s](Demo_new/example%20images/segment_0.0s-20.0s.png)
  