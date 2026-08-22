# Real Dataset Sources

If the dataset requires it, place its source file in `data/source` before running the matching `scripts/bash_scripts/real_experiments/{dataset}/generate_split_data.sh` script.

| Dataset | Underlying source | Expected source file(s) |
| --- | --- | --- |
| `cars` | Kaggle Used Cars Dataset by Austin Reese: <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data> (Version 10, last accessed: 2026-08-08)| `vehicles.csv` |
| `spotify` | TidyTuesday Spotify Songs Dataset: <https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-01-21/spotify_songs.csv> (last accessed: 2026-08-08) | `spotify_songs.csv` |
| `cows` | Rdatasets copy of `nlme::Milk`: <https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Milk.csv> (last accessed: 2026-08-08) | `nlme_Milk.csv` |
| `bike` | UCI Seoul Bike Sharing Demand, dataset 560: <https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand> (last accessed: 2026-08-08)| None, script will automatically fetch from uci, but then stores a copy `SeoulBikeData.csv` |


**IMPORTANT**  
For the `bike` dataset the temporal few-shot splits will not be generated correctly with the `generate_split_data.sh` / `scripts/data/generate_data.py`.
Instead, after running this data generation script for the `bike` dataset, please run `notebooks/data/real/bike_temporal_few_shot_splits.ipynb`, which then overwrites the few-shot data with the correct temporal few-shot splits.
