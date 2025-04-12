import pandas as pd
from tqdm import tqdm


class TimeWindowSegmenter:
    def __init__(self, df_path, window_size=10, sampling_rate=20, step_size=10, time_column='Timestamp', id_column='Subject-id', activity_column='Activity Label'):
        self.window_size = window_size  # in seconds
        self.step_size = step_size      # in seconds
        self.time_column = time_column
        self.id_column = id_column
        self.activity_column = activity_column
        self.df = pd.read_parquet(df_path)
        self.fix_timestamps()
        self.sampling_rate = sampling_rate


    def _fix_unix_timestamp(self, timestamp_series):
        """
        Konwertuje nietypowy timestamp: zakładamy, że milisekundy zaczynają się od 7. cyfry od końca.
        Jeśli timestamp ma więcej niż 10 cyfr, obcinamy do pierwszych 13 (milisekundy).
        """
        ts_str = timestamp_series.astype(str).str.replace(r'\D', '', regex=True)  # na wypadek dziwnych znaków

        def safe_convert(ts):
            return pd.to_datetime(int(ts))
            

        return ts_str.apply(safe_convert)

    def fix_timestamps(self):
        print("fix timestamps")
        self.df[self.time_column] = self._fix_unix_timestamp(self.df[self.time_column])
        self.df.sort_values(by=[self.id_column, self.activity_column, self.time_column], inplace=True)
        print("end of fixing timestamps")

    def segment(self):
        """
        usage: 
        ```python
            for segment in TimeWindowSegmenter.segment():
                print(segment)
        ```   

        Yields:
            _type_: dataframe - window of data by time
        """
        grouped = self.df.groupby([self.id_column, self.activity_column])

        for (_, _), group in tqdm(grouped, total=len(grouped), desc="Segmenting"):
            for window in group.rolling(window=self.window_size*self.sampling_rate, step=self.step_size*self.sampling_rate, on=self.time_column):
                if window.shape[0] != self.window_size * self.sampling_rate:
                    continue
                yield window
            