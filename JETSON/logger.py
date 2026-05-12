# logger.py
import pandas as pd, time, os

class Logger:
    def __init__(self, csv_file='exp.csv'):
        self.rows=[]
        self.csv_file=csv_file

    def log(self, d):
        d['ts'] = time.time()
        self.rows.append(d)
        if len(self.rows)>=50:
            self.flush()

    def flush(self):
        if not self.rows: return
        df = pd.DataFrame(self.rows)
        header = not os.path.exists(self.csv_file)
        df.to_csv(self.csv_file, mode='a', index=False, header=header)
        try:
            pd.read_csv(self.csv_file).to_excel(self.csv_file.replace('.csv','.xlsx'), index=False)
        except Exception:
            pass
        self.rows=[]
