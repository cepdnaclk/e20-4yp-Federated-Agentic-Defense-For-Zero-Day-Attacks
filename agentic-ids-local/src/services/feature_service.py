# services/feature_service.py
import pandas as pd

class FeatureService:
    def __init__(self, rename_map, training_columns, categorical_cols, onehot_cols, numeric_cols, scaler,encoder):
        self.rename_map = rename_map
        self.training_columns = training_columns
        self.categorical_cols = categorical_cols
        self.onehot_cols = onehot_cols
        self.numeric_cols = numeric_cols
        self.scaler = scaler
        self.encoder=encoder

    def preprocess(self, df: pd.DataFrame):
        
        # 1. Rename
        df = df.rename(columns=self.rename_map)

        # 2. Derived features
        if "rate" in self.training_columns and "rate" not in df.columns:
            df["rate"] = (df["spkts"] + df["dpkts"]) / df["dur"].clip(lower=1e-6)

        # 3. One-hot categorical
        X_cat = self.encoder.transform(df[self.categorical_cols])
        X_cat = pd.DataFrame(
            X_cat,
            columns= self.encoder.get_feature_names_out(self.categorical_cols)
        )

        # 4. Numeric
        X_num = df[self.numeric_cols]

        # 5. Combine
        X_final = pd.concat([X_num, X_cat], axis=1)

        # 7. Scale
        X_scaled = self.scaler.transform(X_final)

        return X_scaled
