"""Recorder: 모델·파라미터·지표·예측값 저장소"""
import os, joblib, pandas as pd
from sklearn.metrics import r2_score


class Recorder:
    def __init__(self, targets):
        self.targets = targets
        self.best_model = {}
        self.best_param = {}
        self.train_r2   = {}
        self.oos_pred   = {}

    # ── 저장 메서드 ───────────────────────────────────────
    def save_model(self, ds, model, param):
        self.best_model[ds] = model
        self.best_param[ds] = param

    def save_train_r2(self, ds, y_true, y_pred):
        r2 = {c: r2_score(y_true[c], y_pred[:, i])
              for i, c in enumerate(self.targets)}
        self.train_r2[ds] = pd.DataFrame([r2], index=[ds])

    def save_pred(self, ds, y_hat):
        self.oos_pred[ds] = pd.DataFrame([y_hat], index=[ds], columns=self.targets)

    
    # ── 직렬화 / 복원 메서드 ───────────────────────────────
    def dump(self, filepath: str, compress: bool = True):
        """
        모든 결과(best_model 제외 옵션 가능)를 피클 파일로 저장.

        Parameters
        ----------
        filepath : str
            저장할 *.pkl* 파일 경로.
        compress : bool, default True
            joblib 압축 사용 여부(.gz). 대형 모델이면 True 권장.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "best_model": self.best_model,     # 필요 없으면 제거 가능
            "best_param": self.best_param,
            "train_r2":   self.train_r2,
            "oos_pred":   self.oos_pred,
            "targets":    self.targets,
        }
        joblib.dump(data, filepath, compress=3 if compress else 0)

    @classmethod
    def load(cls, filepath: str) -> "Recorder":
        """
        저장된 Recorder 객체를 복원해 반환.
        """
        data = joblib.load(filepath)
        rec  = cls(data["targets"])
        rec.best_model = data["best_model"]
        rec.best_param = data["best_param"]
        rec.train_r2   = data["train_r2"]
        rec.oos_pred   = data["oos_pred"]
        return rec