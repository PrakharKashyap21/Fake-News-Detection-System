from typing import Optional
from backend.app.predictor import get_predictor, NewsPredictor
from backend.app.v2.schemas import LinguisticSignal


class SVMSignalProvider:
    """Wraps the V1 LinearSVC + TF-IDF model to provide stylistic/linguistic pattern scores."""

    def __init__(self):
        self._predictor: Optional[NewsPredictor] = None

    def _get_predictor(self) -> NewsPredictor:
        if self._predictor is None:
            self._predictor = get_predictor()
        return self._predictor

    def get_signal(self, title: str = "", text: str = "") -> LinguisticSignal:
        """Evaluates the linguistic pattern score using the V1 LinearSVC model."""
        predictor = self._get_predictor()
        res = predictor.predict(title=title, text=text)
        return LinguisticSignal(
            label=res["label"],
            prediction=res["prediction"],
            confidence=res["confidence"],
            message=res["message"]
        )


_svm_signal_instance: Optional[SVMSignalProvider] = None


def get_svm_signal_provider() -> SVMSignalProvider:
    global _svm_signal_instance
    if _svm_signal_instance is None:
        _svm_signal_instance = SVMSignalProvider()
    return _svm_signal_instance
