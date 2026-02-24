import numpy as np


class GaussianAffineSkew:
    """
    Simula attribute skew mediante transformación afín Gaussiana.
    
    X' = A * X + b
    
    Solo se aplica a variables numéricas.
    """

    def __init__(self,
                 sigma_scale: float = 0.1,
                 sigma_shift: float = 0.1,
                 seed: int | None = None):
        """
        sigma_scale : desviación estándar del factor multiplicativo
        sigma_shift : desviación estándar del término aditivo
        seed        : para reproducibilidad
        """
        self.sigma_scale = sigma_scale
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def transform(self, X: np.ndarray, numeric_idx: list[int]) -> np.ndarray:
        """
        Aplica la transformación afín solo a columnas numéricas.
        
        X : array (n_samples, n_features)
        numeric_idx : índices de columnas numéricas
        """
        X_new = X.copy()

        scale_factors = self.rng.normal(
            loc=1.0,
            scale=self.sigma_scale,
            size=len(numeric_idx)
        )

        shift_terms = self.rng.normal(
            loc=0.0,
            scale=self.sigma_shift,
            size=len(numeric_idx)
        )

        for i, col in enumerate(numeric_idx):
            X_new[:, col] = (
                X_new[:, col] * scale_factors[i] + shift_terms[i]
            )

        return X_new