import numpy as np

def generate_correlated_boston_like_data(
    m: int = 2000,
    n: int = 2000,
    seed: int = 42,
    noise_std: float = 2.0,
    rho_block: float = 0.9,
    cond_number: float = 1e4,
    sparsity: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera un design matrix A ∈ R^{m×n} con numero di feature n=2000,
    un condizionamento controllato e blocchi di correlazione, e un vettore
    di coefficienti veri x_true sparsi di dimensione n.

    Parametri
    ----------
    m : int
        Numero di esempi.
    n : int
        Numero di feature.
    seed : int
        Seed del RNG.
    noise_std : float
        Deviazione standard del rumore su b.
    rho_block : float
        Correlazione interna ai blocchi di feature (tra coppie successive).
    cond_number : float
        Condizionamento desiderato di A (rapporto tra σ_max e σ_min).
    sparsity : int
        Numero di feature non-zero in x_true.

    Ritorna
    -------
    A : np.ndarray, shape (m, n)
        Design matrix con spectrum decrescente log-spaziato.
    b : np.ndarray, shape (m,)
        Output A @ x_true + rumore.
    x_true : np.ndarray, shape (n,)
        Vettore dei coefficienti, sparso con `sparsity` non-zero.
    """
    rng = np.random.default_rng(seed)

    # 1) genera matrici ortogonali U, V via QR di gaussiane
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # 2) singular values log-spaziati tra cond_number e 1
    s = np.logspace(0, -np.log10(cond_number), num=min(m, n))

    # 3) costruisci A = U Σ Vᵀ
    Σ = np.zeros((m, n))
    Σ[np.arange(len(s)), np.arange(len(s))] = s
    A = U @ Σ @ V.T

    # 4) aggiungi un po’ di correlazione a blocchi lungo le prime 100 feature
    #    (ogni coppia (2i,2i+1) ha correlazione rho_block)
    for i in range(0, 100, 2):
        block = rng.multivariate_normal([0, 0],
                                        [[1.0, rho_block], [rho_block, 1.0]],
                                        size=m)
        A[:, i  ] = block[:, 0]
        A[:, i+1] = block[:, 1]

    # 5) crea x_true sparso
    x_true = np.zeros(n)
    # posizioni casuali dei non-zero
    idx = rng.choice(n, size=sparsity, replace=False)
    x_true[idx] = rng.uniform(-10, 10, size=sparsity)

    # 6) genera b con rumore
    noise = rng.normal(0, noise_std, size=m)
    b = A @ x_true + noise

    return A, b, x_true
