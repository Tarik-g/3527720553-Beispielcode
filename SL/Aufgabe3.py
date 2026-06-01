"""
Aufgabe 3: Restricted Boltzmann Machine (RBM) als Autoencoder für MNIST
========================================================================
- Eingang: Handgeschriebene Ziffern (8x8 = 64 sichtbare Neuronen)
- Experiment A: 32 Hidden Neuronen
- Experiment B: 10 Hidden Neuronen (stark reduziert = starker Bottleneck)
- Rekonstruktion durch Forward- und Backward-Pass (CD-1)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════
# 1.  RBM – Klasse
# ═══════════════════════════════════════════════════════

class RBM:
    """
    Restricted Boltzmann Machine mit binären stochastischen Neuronen.
    Training via Contrastive Divergence-1 (CD-1).

    Architektur:
        v  ←──W──→  h
    Sichtbare (v_dim) und versteckte (h_dim) Schicht sind
    vollständig verbunden; keine internen Verbindungen (»restricted«).
    """

    def __init__(self, n_visible: int, n_hidden: int,
                 learning_rate: float = 0.05,
                 n_epochs: int = 50,
                 batch_size: int = 32,
                 seed: int = 42):
        self.n_visible  = n_visible
        self.n_hidden   = n_hidden
        self.lr         = learning_rate
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        rng = np.random.RandomState(seed)

        # Gewichte (Xavier-Initialisierung)
        scale = np.sqrt(2.0 / (n_visible + n_hidden))
        self.W  = rng.normal(0, scale, (n_visible, n_hidden))
        self.bv = np.zeros(n_visible)   # Bias der sichtbaren Schicht
        self.bh = np.zeros(n_hidden)    # Bias der versteckten Schicht

        self.train_errors = []
        self.label = f"RBM ({n_visible}→{n_hidden}→{n_visible})"

    # ── Hilfsfunktionen ─────────────────────────────────
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _sample(self, probs):
        return (np.random.random(probs.shape) < probs).astype(np.float32)

    # ── Forward-Pass:  v → p(h|v) ───────────────────────
    def encode(self, v):
        """Aktivierungswahrsch. der Hidden-Schicht (Encode)."""
        ph = self._sigmoid(v @ self.W + self.bh)
        return ph

    # ── Backward-Pass: h → p(v|h) ───────────────────────
    def decode(self, h):
        """Rekonstruktionswahrsch. der sichtbaren Schicht (Decode)."""
        pv = self._sigmoid(h @ self.W.T + self.bv)
        return pv

    # ── Vollständiger Autoencoder-Pass ───────────────────
    def reconstruct(self, v):
        """Deterministischer Encode→Decode-Pass (keine Stochastik)."""
        return self.decode(self.encode(v))

    # ── CD-1 Training ────────────────────────────────────
    def fit(self, X, verbose=True):
        n = X.shape[0]
        if verbose:
            print(f"\n{'═'*52}")
            print(f"  Training  {self.label}")
            print(f"  Epochen={self.n_epochs}  LR={self.lr}  Batch={self.batch_size}")
            print(f"{'═'*52}")

        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n)
            X_shuf = X[idx]
            ep_err = 0.0

            for s in range(0, n, self.batch_size):
                v0 = X_shuf[s : s + self.batch_size]
                b  = v0.shape[0]

                # ── Positive Phase ───────────────────
                ph0 = self.encode(v0)
                h0  = self._sample(ph0)

                # ── Negative Phase (1 Gibbs-Schritt) ─
                pv1 = self.decode(h0)
                v1  = self._sample(pv1)
                ph1 = self.encode(v1)

                # ── Gradienten (CD-1) ────────────────
                dW  = (v0.T @ ph0  -  v1.T @ ph1) / b
                dbv = (v0  - v1 ).mean(axis=0)
                dbh = (ph0 - ph1).mean(axis=0)

                # ── Momentum-freier SGD ──────────────
                self.W  += self.lr * dW
                self.bv += self.lr * dbv
                self.bh += self.lr * dbh

                ep_err += np.mean((v0 - pv1) ** 2) * b

            mse = ep_err / n
            self.train_errors.append(mse)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"  Epoche {epoch+1:3d}/{self.n_epochs}  │  MSE = {mse:.5f}")

        if verbose:
            print(f"{'═'*52}\n")
        return self


# ═══════════════════════════════════════════════════════
# 2.  Daten laden
# ═══════════════════════════════════════════════════════

def load_data():
    """Lädt den sklearn-Digits-Datensatz (8×8 Pixel, 10 Klassen)."""
    digits = load_digits()
    X = minmax_scale(digits.data.astype(np.float32))   # → [0, 1]
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Daten geladen:  Training={X_train.shape}  Test={X_test.shape}  "
          f"Features={X.shape[1]} (8×8 Pixel)")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════
# 3.  Visualisierung
# ═══════════════════════════════════════════════════════

DARK_BG  = '#0d1117'
DARK_AX  = '#161b22'
COL_A    = '#58a6ff'   # blau  = RBM-A (32 Hidden)
COL_B    = '#f78166'   # rot   = RBM-B (10 Hidden)
COL_ORIG = '#3fb950'   # grün  = Original


def _digit_grid(data, ncols, ax_size=0.8):
    """Hilfsfunktion: n Bilder als Grid."""
    n = len(data)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * ax_size, nrows * ax_size))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(data[i].reshape(8, 8), cmap='hot',
                      vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
    return fig, axes


def plot_comparison(originals, recon_a, recon_b, labels, n=20):
    """3-Reihen-Vergleich: Original / 32 Hidden / 10 Hidden."""
    ncols = n
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 1.0, 4.0))
    fig.patch.set_facecolor(DARK_BG)

    row_labels = ['Original', '32 Hidden', '10 Hidden']
    row_colors = [COL_ORIG, COL_A, COL_B]
    rows_data  = [originals, recon_a, recon_b]

    for r, (rdata, rlabel, rcolor) in enumerate(
            zip(rows_data, row_labels, row_colors)):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(rdata[c].reshape(8, 8), cmap='hot',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(rlabel, color=rcolor, fontsize=9,
                              rotation=90, labelpad=4)
                ax.yaxis.set_label_position('left')
            if r == 0:
                ax.set_title(str(labels[c]), color='white', fontsize=8, pad=2)

    fig.suptitle(
        "RBM Autoencoder – Rekonstruktionsvergleich\n"
        "(32 vs. 10 Hidden Neuronen)",
        color='white', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.3)
    return fig


def plot_learning_curves(rbm_a, rbm_b):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)

    ep_a = range(1, len(rbm_a.train_errors) + 1)
    ep_b = range(1, len(rbm_b.train_errors) + 1)

    ax.plot(ep_a, rbm_a.train_errors, color=COL_A, lw=2.2,
            label=f'32 Hidden  (Final-MSE={rbm_a.train_errors[-1]:.4f})',
            marker='o', markersize=3)
    ax.plot(ep_b, rbm_b.train_errors, color=COL_B, lw=2.2, ls='--',
            label=f'10 Hidden  (Final-MSE={rbm_b.train_errors[-1]:.4f})',
            marker='s', markersize=3)
    ax.fill_between(ep_a, rbm_a.train_errors, alpha=0.10, color=COL_A)
    ax.fill_between(ep_b, rbm_b.train_errors, alpha=0.10, color=COL_B)

    ax.set_xlabel('Epoche', color='white', fontsize=11)
    ax.set_ylabel('MSE (Rekonstruktionsfehler)', color='white', fontsize=11)
    ax.set_title('Lernkurven: CD-1 Training', color='white',
                 fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#444')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(facecolor='#0d1117', labelcolor='white', fontsize=10)
    ax.grid(alpha=0.12, color='white')
    plt.tight_layout()
    return fig


def plot_weights(rbm, title, size=8):
    """Gelernte Gewichte als Bild-Filter."""
    n = rbm.n_hidden
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * size/8, rows * size/8))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            w = rbm.W[:, i].reshape(8, 8)
            vmax = np.abs(w).max() or 1
            ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.axis('off')
    fig.suptitle(title, color='white', fontsize=11, fontweight='bold')
    plt.tight_layout(pad=0.2)
    return fig


def plot_architecture():
    """Schematische Darstellung der RBM-Architektur."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis('off')

    # Schichten
    layers = [
        (1.0, 'Visible\n(Eingang)', '784 / 64\nNeuronen', '#3fb950'),
        (5.0, 'Hidden\n(Bottleneck)', '32 oder\n10 Neuronen', '#58a6ff'),
        (9.0, 'Visible\n(Ausgang)', '784 / 64\nNeuronen', '#f78166'),
    ]
    for x, title, sub, color in layers:
        rect = plt.Rectangle((x - 0.6, 0.8), 1.2, 2.4,
                              facecolor=color + '22', edgecolor=color,
                              linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, 3.5, title, ha='center', va='center',
                color=color, fontsize=10, fontweight='bold')
        ax.text(x, 2.0, sub, ha='center', va='center',
                color='white', fontsize=9)

    # Pfeile
    for x1, x2, lbl, col in [
            (1.6, 4.4, 'Encode\nv → h', COL_A),
            (5.6, 8.4, 'Decode\nh → v', COL_B)]:
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2))
        ax.text((x1 + x2)/2, 2.4, lbl, ha='center',
                color=col, fontsize=9)

    ax.set_title('RBM als Autoencoder – Architektur',
                 color='white', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════
# 4.  Hauptprogramm
# ═══════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    out = '/mnt/user-data/outputs'

    # ── Daten ──────────────────────────────────────────
    X_train, X_test, _, y_test = load_data()
    n_vis = X_train.shape[1]   # 64 (8×8)

    # ── Architektur-Diagramm ───────────────────────────
    fig0 = plot_architecture()
    fig0.savefig(f'{out}/rbm_architektur.png',
                 dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig0)

    # ── Experiment A: 32 Hidden Neuronen ──────────────
    print("\n▶ Experiment A: 32 Hidden Neuronen")
    rbm_a = RBM(n_visible=n_vis, n_hidden=32,
                learning_rate=0.05, n_epochs=80, batch_size=32)
    rbm_a.fit(X_train)
    recon_a = rbm_a.reconstruct(X_test)
    mse_a   = np.mean((X_test - recon_a) ** 2)
    print(f"  Test-MSE (32 Hidden): {mse_a:.5f}")

    # ── Experiment B: 10 Hidden Neuronen ──────────────
    print("\n▶ Experiment B: 10 Hidden Neuronen")
    rbm_b = RBM(n_visible=n_vis, n_hidden=10,
                learning_rate=0.05, n_epochs=80, batch_size=32)
    rbm_b.fit(X_train)
    recon_b = rbm_b.reconstruct(X_test)
    mse_b   = np.mean((X_test - recon_b) ** 2)
    print(f"  Test-MSE (10 Hidden): {mse_b:.5f}")

    # ── Plots ──────────────────────────────────────────
    print("\nErstelle Plots …")

    # Rekonstruktions-Vergleich (erste 20 Testbilder)
    fig1 = plot_comparison(X_test[:20], recon_a[:20], recon_b[:20],
                           y_test[:20], n=20)
    fig1.savefig(f'{out}/rbm_rekonstruktion.png',
                 dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig1)

    # Lernkurven
    fig2 = plot_learning_curves(rbm_a, rbm_b)
    fig2.savefig(f'{out}/rbm_lernkurven.png',
                 dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig2)

    # Gelernte Filter
    fig3 = plot_weights(rbm_a, "Gelernte Gewichte – 32 Hidden Neuronen")
    fig3.savefig(f'{out}/rbm_gewichte_32.png',
                 dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig3)

    fig4 = plot_weights(rbm_b, "Gelernte Gewichte – 10 Hidden Neuronen")
    fig4.savefig(f'{out}/rbm_gewichte_10.png',
                 dpi=140, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig4)

    # ── Zusammenfassung ────────────────────────────────
    print("\n" + "═"*52)
    print("  ZUSAMMENFASSUNG")
    print("═"*52)
    print(f"  Test-MSE (32 Hidden):  {mse_a:.5f}   Kompression: {n_vis}→32")
    print(f"  Test-MSE (10 Hidden):  {mse_b:.5f}   Kompression: {n_vis}→10")
    diff = (mse_b - mse_a) / mse_a * 100
    print(f"  Fehler-Anstieg:        +{diff:.1f}%")
    print()
    print("  Interpretation:")
    print("  • Mit 32 Hidden Neuronen ist die Rekonstruktion gut")
    print("    erkennbar – der Autoencoder komprimiert 64→32 Bit.")
    print("  • Mit nur 10 Hidden Neuronen (64→10→64) geht Schärfe")
    print("    verloren, aber Ziffern bleiben erkennbar – die RBM")
    print("    hat die wichtigsten statistischen Merkmale gelernt.")
    print("  • Möglich, weil MNIST-Ziffern niederdimensionale")
    print("    Struktur in einem hochdimensionalen Raum besitzen.")
    print("═"*52)


if __name__ == "__main__":
    main()