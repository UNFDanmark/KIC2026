from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import io

mpl.rcParams['animation.embed_limit'] = 100


def gradient_linje(a, b, punkter):
    """Gradienten af squared loss for linjen y = a·x + b, mht. a og b. Bruges til opg5's gradient descent."""
    grad_a, grad_b = 0.0, 0.0
    for x, y in punkter:
        err    = 2 * (a * x + b - y)
        grad_a += err * x
        grad_b += err
    return grad_a, grad_b


def closed_form_linje(punkter):
    """
    Analytisk mindste-kvadraters løsning for y = a·x + b — "de normale ligninger",
    w = (XᵀX)⁻¹Xᵀy, samme lineære algebra som opgave 1_4. Bruges til at sammenligne
    med den linje jeres egen gradient descent finder.
    """
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)
    A = np.column_stack([xs, np.ones_like(xs)])
    w = np.linalg.inv(A.T @ A) @ A.T @ ys
    return float(w[0]), float(w[1])


# ── Delte hjælpefunktioner til loss-flade og fejl-kvadrater ────────────────

def _kurve_label(params, prefix=""):
    """Formatterer 'y = a·x + b' eller 'y = a·x² + b·x + c' afhængig af antal parametre."""
    try:
        if len(params) == 2:
            a, b = params
            sign = '+' if b >= 0 else '-'
            return f"{prefix}: y = {a:.3f}·x {sign} {abs(b):.3f}"
        a, b, c = params
        return f"{prefix}: y = {a:.3f}·x² + {b:.3f}·x + {c:.3f}"
    except Exception:
        return f"{prefix}: {params}"


def _eval_kurve(params, x):
    """Evaluerer linjen (2 params) eller parablen (3 params) i x — x må gerne være et numpy-array."""
    if len(params) == 2:
        a, b = params
        return a * x + b
    a, b, c = params
    return a * x**2 + b * x + c


def _tegn_matrix_bracket(ax, x, y_top, y_bot, retning=1, tick=0.15, lw=1.6):
    """Tegner en kantet-parentes-side ('[' hvis retning=1, ']' hvis retning=-1) — ren matplotlib,
    ingen afhængighed af LaTeX/usetex."""
    ax.plot([x, x], [y_top, y_bot], color='black', linewidth=lw, solid_capstyle='butt')
    ax.plot([x, x + retning * tick], [y_top, y_top], color='black', linewidth=lw, solid_capstyle='butt')
    ax.plot([x, x + retning * tick], [y_bot, y_bot], color='black', linewidth=lw, solid_capstyle='butt')


def _matrix_hojde(M, cell_h=0.55):
    """Hvor høj matricen M bliver når den tegnes med _tegn_matrix (bruges til at placere rækker)."""
    M = np.array(M, dtype=object)
    rows = M.shape[0] if M.ndim > 1 else len(M)
    return rows * cell_h


def _tegn_matrix(ax, x, y, label, M, fmt="{:.2f}", cell_w=1.05, cell_h=0.55, fontsize=11):
    """
    Tegner matricen M (1D vises som søjlevektor) centreret om (x, y) som venstre kant.
    fmt=None betyder symbolske entries (fx bogstaver som 'a'/'b') — de vises som de er,
    uden talformattering. Returnerer x-positionen lige efter matricen, så flere matricer
    og symboler kan sættes op i forlængelse af hinanden — vandret i samme kald, lodret
    via y (se linalg_losning).
    """
    M = np.array(M, dtype=object) if fmt is None else np.array(M, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows, cols = M.shape
    height = rows * cell_h
    width  = cols * cell_w
    pad    = cell_w * 0.35
    y_top, y_bot = y + height / 2, y - height / 2

    for i in range(rows):
        for j in range(cols):
            cx = x + pad + j * cell_w + cell_w / 2
            cy = y_top - i * cell_h - cell_h / 2
            tekst = str(M[i, j]) if fmt is None else fmt.format(M[i, j])
            ax.text(cx, cy, tekst, ha='center', va='center', fontsize=fontsize)

    x_right = x + width + 2 * pad
    _tegn_matrix_bracket(ax, x,       y_top, y_bot, retning=1)
    _tegn_matrix_bracket(ax, x_right, y_top, y_bot, retning=-1)
    ax.text((x + x_right) / 2, y_top + 0.35, label, ha='center', fontsize=fontsize + 1)
    return x_right


def _tegn_symbol(ax, x, y, symbol, bredde=0.7, fontsize=18):
    """Tegner et regneoperator-symbol ('·', '=', ...) mellem to matricer, se linalg_losning."""
    ax.text(x + bredde / 2, y, symbol, ha='center', va='center', fontsize=fontsize)
    return x + bredde


def _loss_grid(loss_fn, a_range, b_range, resolution=60):
    """Bygger (A, B, L) meshgrid ved at evaluere loss_fn(a, b) over gridet."""
    a_vals = np.linspace(a_range[0], a_range[1], resolution)
    b_vals = np.linspace(b_range[0], b_range[1], resolution)
    A, B = np.meshgrid(a_vals, b_vals)
    L = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            L[i, j] = loss_fn(A[i, j], B[i, j])
    return A, B, L


def _tegn_kontur(ax, A, B, L, colorbar=True):
    cs = ax.contourf(A, B, L, levels=30, cmap='viridis')
    if colorbar:
        # Bemærk: hvert kald skrumper ax'et lidt for at gøre plads til farveskalaen,
        # så ved gentagne redraws af samme ax (fx i animate()) bør colorbar=False bruges.
        plt.colorbar(cs, ax=ax, shrink=0.8)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title('Loss (kontur)')
    return cs


def _tegn_3d(ax, A, B, L):
    # matplotlib 3D-akser regner selv ud i hvilken rækkefølge artists tegnes (baseret på
    # kamera-afstand), så et scatter-punkt kan ende "begravet" i fladen selvom det har
    # højere zorder. computed_zorder=False slår det fra, så vi selv styrer rækkefølgen.
    ax.computed_zorder = False
    ax.plot_surface(A, B, L, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9, zorder=1)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('loss')
    ax.set_title('Loss (3D-flade)')


def _tegn_kvadrater_panel(ax, xs, ys, params, x_min=None, x_max=None, y_min=None, y_max=None, label_prefix="model"):
    """
    Tegner datapunkter, modellen (linje hvis params=(a,b), parabel hvis params=(a,b,c)),
    og et kvadrat per fejl. Kvadratets areal = fejlens bidrag til squared loss.
    Returnerer fejlene (y_pred - y) så den kaldende kode kan udregne loss.
    Angiv x_min/x_max/y_min/y_max for at låse aksernes skala på tværs af flere kald
    (ellers deformeres kvadraterne visuelt når fejlens størrelse ændrer sig).
    """
    y_preds = _eval_kurve(params, xs)
    errors  = y_preds - ys

    max_e = float(np.max(np.abs(errors))) if len(errors) else 1.0
    if x_min is None:
        x_min = xs.min() - 0.5
    if x_max is None:
        x_max = xs.max() + max_e + 0.5
    x_line = np.linspace(x_min, x_max, 300)

    ax.plot(x_line, _eval_kurve(params, x_line), color='royalblue', linewidth=2,
            label=_kurve_label(params, prefix=label_prefix), zorder=2)

    for x, y, y_pred, e in zip(xs, ys, y_preds, errors):
        side  = abs(float(e))
        y_bot = min(float(y), float(y_pred))
        if side > 1e-10:
            ax.add_patch(Rectangle(
                (x, y_bot), side, side,
                linewidth=1, edgecolor='tomato', facecolor='tomato', alpha=0.25,
                zorder=3,
            ))
            ax.text(x + side / 2, y_bot + side / 2, f'{e**2:.2f}',
                    ha='center', va='center', fontsize=8, color='darkred', zorder=5, clip_on=True)
        ax.plot([x, x], [y, y_pred], color='tomato', linewidth=1.5, zorder=4)

    ax.scatter(xs, ys, color='black', s=60, zorder=6, label='punkter')
    ax.set_xlim(x_min, x_max)
    if y_min is None or y_max is None:
        y_all = np.concatenate([ys, y_preds])
        pad   = 0.5
        y_min = y_all.min() - pad if y_min is None else y_min
        y_max = y_all.max() + pad if y_max is None else y_max
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return errors


# ── Komponerbare paneler: virker alene, eller samlet via display() ────────
# (samme idé som Maples plot(...) / display(p1, p2) — et panel viser sig selv
# hvis det står alene i en celle, eller kan lægges sammen med andre paneler.)

class _Panel:
    def __init__(self, tegn_fn, figsize=(6, 5), projection=None):
        self._tegn      = tegn_fn
        self.figsize    = figsize
        self.projection = projection

    def _repr_png_(self):
        fig = plt.figure(figsize=self.figsize)
        self._tegn(fig.add_subplot(1, 1, 1, projection=self.projection))
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return buf.getvalue()


def display(*paneler, figsize=None):
    """Vis flere paneler side om side, fx: sesy_viz.display(sesy_viz.loss_kontur(...), sesy_viz.loss_3d(...))"""
    n = len(paneler)
    fig = plt.figure(figsize=figsize or (6 * n, 5))
    for i, panel in enumerate(paneler):
        panel._tegn(fig.add_subplot(1, n, i + 1, projection=panel.projection))
    plt.tight_layout()
    plt.show()


def loss_kontur(loss_fn, a_range, b_range, resolution=60, punkt=None, path=None, gradient=None, colorbar=True):
    """
    Loss-flade som kontur-plot over (a, b). loss_fn skal være en funktion af (a, b) —
    typisk en lambda der wrapper en af jeres egne opg3/opg4-funktioner, fx:
        loss_fn = lambda a, b: opg3_2(a, b, punkter)
    Har din opgave 3 parametre (fx opg3_3/opg3_4), så fastlås den tredje i lambdaen:
        loss_fn = lambda a, b: opg3_3(a, b, c_fast, punkter)

    punkt: valgfri (a, b)-markør, fx den man selv har valgt at prøve.
    path: valgfri liste af (a, b)-punkter der tegnes som en sti med en markør ved det sidste
    (fx den vej gradient descent har gået indtil videre) — bruges i stedet for punkt.
    gradient: valgfri (grad_a, grad_b) — tegner en pil fra punkt/sti's sidste sted som
    den fulde vektor (grad_a, grad_b), altså i skala med (a, b)-akserne. Den rå gradient
    er ofte for stor til at se noget fornuftigt — skalér den selv (fx med jeres learning
    rate) før I sender den ind, ligesom I allerede gør ved selve vægtopdateringen.
    colorbar: sæt til False når panelet skal redraws gentagne gange på samme akse (fx i
    animate()) — ellers skrumper aksen lidt for hvert kald.
    Kør alene for at se plottet, eller giv den til display()/animate() sammen med andre paneler.
    """
    def tegn(ax):
        A, B, L = _loss_grid(loss_fn, a_range, b_range, resolution)
        _tegn_kontur(ax, A, B, L, colorbar=colorbar)
        ax.autoscale(False)  # markør/sti må ikke flytte akserne
        sted = None
        if path:
            a_vals = [p[0] for p in path]
            b_vals = [p[1] for p in path]
            ax.plot(a_vals, b_vals, color='white', linewidth=1.5, alpha=0.8)
            ax.plot([a_vals[-1]], [b_vals[-1]], 'o', color='tomato', markersize=10, zorder=5)
            sted = (a_vals[-1], b_vals[-1])
        elif punkt is not None:
            ax.plot([punkt[0]], [punkt[1]], 'o', color='tomato', markersize=10, zorder=5)
            sted = punkt
        if gradient is not None and sted is not None:
            grad_a, grad_b = gradient
            # tolerance i stedet for præcis 0-tjek: ved den optimale løsning er gradienten
            # matematisk 0, men flydende-komma-regning giver typisk noget i stil med 1e-14
            if abs(grad_a) > 1e-9 or abs(grad_b) > 1e-9:
                ax.annotate('', xy=(sted[0] + grad_a, sted[1] + grad_b), xytext=sted,
                             arrowprops=dict(facecolor='white', edgecolor='black',
                                              linewidth=1.5, width=3, headwidth=10, headlength=10),
                             zorder=7)
    return _Panel(tegn, figsize=(6, 5))


def loss_3d(loss_fn, a_range, b_range, resolution=60, punkt=None):
    """Samme loss-flade som loss_kontur, men som en 3D-flade. Se loss_kontur for detaljer om loss_fn."""
    def tegn(ax):
        A, B, L = _loss_grid(loss_fn, a_range, b_range, resolution)
        _tegn_3d(ax, A, B, L)
        if punkt is not None:
            a, b = punkt
            ax.scatter([a], [b], [loss_fn(a, b)], color='tomato', s=80, zorder=10,
                       depthshade=False, edgecolor='black', linewidth=0.5)
    return _Panel(tegn, figsize=(6, 5), projection='3d')


def modelfit(*args, x_range=None, y_range=None):
    """
    Vis modellen sammen med punkterne og fejl-kvadraterne — linjen y=a·x+b hvis I giver
    2 parametre, parablen y=a·x²+b·x+c hvis I giver 3. Kald fx som:
        sesy_viz.modelfit(a, b, punkter)
        sesy_viz.modelfit(a, b, c, punkter)
    Angiv x_range/y_range for at låse aksernes skala på tværs af flere kald med
    forskellige parametre — ellers deformeres kvadraterne når fejlens størrelse ændrer sig.
    Kør alene for at se plottet, eller giv den til display() sammen med andre paneler.
    """
    *params, punkter = args
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)
    # uden eksplicit x_range/y_range: lad _tegn_kvadrater_panel selv beregne plads til
    # fejl-kvadraterne (ellers bliver de skåret af når fejlen er stor)
    x_min, x_max = x_range if x_range else (None, None)
    y_min, y_max = y_range if y_range else (None, None)

    def tegn(ax):
        errors = _tegn_kvadrater_panel(ax, xs, ys, params, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        ax.set_aspect('equal', adjustable='box')
        loss = float(np.sum(errors**2))
        label = '  '.join(f'{navn}={v:.2f}' for navn, v in zip('abc', params))
        ax.set_title(f'{label}   loss={loss:.2f}')
        ax.legend(fontsize=9)
    return _Panel(tegn, figsize=(6, 5))


def linalg_losning(A, y, AtA, Aty, AtA_inv, w, fmt="{:.2f}"):
    """
    Visualiserer regnestykket bag den analytiske løsning, byggesten for byggesten —
    I bygger selv A, y, AᵀA, Aᵀy og w i cellen (se opgave 1_4/closed_form_linje for
    samme lineære algebra), denne funktion tegner dem bare op som matricer:
        Aᵀ · A = AᵀA
        Aᵀ · y = Aᵀy
        (AᵀA)⁻¹ · Aᵀy = w
    Alle entries i A, Aᵀ og y er synlige — det eneste der er en "sort boks" er selve
    matrix-inversionen (AᵀA)⁻¹, resten er bare matrix-multiplikation I kan tjekke tal for tal.
    Kør alene for at se plottet, eller giv den til display() sammen med andre paneler.
    """
    At = np.array(A, dtype=float).T
    rows_spec = [
        [("matrix", "Aᵀ", At), ("symbol", "·"), ("matrix", "A", A), ("symbol", "="), ("matrix", "AᵀA", AtA)],
        [("matrix", "Aᵀ", At), ("symbol", "·"), ("matrix", "y", y), ("symbol", "="), ("matrix", "Aᵀy", Aty)],
        [("matrix", "(AᵀA)⁻¹", AtA_inv), ("symbol", "·"), ("matrix", "Aᵀy", Aty), ("symbol", "="), ("matrix", "w", w),
         ("symbol", "="), ("matrix", "", ["a", "b"], None)],
    ]

    def tegn(ax):
        ax.axis('off')
        margin  = 0.85
        højder  = [max(_matrix_hojde(item[2]) for item in row if item[0] == "matrix")
                   for row in rows_spec]
        y_cursor = sum(højder) / 2 + margin * (len(rows_spec) - 1) / 2
        centre = []
        for h in højder:
            y_cursor -= h / 2
            centre.append(y_cursor)
            y_cursor -= h / 2 + margin

        bredde = 0.0
        for row, yc in zip(rows_spec, centre):
            x = 0.0
            for item in row:
                if item[0] == "matrix":
                    item_fmt = item[3] if len(item) > 3 else fmt
                    x = _tegn_matrix(ax, x, yc, item[1], item[2], item_fmt)
                else:
                    x = _tegn_symbol(ax, x, yc, item[1])
            bredde = max(bredde, x)

        ax.set_xlim(-0.3, bredde + 0.3)
        ax.set_ylim(centre[-1] - højder[-1] / 2 - 0.4, centre[0] + højder[0] / 2 + 0.4)
    return _Panel(tegn, figsize=(11, 6.5))


def loss_over_tid(losses):
    """Viser loss pr. skridt som en kurve, med en markør ved det seneste skridt."""
    def tegn(ax):
        ax.plot(range(len(losses)), losses, color='lightgray', linewidth=1.5)
        ax.plot([len(losses) - 1], [losses[-1]], 'o', color='tomato', markersize=8, zorder=5)
        ax.set_xlabel('skridt')
        ax.set_ylabel('loss')
        ax.set_title('Squared loss over tid')
    return _Panel(tegn, figsize=(6, 4.5))


def animate(paneler_per_frame, interval=120, figsize=None):
    """
    Animér en liste af panel-tupler, én tuple pr. frame — genbruger de samme
    panel-funktioner som display(), fx:
        paneler_per_frame = [
            (modelfit(a0, b0, punkter, x_range=(0,5), y_range=(0,5)),
             loss_kontur(loss_fn, a_range=(-3,4), b_range=(-2,6), path=[(a0,b0)])),
            (modelfit(a1, b1, punkter, x_range=(0,5), y_range=(0,5)),
             loss_kontur(loss_fn, a_range=(-3,4), b_range=(-2,6), path=[(a0,b0),(a1,b1)])),
            ...
        ]
        sesy_viz.animate(paneler_per_frame)
    Hvert frame tegnes forfra med de(t) panel(er) man har bygget til lige netop det skridt.
    """
    n = len(paneler_per_frame[0])
    fig  = plt.figure(figsize=figsize or (6 * n, 5))
    axes = [fig.add_subplot(1, n, i + 1, projection=paneler_per_frame[0][i].projection) for i in range(n)]
    # oprindelig placering af hver ax — skal genskabes hvert frame, ellers skrumper fx
    # colorbars aksen lidt mere for hvert kald og den ender som en tynd streg
    positioner = [ax.get_position() for ax in axes]

    def _animate(i):
        for ax in list(fig.axes):       # ryd ekstra akser (fx colorbars) fra forrige frame
            if ax not in axes:
                fig.delaxes(ax)
        for ax, panel, pos in zip(axes, paneler_per_frame[i], positioner):
            ax.clear()
            ax.set_position(pos)
            panel._tegn(ax)
        return axes

    anim = FuncAnimation(fig, _animate, frames=len(paneler_per_frame), interval=interval, blit=False)
    plt.tight_layout()
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


# ── Feedback-plots til regression_test.py ─────────────────────────────────

def feedback_kurve(punkter, got_params, exp_params):
    """
    Feedback for opg1/opg2: viser datapunkter, studentens kurve (rød stiplet)
    og facit-kurven (grøn).
    Bruges ved fejl i linje- og parabeL-opgaver.
    """
    xs = [p[0] for p in punkter]
    ys = [p[1] for p in punkter]
    pad    = max((max(xs) - min(xs)) * 0.2, 0.5)
    x_min  = min(xs) - pad
    x_max  = max(xs) + pad
    x_line = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, color='black', s=60, zorder=5, label='punkter')

    try:
        ax.plot(x_line, _eval_kurve(got_params, x_line), 'r--', linewidth=2,
                label=_kurve_label(got_params, prefix="din"))
    except Exception:
        pass

    ax.plot(x_line, _eval_kurve(exp_params, x_line), color='green', linewidth=2,
            label=_kurve_label(exp_params, prefix="korrekt"))

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def feedback_haeldning(f, x, got, exp, span=None):
    """
    Feedback for opg4's differentiations-opgaver: viser grafen for f (forskriften),
    punktet (x, f(x)), og hældningen som en pil igennem punktet — jeres svar (rød)
    mod facit (grøn). En forkert hældning ses direkte som en forkert vinkel på pilen.
    """
    x = float(x)
    y = float(f(x))
    if span is None:
        span = max(abs(x), 1.0) * 2.0 + 2.0
    xs = np.linspace(x - span, x + span, 300)
    ys = [float(f(xv)) for xv in xs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, ys, color='royalblue', linewidth=2, label='f(x)', zorder=2)
    ax.scatter([x], [y], color='black', s=60, zorder=5, label=f'punkt: x={x:.2g}')

    pad = (max(ys) - min(ys)) * 0.3 + 0.5
    ax.set_xlim(x - span, x + span)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.autoscale(False)  # pilene må ikke skubbe grafen ud af syne, selv hvis hældningen er meget forkert

    dx = span * 0.3
    for hældning, farve, navn in [(exp, 'green', 'korrekt'), (got, 'tomato', 'din')]:
        ax.annotate('', xy=(x + dx, y + hældning * dx), xytext=(x, y),
                     arrowprops=dict(facecolor=farve, edgecolor=farve, linewidth=2,
                                      width=2.5, headwidth=9, headlength=10), zorder=6)
        ax.plot([], [], color=farve, linewidth=2.5, label=f"{navn}: f'({x:.2g}) = {hældning:.3g}")

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def feedback_gradient_3d(f, x, y, got, exp, span=None, resolution=40):
    """
    Feedback for opg5's gradient-opgaver: viser fladen z=f(x,y) i 3D, punktet
    (x, y, f(x,y)), og gradienten som en pil i tangentplanen — jeres svar (rød)
    mod facit (grøn). Den korrekte pil følger fladen tæt omkring punktet;
    en forkert gradient peger i en forkert retning/hældning oven på fladen.
    got og exp er hver (df/dx, df/dy).
    """
    x, y = float(x), float(y)
    z = float(f(x, y))
    if span is None:
        span = max(abs(x), abs(y), 1.0) * 1.5 + 1.5

    xs = np.linspace(x - span, x + span, resolution)
    ys = np.linspace(y - span, y + span, resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.computed_zorder = False
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.75, zorder=1)
    ax.scatter([x], [y], [z], color='black', s=50, zorder=10, depthshade=False, edgecolor='black')

    d = span * 0.35
    for (gx, gy), farve, navn in [(exp, 'green', 'korrekt'), (got, 'tomato', 'din')]:
        dz = gx * d + gy * d  # stigningen langs tangentplanen i den retning
        ax.quiver(x, y, z, d, d, dz, color=farve, linewidth=2.5, arrow_length_ratio=0.2, zorder=11,
                   label=f"{navn}: ∇f = ({gx:.2g}, {gy:.2g})")

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def feedback_loss(model_params, punkter, got_loss, exp_loss):
    """
    Feedback for opg3/opg4: viser kurven, punkterne og et kvadrat per fejl.
    Kvadratets areal = fejlens bidrag til squared loss.
    Bruges ved fejl i loss-beregnings-opgaver. model_params er (a,b) eller (a,b,c).
    """
    xs = np.array([p[0] for p in punkter], dtype=float)
    ys = np.array([p[1] for p in punkter], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    _tegn_kvadrater_panel(ax, xs, ys, model_params)
    ax.set_title(
        f'Kvadraternes arealer summer til: {got_loss:.4f}\n'
        f'Korrekt loss = {exp_loss:.4f}'
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


