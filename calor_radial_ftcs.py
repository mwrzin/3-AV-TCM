"""
Simulacao transiente de calor radial 1D ao redor de um poco de petroleo.

Modelo fisico:
    dT/dt = alpha * (d2T/dr2 + (1/r) * dT/dr)

Metodo numerico:
    Diferencas Finitas Explicitas FTCS
    Forward-Time Central-Space em uma malha radial uniforme.

Contexto de engenharia:
    Este modelo representa, de forma simplificada, o aquecimento radial da
    formacao rochosa por uma parede de poco mantida a temperatura constante.
    Em garantia de escoamento, esse tipo de simulacao ajuda a visualizar
    como a energia termica se propaga na rocha ao longo do tempo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# =============================================================================
# 1. PARAMETROS FISICOS E NUMERICOS
# =============================================================================

# Geometria radial do problema [m]
r_w = 0.10  # raio do poco
r_e = 4.00  # raio externo do dominio de simulacao

# Propriedade termica efetiva da rocha [m2/s]
alpha = 1.0e-6

# Temperaturas de contorno e inicial [graus Celsius]
T_f = 40.0  # temperatura estatica inicial da formacao
T_w = 90.0  # temperatura constante na parede do poco

# Conversao de tempo
SECONDS_PER_DAY = 24.0 * 60.0 * 60.0

# Malha radial.
# Um valor maior melhora a resolucao espacial, mas reduz o passo de tempo
# permitido pelo criterio de estabilidade explicito.
n_r = 401
r = np.linspace(r_w, r_e, n_r)
dr = r[1] - r[0]

# Fator de seguranca aplicado ao passo de tempo critico.
# Para esquema explicito, nunca use 1.0 em aplicacoes praticas sem margem.
safety_factor = 0.90


# =============================================================================
# 2. CRITERIO DE ESTABILIDADE DE VON NEUMANN PARA A FORMA RADIAL
# =============================================================================

def critical_fourier_number_radial(dr_value: float, radial_positions: np.ndarray) -> tuple[float, float]:
    """Calcula o numero de Fourier critico para o FTCS radial.

    A discretizacao usada no no interno i e:

        T_i^(n+1) = T_i^n
                    + Fo * [T_(i+1)^n - 2*T_i^n + T_(i-1)^n]
                    + Fo * [dr/(2*r_i)] * [T_(i+1)^n - T_(i-1)^n]

    em que:

        Fo = alpha * dt / dr^2

    Para a analise de Von Neumann, congelamos localmente o coeficiente 1/r_i
    e substituimos uma perturbacao harmonica exp(j*i*theta). O fator de
    amplificacao local fica:

        G = 1 - 4*Fo*sin^2(theta/2)
            + j*Fo*(dr/r_i)*sin(theta)

    Exigindo |G| <= 1 para todos os modos theta:

        Fo <= 2 / max(4, (dr/r_i)^2)

    Como o termo dr/r_i e maximo perto do poco, usamos o maior valor em todos
    os nos internos. Para malhas usuais com dr/r_i <= 2, a condicao volta ao
    criterio classico Fo <= 1/2.
    """

    lambda_max = np.max(dr_value / radial_positions)
    denominator = max(4.0, lambda_max**2)
    fo_critical = 2.0 / denominator
    return fo_critical, lambda_max


r_internal = r[1:-1]
Fo_crit, lambda_max = critical_fourier_number_radial(dr, r_internal)
dt_crit = Fo_crit * dr**2 / alpha
dt = safety_factor * dt_crit
Fo = alpha * dt / dr**2

if Fo > Fo_crit:
    raise RuntimeError("Passo de tempo instavel: reduza dt ou aumente a malha radial.")


# =============================================================================
# 3. OPERADOR FTCS E SOLVER TRANSIENTE
# =============================================================================

def initial_temperature_profile() -> np.ndarray:
    """Retorna a condicao inicial com as condicoes de contorno aplicadas."""

    T = np.full(n_r, T_f, dtype=float)

    # Condicao de contorno interna:
    # a parede do poco e mantida quente pelo fluido produzido.
    T[0] = T_w

    # Condicao de contorno externa:
    # no raio externo, a formacao permanece nao perturbada.
    T[-1] = T_f
    return T


def ftcs_step(T_old: np.ndarray, dt_step: float) -> np.ndarray:
    """Avanca a solucao em um passo de tempo usando FTCS explicito.

    Discretizacao em cada no interno:

        d2T/dr2  ~= (T[i+1] - 2*T[i] + T[i-1]) / dr^2
        dT/dr    ~= (T[i+1] - T[i-1]) / (2*dr)

    Substituindo na EDP radial:

        T_new[i] = T_old[i]
                   + alpha*dt * (
                       (T[i+1] - 2*T[i] + T[i-1]) / dr^2
                       + (1/r_i)*(T[i+1] - T[i-1])/(2*dr)
                     )

    A forma implementada abaixo usa Fo = alpha*dt/dr^2:

        T_new[i] = T_old[i]
                   + Fo * [
                       T[i+1] - 2*T[i] + T[i-1]
                       + dr/(2*r_i)*(T[i+1] - T[i-1])
                     ]
    """

    Fo_local = alpha * dt_step / dr**2
    T_new = T_old.copy()

    second_derivative_term = T_old[2:] - 2.0 * T_old[1:-1] + T_old[:-2]
    radial_term = (dr / (2.0 * r[1:-1])) * (T_old[2:] - T_old[:-2])

    T_new[1:-1] = T_old[1:-1] + Fo_local * (second_derivative_term + radial_term)

    # Reaplicacao rigorosa das condicoes de contorno de Dirichlet.
    T_new[0] = T_w
    T_new[-1] = T_f
    return T_new


def solve_until_times(output_times_seconds: np.ndarray) -> np.ndarray:
    """Resolve o problema e retorna perfis T(r,t) nos tempos solicitados.

    O solver marcha uma unica vez no tempo. Quando o proximo tempo de saida
    esta mais proximo do que o dt nominal, usa-se um subpasso menor. Isso
    garante que os perfis sejam salvos exatamente nos tempos pedidos sem
    violar a estabilidade, pois dt_step <= dt.
    """

    output_times_seconds = np.asarray(output_times_seconds, dtype=float)
    if np.any(output_times_seconds < 0.0):
        raise ValueError("Os tempos de saida devem ser nao negativos.")

    order = np.argsort(output_times_seconds)
    sorted_times = output_times_seconds[order]

    profiles_sorted = np.empty((len(sorted_times), n_r), dtype=float)
    T = initial_temperature_profile()
    current_time = 0.0

    for sorted_index, target_time in enumerate(sorted_times):
        while current_time < target_time - 1.0e-9:
            dt_step = min(dt, target_time - current_time)
            T = ftcs_step(T, dt_step)
            current_time += dt_step

        profiles_sorted[sorted_index] = T.copy()

    profiles = np.empty_like(profiles_sorted)
    profiles[order] = profiles_sorted
    return profiles


def thermal_front_radius(T_profile: np.ndarray, threshold_celsius: float = 0.50) -> float:
    """Estima a posicao da frente termica.

    A frente e definida como o ponto radial mais distante em que a temperatura
    ainda difere da temperatura original da formacao por pelo menos
    threshold_celsius. Esse criterio e visual, nao uma nova condicao fisica.
    """

    disturbed_nodes = np.where(np.abs(T_profile - T_f) >= threshold_celsius)[0]
    if disturbed_nodes.size == 0:
        return r_w
    return r[disturbed_nodes[-1]]


# =============================================================================
# 4. VISUALIZACAO 1 - GRAFICO ESTATICO EM 5 TEMPOS
# =============================================================================

def plot_static_profiles() -> None:
    """Gera e salva o grafico estatico pedido no enunciado."""

    times_days = np.array([0.0, 0.1, 0.5, 1.0, 5.0])
    times_seconds = times_days * SECONDS_PER_DAY
    profiles = solve_until_times(times_seconds)

    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    cmap = plt.get_cmap("inferno")
    norm = plt.Normalize(vmin=times_days.min(), vmax=times_days.max())

    for time_day, profile in zip(times_days, profiles):
        label = "t = 0 dia (condicao inicial)" if time_day == 0.0 else f"t = {time_day:g} dia(s)"
        ax.plot(
            r,
            profile,
            color=cmap(norm(time_day)),
            linewidth=2.4,
            label=label,
        )

    ax.axvline(r_w, color="tab:red", linestyle="--", linewidth=1.4, label="Parede do poco")
    ax.axvline(r_e, color="tab:blue", linestyle="--", linewidth=1.2, label="Fronteira externa")

    ax.set_title("Difusao transiente de calor radial ao redor do poco")
    ax.set_xlabel("Raio radial, r [m]")
    ax.set_ylabel(r"Temperatura, $T$ [$^\circ$C]")
    ax.set_xlim(r_w, r_e)
    ax.set_ylim(T_f - 3.0, T_w + 5.0)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(loc="best", frameon=True)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Tempo [dias]")

    output_path = Path(__file__).with_name("perfil_estatico.png")
    fig.savefig(output_path, dpi=220)
    print(f"Grafico estatico salvo em: {output_path}")

    plt.show()


# =============================================================================
# 5. VISUALIZACAO 2 - ANIMACAO DA FRENTE TERMICA
# =============================================================================

def animate_temperature_profile() -> FuncAnimation:
    """Cria uma animacao com FuncAnimation mostrando T(r,t)."""

    t_final_days = 5.0
    n_frames = 180
    animation_times_days = np.linspace(0.0, t_final_days, n_frames)
    animation_times_seconds = animation_times_days * SECONDS_PER_DAY
    profiles = solve_until_times(animation_times_seconds)

    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)

    line, = ax.plot([], [], color="tab:red", linewidth=2.7, label="Perfil T(r,t)")
    front_line = ax.axvline(
        r_w,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Frente termica aproximada",
    )

    ax.axhline(T_f, color="tab:blue", linestyle=":", linewidth=1.5, label=r"$T_f$")
    ax.axhline(T_w, color="tab:red", linestyle=":", linewidth=1.2, label=r"$T_w$")
    ax.axvline(r_w, color="0.25", linestyle="-.", linewidth=1.0)

    time_text = ax.text(
        0.03,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )

    ax.set_title("Animacao da propagacao da frente termica na formacao")
    ax.set_xlabel("Raio radial, r [m]")
    ax.set_ylabel(r"Temperatura, $T$ [$^\circ$C]")
    ax.set_xlim(r_w, r_e)
    ax.set_ylim(T_f - 3.0, T_w + 5.0)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(loc="lower right", frameon=True)

    def init() -> tuple:
        line.set_data(r, profiles[0])
        front_line.set_xdata([r_w, r_w])
        time_text.set_text("Tempo: 0.00 dias")
        return line, front_line, time_text

    def update(frame_index: int) -> tuple:
        profile = profiles[frame_index]
        time_day = animation_times_days[frame_index]
        front_radius = thermal_front_radius(profile)

        line.set_data(r, profile)
        front_line.set_xdata([front_radius, front_radius])
        time_text.set_text(f"Tempo: {time_day:5.2f} dias")

        return line, front_line, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=45,
        blit=True,
        repeat=True,
    )

    return animation


# =============================================================================
# 6. EXECUCAO PRINCIPAL
# =============================================================================

def main() -> None:
    print("\n=== Simulacao termica radial 1D - FTCS explicito ===")
    print(f"r_w = {r_w:.3f} m")
    print(f"r_e = {r_e:.3f} m")
    print(f"alpha = {alpha:.3e} m2/s")
    print(f"T_f = {T_f:.1f} C")
    print(f"T_w = {T_w:.1f} C")
    print("\n--- Malha e estabilidade ---")
    print(f"n_r = {n_r}")
    print(f"dr = {dr:.6f} m")
    print(f"lambda_max = max(dr/r_i) = {lambda_max:.6f}")
    print(f"Fo_critico = {Fo_crit:.6f}")
    print(f"Fo_usado = {Fo:.6f}")
    print(f"dt_critico = {dt_crit:.3f} s")
    print(f"dt_usado = {dt:.3f} s ({dt / 60.0:.3f} min)")

    plot_static_profiles()

    # Manter a referencia em uma variavel evita que o coletor de lixo remova
    # a animacao antes de a janela interativa terminar de renderizar.
    animation = animate_temperature_profile()
    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
