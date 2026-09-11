import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg as la

n_floors = 76
m_f, k_f_orig = 1.0e6, 1.0e9
k_f_deg = k_f_orig * 0.90  # 10% Stiffness Loss
mt = (n_floors * m_f) * 0.01
FORCE_CAP = 2.0e6

f_opt = 1 / (1 + 0.01)
omega_n_orig = np.sqrt(k_f_orig / m_f) * (np.pi / (2 * n_floors))
kt = mt * (omega_n_orig * f_opt) ** 2
ct = 2 * mt * (omega_n_orig * f_opt) * np.sqrt(3 * 0.01 / (8 * (1 + 0.01)))

def get_matrices(k_val):
  dof = n_floors + 1
  M = np.eye(dof);
  M[:n_floors] *= m_f;
  M[-1] = mt
  K = np.zeros((dof, dof))
  for i in range(n_floors):
      K[i, i] += k_val
      if i > 0:
          K[i, i] += k_val
          K[i, i - 1] = K[i - 1, i] = -k_val
  K[n_floors - 1, n_floors - 1] += kt;
  K[n_floors, n_floors] += kt
  K[n_floors - 1, n_floors] = K[n_floors, n_floors - 1] = -kt
  C = 0.02 * M + 0.005 * K
  return M, C, K

def get_lqr_gain(M, C, K):
  dof = n_floors + 1
  M_inv = la.inv(M)
  A = np.vstack([np.hstack([np.zeros((dof, dof)), np.eye(dof)]),
                 np.hstack([-M_inv @ K, -M_inv @ C])])
  B = np.vstack([np.zeros((dof, 1)), M_inv @ (np.array([([0] * (n_floors - 1)) + [-1, 1]]).T)])
  Q = np.eye(2 * dof) * 1e4
  Q[n_floors - 1, n_floors - 1] = 1e12
  R = np.array([[1e-6]])
  return la.inv(R) @ B.T @ la.solve_continuous_are(A, B, Q, R)

def run_sim(M, C, K, K_lqr, active):
  dof = n_floors + 1
  M_inv = la.inv(M)
  def dynamics(t, state):
      u = np.clip(-(K_lqr @ state)[0], -FORCE_CAP, FORCE_CAP) if active else 0
      x, v = state[:dof], state[dof:]
      f = np.zeros(dof);
      f[n_floors - 1], f[n_floors] = -u, u
      return np.concatenate([v, M_inv @ (-K @ x - C @ v + f)])
  #initial 20cm snap
  y0 = np.zeros(2 * dof);
  y0[n_floors - 1] = 0.2
  return solve_ivp(dynamics, (0, 60), y0, t_eval=np.linspace(0, 60, 10000), method='LSODA')

#simulations yayy
M_o, C_o, K_o = get_matrices(k_f_orig);
LQR_o = get_lqr_gain(M_o, C_o, K_o)
sol_o_p = run_sim(M_o, C_o, K_o, LQR_o, False);
sol_o_a = run_sim(M_o, C_o, K_o, LQR_o, True)

M_d, C_d, K_d = get_matrices(k_f_deg);
LQR_d = get_lqr_gain(M_d, C_d, K_d)
sol_d_p = run_sim(M_d, C_d, K_d, LQR_d, False);
sol_d_a = run_sim(M_d, C_d, K_d, LQR_d, True)

#analysis
def get_stats(sol, K_lqr, active):
  y = sol.y[n_floors - 1, :]
  #threshold = 0.01mm (1e-5) for microvibrations
  thresh = 1e-4
  idx = np.where(np.abs(y) > thresh)[0]
  ts = sol.t[idx[-1]] if len(idx) > 0 else 0.01
  v_rel = sol.y[n_floors + n_floors - 1, :] - sol.y[n_floors + n_floors, :]
  force = np.clip(-(K_lqr @ sol.y), -FORCE_CAP, FORCE_CAP) if active else np.zeros_like(v_rel)
  peak_pwr = np.max(np.abs(force * v_rel)) / 1e6
  return ts, peak_pwr
s = {k: get_stats(sol, g, a) for k, sol, g, a in [
  ("OP", sol_o_p, LQR_o, False), ("OA", sol_o_a, LQR_o, True),
  ("DP", sol_d_p, LQR_d, False), ("DA", sol_d_a, LQR_d, True)]}

#graphs
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
imp_o = ((s["OP"][0] - s["OA"][0]) / s["OP"][0] * 100) if s["OP"][0] > 0 else 0
imp_d = ((s["DP"][0] - s["DA"][0]) / s["DP"][0] * 100) if s["DP"][0] > 0 else 0
for i, (p_sol, a_sol, p_t, a_t, imp) in enumerate([
  (sol_o_p, sol_o_a, s["OP"][0], s["OA"][0], imp_o),
  (sol_d_p, sol_d_a, s["DP"][0], s["DA"][0], imp_d)]):
  # linear plots
  axes[i, 0].plot(p_sol.t, p_sol.y[n_floors - 1, :], 'g', alpha=0.5, label=f"Passive ({p_t:.2f}s)")
  axes[i, 0].plot(a_sol.t, a_sol.y[n_floors - 1, :], 'orange', label=f"Active ({a_t:.2f}s)")
  axes[i, 0].set_title(f"{['Original', '10% Degraded'][i]} Building: {imp:.1f}% Improvement")
  axes[i, 0].legend();
  axes[i, 0].grid(True, alpha=0.3)

  # log plots
  axes[i, 1].semilogy(p_sol.t, np.abs(p_sol.y[n_floors - 1, :]), 'g', alpha=0.4)
  axes[i, 1].semilogy(a_sol.t, np.abs(a_sol.y[n_floors - 1, :]), 'orange')
  axes[i, 1].set_title(f"{['Original', '10% Degraded'][i]}: Microvibration Energy (Log)")
  axes[i, 1].grid(True, which='both', alpha=0.2)
plt.tight_layout();
plt.show()

print(f"Original Building: {imp_o:.1f}% Improvement | Peak Power: {s['OA'][1]:.2f} MW")
print(f"Degraded Building: {imp_d:.1f}% Improvement | Peak Power: {s['DA'][1]:.2f} MW")