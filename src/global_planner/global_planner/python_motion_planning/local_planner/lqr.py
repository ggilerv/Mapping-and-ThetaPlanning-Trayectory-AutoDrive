# lqr.py
import numpy as np
from .local_planner import LocalPlanner
from python_motion_planning.utils import Env
from python_motion_planning.utils import SearchFactory

class LQR(LocalPlanner):
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", **params) -> None:
        super().__init__(start, goal, env, heuristic_type, **params)

        # LQR params (ajústalos si quieres)
        self.Q = params.get("Q", np.diag([1.0, 1.0, 1.0]))
        self.R = params.get("R", np.diag([1.0, 1.0]))
        self.lqr_iteration = params.get("lqr_iteration", 100)
        self.eps_iter = params.get("eps_iter", 1e-1)

        # ---- Ruta global ----
        # Si el usuario ya pasó una ruta (lista de (ix,iy)), úsala.
        # Si no, opcionalmente puedes computarla aquí (solo si tu framework lo soporta).
        if "path" in params and params["path"]:
            self.path = params["path"]   # se espera start->goal

    def __str__(self) -> str:
        return "Linear Quadratic Regulator (LQR)"

    # def plan(self):
        dt = self.params["TIME_STEP"]

        # tolerancias razonables (ajusta a tu grid)
        GOAL_POS_TOL = self.params.get("GOAL_POS_TOL", 1.0)    # en celdas
        GOAL_YAW_TOL = self.params.get("GOAL_YAW_TOL", np.deg2rad(10))

        # velocidad mínima para “despegar” cuando ya apunto bien
        V_FEED = self.params.get("V_FEED", 0.3)

        # detector de atasco
        stuck_window = 200          # pasos para evaluar
        min_prog     = 0.2          # avance mínimo en celdas
        last_check_p = np.array([self.robot.px, self.robot.py], dtype=float)
        steps_since  = 0

        for _ in range(self.params["MAX_ITERATION"]):
            # criterio de llegada más laxo
            dx = self.robot.px - self.goal[0]
            dy = self.robot.py - self.goal[1]
            dpos = np.hypot(dx, dy)
            if dpos < GOAL_POS_TOL and abs(self.regularizeAngle(self.robot.theta - self.goal[2])) < GOAL_YAW_TOL:
                return True, self.robot.history_pose

            # lookahead robusto
            out = self.getLookaheadPoint()
            if out is None:
                # si tu getLookaheadPoint no puede, empuja hacia goal directo
                lookahead_pt = (self.goal[0], self.goal[1])
                theta_trj = self.angle(self.robot.position, lookahead_pt)
                kappa = 0.0
            else:
                lookahead_pt, theta_trj, kappa = out

            # control
            e_theta_goal = self.regularizeAngle(self.robot.theta - self.goal[2])

            # ¿me muevo hacia waypoint o roto?
            # Regla simple: si el heading hacia el lookahead es razonable, avanza.
            e_theta_path = self.regularizeAngle(self.angle(self.robot.position, lookahead_pt) - self.robot.theta)
            heading_ok = abs(e_theta_path) < np.deg2rad(20)

            if not heading_ok:
                # rota en el sitio
                u = np.array([[0.0], [self.angularRegularization(e_theta_path / dt)]])
            else:
                # LQR con feedforward; si v≈0, fuerza un V_FEED
                v_ref = max(self.robot.v, V_FEED)
                u_r = (v_ref, v_ref * kappa)

                s   = (self.robot.px, self.robot.py, self.robot.theta)
                s_d = (lookahead_pt[0], lookahead_pt[1], theta_trj)

                # --- LQR robusto (pequeña reg. a S) ---
                dt = self.params["TIME_STEP"]
                A = np.identity(3)
                A[0, 2] = -u_r[0] * np.sin(s_d[2]) * dt
                A[1, 2] =  u_r[0] * np.cos(s_d[2]) * dt
                B = np.zeros((3, 2))
                B[0, 0] = np.cos(s_d[2]) * dt
                B[1, 0] = np.sin(s_d[2]) * dt
                B[2, 1] = dt

                P = self.Q.copy()
                for _it in range(self.lqr_iteration):
                    BT_P = B.T @ P
                    S = self.R + BT_P @ B
                    # regularización (muy pequeña) para evitar singularidad
                    S = S + 1e-9 * np.eye(S.shape[0])
                    Ktmp = np.linalg.inv(S) @ (BT_P @ A)
                    P_new = self.Q + A.T @ P @ A - A.T @ P @ B @ Ktmp
                    if np.max(np.abs(P_new - P)) < self.eps_iter:
                        P = P_new
                        break
                    P = P_new

                S = self.R + B.T @ P @ B
                S = S + 1e-9 * np.eye(S.shape[0])
                K = np.linalg.inv(S) @ (B.T @ P @ A)

                e = np.array([[s[0] - s_d[0]],
                            [s[1] - s_d[1]],
                            [self.regularizeAngle(s[2] - s_d[2])]])

                u = np.array([[u_r[0]], [u_r[1]]]) - K @ e
                u = np.array([
                    [self.linearRegularization(float(u[0]))],
                    [self.angularRegularization(float(u[1]))]
                ])

            # cinemática
            self.robot.kinematic(u, dt)

            # --- detector de atasco: si no avanzas, cambia estrategia ---
            steps_since += 1
            if steps_since >= stuck_window:
                now_p = np.array([self.robot.px, self.robot.py], dtype=float)
                if np.linalg.norm(now_p - last_check_p) < min_prog:
                    # empujón: ignora LQR una vez y ve directo al lookahead con avance fijo
                    e_theta_path = self.regularizeAngle(self.angle((self.robot.px, self.robot.py), lookahead_pt) - self.robot.theta)
                    self.robot.kinematic(np.array([[V_FEED], [self.angularRegularization(e_theta_path / dt)]]), dt)
                last_check_p = now_p
                steps_since = 0

        return False, None

    def plan(self):
        dt = self.params["TIME_STEP"]
        POS_TOL  = self.params.get("GOAL_POS_TOL", 1.0)
        YAW_TOL  = self.params.get("GOAL_YAW_TOL", np.deg2rad(12))
        V_FEED   = self.params.get("V_FEED", 0.25)

        for _ in range(self.params["MAX_ITERATION"]):
            # goal check
            dx, dy = self.robot.px - self.goal[0], self.robot.py - self.goal[1]
            if np.hypot(dx, dy) < POS_TOL and abs(self.regularizeAngle(self.robot.theta - self.goal[2])) < YAW_TOL:
                return True, self.robot.history_pose

            out = self.getLookaheadPoint()
            if out is None:
                lookahead_pt = (self.goal[0], self.goal[1])
                theta_trj, kappa = self.angle(self.robot.position, lookahead_pt), 0.0
            else:
                lookahead_pt, theta_trj, kappa = out

            e_theta = self.regularizeAngle(self.angle(self.robot.position, lookahead_pt) - self.robot.theta)
            if abs(e_theta) > np.deg2rad(20):
                u = np.array([[0.0], [self.angularRegularization(e_theta / dt)]])
            else:
                v_ref = max(self.robot.v, V_FEED)
                u_r   = (v_ref, v_ref * kappa)
                s, s_d = (self.robot.px, self.robot.py, self.robot.theta), (lookahead_pt[0], lookahead_pt[1], theta_trj)

                # LQR mínimo con regularización
                A = np.eye(3); A[0,2] = -u_r[0]*np.sin(s_d[2])*dt; A[1,2] = u_r[0]*np.cos(s_d[2])*dt
                B = np.zeros((3,2)); B[0,0] = np.cos(s_d[2])*dt; B[1,0] = np.sin(s_d[2])*dt; B[2,1] = dt
                P = self.Q.copy()
                for _it in range(self.lqr_iteration):
                    BT_P = B.T @ P
                    S = self.R + BT_P @ B + 1e-9*np.eye(2)
                    Ktmp = np.linalg.inv(S) @ (BT_P @ A)
                    P_new = self.Q + A.T @ P @ A - A.T @ P @ B @ Ktmp
                    if np.max(np.abs(P_new - P)) < self.eps_iter: P = P_new; break
                    P = P_new

                S = self.R + B.T @ P @ B + 1e-9*np.eye(2)
                K = np.linalg.inv(S) @ (B.T @ P @ A)
                e = np.array([[s[0]-s_d[0]],[s[1]-s_d[1]],[self.regularizeAngle(s[2]-s_d[2])]])
                u = np.array([[u_r[0]],[u_r[1]]]) - K @ e
                u = np.array([[self.linearRegularization(float(u[0]))],
                            [self.angularRegularization(float(u[1]))]])

            self.robot.kinematic(u, dt)

        return False, None


    def run(self):
        ok, history_pose = self.plan()
        if not ok or not history_pose:
            raise ValueError("Path not found and planning failed!")
        path = np.array(history_pose)[:, 0:2]
        cost = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1, keepdims=True)))
        self.plot.plotPath(self.path, path_color="r", path_style="--")
        self.plot.animation(path, str(self), cost, history_pose=history_pose)

    def lqrControl(self, s: tuple, s_d: tuple, u_r: tuple) -> np.ndarray:
        dt = self.params["TIME_STEP"]

        # linealización en torno al estado deseado y u_r
        A = np.identity(3)
        A[0, 2] = -u_r[0] * np.sin(s_d[2]) * dt
        A[1, 2] =  u_r[0] * np.cos(s_d[2]) * dt

        B = np.zeros((3, 2))
        B[0, 0] = np.cos(s_d[2]) * dt
        B[1, 0] = np.sin(s_d[2]) * dt
        B[2, 1] = dt

        # Riccati discreto por iteración
        P  = self.Q.copy()
        for _ in range(self.lqr_iteration):
            BT_P = B.T @ P
            S = self.R + BT_P @ B
            Ktmp = np.linalg.inv(S) @ (BT_P @ A)
            P_new = self.Q + A.T @ P @ A - A.T @ P @ B @ Ktmp
            if np.max(np.abs(P_new - P)) < self.eps_iter:
                P = P_new
                break
            P = P_new

        # Ganancia y control
        S = self.R + B.T @ P @ B
        K = np.linalg.inv(S) @ (B.T @ P @ A)
        e = np.array([[s[0] - s_d[0]],
                      [s[1] - s_d[1]],
                      [self.regularizeAngle(s[2] - s_d[2])]])
        u = np.array([[u_r[0]], [u_r[1]]]) - K @ e

        return np.array([
            [self.linearRegularization(float(u[0]))],
            [self.angularRegularization(float(u[1]))]
        ])
