import casadi as ca
import numpy as np
import torch


class BilinearMPC:
    def __init__(self, model, cfg):
        """
        model: 학습된 DeepBilinearKoopman 모델 (PyTorch)
        cfg: 설정 딕셔너리
        """
        self.model = model
        self.cfg = cfg

        # 차원 설정
        self.z_dim = cfg['dims']['full_state']  # 35
        self.u_dim = cfg['dims']['control']  # 6
        self.N = cfg['mpc']['prediction_horizon']

        # 가중치 설정
        self.Q_diag = cfg['mpc']['weights']['Q']
        self.R_diag = cfg['mpc']['weights']['R']
        self.P_diag = cfg['mpc']['weights']['P']  # 터미널 코스트 (P)

        # 제어 한계
        self.u_lim = cfg['mpc']['control_limit']

        # CasADi 최적화 설정
        self.opti = ca.Opti()

        # 변수 정의
        self.Z = self.opti.variable(self.z_dim, self.N + 1)  # 상태 궤적 (N+1)
        self.U = self.opti.variable(self.u_dim, self.N)  # 제어 궤적 (N)

        # 파라미터 정의 (매 스텝 런타임에 업데이트됨)
        self.z_init = self.opti.parameter(self.z_dim)
        self.z_ref = self.opti.parameter(self.z_dim)
        self.A_num = self.opti.parameter(self.z_dim, self.z_dim)
        self.B_eff_num = self.opti.parameter(self.z_dim, self.u_dim)

        # 목적 함수 (Cost Function)
        obj = 0
        Q = self.Q_diag * ca.MX.eye(self.z_dim)
        R = self.R_diag * ca.MX.eye(self.u_dim)
        P_term = self.P_diag * ca.MX.eye(self.z_dim)

        for k in range(self.N):
            st_err = self.Z[:, k] - self.z_ref
            con = self.U[:, k]
            obj += ca.mtimes([st_err.T, Q, st_err]) + ca.mtimes([con.T, R, con])

        # 터미널 코스트
        term_err = self.Z[:, self.N] - self.z_ref
        obj += ca.mtimes([term_err.T, P_term, term_err])

        self.opti.minimize(obj)

        # 제약 조건 (Constraints)
        self.opti.subject_to(self.Z[:, 0] == self.z_init)  # 초기값

        for k in range(self.N):
            # 선형화된 동역학: z_{k+1} = A z_k + B_eff u_k
            z_next = ca.mtimes(self.A_num, self.Z[:, k]) + ca.mtimes(self.B_eff_num, self.U[:, k])
            self.opti.subject_to(self.Z[:, k + 1] == z_next)

        # 제어 입력 범위 제한
        self.opti.subject_to(self.opti.bounded(-self.u_lim, self.U, self.u_lim))

        # 상태 제약 조건 Gx <= d 로드 (논문 수식 4, 29 반영) [cite: 315, 504]
        # cfg['mpc']['constraints']가 있다고 가정
        if 'constraints' in cfg['mpc']:
            G_mat = np.array(cfg['mpc']['constraints']['G'])  # [n_cons, 15]
            d_vec = np.array(cfg['mpc']['constraints']['d'])  # [n_cons]

            # 논문 수식 (30)의 G_hat = [G, 0] 반영 [cite: 509]
            G_lifted = np.zeros((G_mat.shape[0], self.z_dim))
            G_lifted[:, :G_mat.shape[1]] = G_mat

            for k in range(self.N):
                # z_{k+1}에 대한 상태 제약 조건 추가: G_hat * z_{k+1} <= d_hat
                self.opti.subject_to(ca.mtimes(G_lifted, self.Z[:, k + 1]) <= d_vec)
        # # 솔버 설정 (IPOPT)
        # opts = {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'}
        # self.opti.solver('ipopt', opts)
        # 솔버 설정 (논문의 권장사항에 따라 QP 전용 솔버 OSQP 사용)
        # 참고: 환경에 osqp가 설치되어 있어야 합니다 (pip install osqp)
        # 솔버 설정 (OSQP 로딩 에러로 인해 IPOPT 사용)
        opts = {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)
        print("안정성을 위해 IPOPT 솔버를 사용합니다.")
    def update_model_matrices(self, z_current_np):
        """현재 상태 z를 기준으로 B_eff를 선형화 업데이트합니다."""
        with torch.no_grad():
            A_torch = self.model.A
            B_torch = self.model.B
            H_torch = self.model.H  # [control_dim, full_dim, full_dim]

            self.A_val = A_torch.cpu().numpy()
            B_val = B_torch.cpu().numpy()
            B_eff = B_val.copy()

            z_torch = torch.from_numpy(z_current_np).float().to(H_torch.device)

            for i in range(self.u_dim):
                # 논문 수식 반영: B_eff[:, i] = B[:, i] + Hi * z
                h_z = (H_torch[i] @ z_torch).cpu().numpy()
                B_eff[:, i] += h_z

            self.B_eff_val = B_eff

    def solve(self, z0, z_ref):
        """MPC 최적화 문제를 풀어 첫 번째 제어 입력을 반환합니다."""
        self.opti.set_value(self.z_init, z0)
        self.opti.set_value(self.z_ref, z_ref)
        self.opti.set_value(self.A_num, self.A_val)
        self.opti.set_value(self.B_eff_num, self.B_eff_val)

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            if self.N == 1:
                return u_opt.reshape(-1)
            return u_opt[:, 0]  # 첫 번째 제어 액션 반환
        except Exception as e:
            return np.zeros(self.u_dim)