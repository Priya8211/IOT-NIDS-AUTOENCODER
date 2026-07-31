import numpy as np

class Damped1D:
    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w = 0.0
        self.mean = 0.0
        self.var = 0.0
        self.t_last = 0.0

    def update(self, t, x):
        if self.w == 0.0:
            self.w = 1.0
            self.mean = x
            self.var = 0.0
            self.t_last = t
            return
        
        dt = max(0.0, t - self.t_last)
        alpha = np.exp(-self.lambda_val * dt)
        
        w_old = self.w
        mean_old = self.mean
        var_old = self.var
        
        self.w = w_old * alpha + 1.0
        self.mean = mean_old + (x - mean_old) / self.w
        
        ss_old = var_old * w_old
        ss_new = ss_old * alpha + (x - mean_old) * (x - self.mean)
        self.var = max(0.0, ss_new / self.w)
        self.t_last = t

class Damped2D:
    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
        self.w = 0.0
        
        # Stats for X (direction A->B)
        self.w_x = 0.0
        self.mean_x = 0.0
        self.var_x = 0.0
        
        # Stats for Y (direction B->A)
        self.w_y = 0.0
        self.mean_y = 0.0
        self.var_y = 0.0
        
        # Covariance
        self.cov = 0.0
        self.t_last = 0.0

    def update(self, t, val, is_x):
        if self.w == 0.0:
            self.w = 1.0
            self.t_last = t
            if is_x:
                self.w_x = 1.0
                self.mean_x = val
                self.var_x = 0.0
            else:
                self.w_y = 1.0
                self.mean_y = val
                self.var_y = 0.0
            self.cov = 0.0
            return
            
        dt = max(0.0, t - self.t_last)
        alpha = np.exp(-self.lambda_val * dt)
        
        w_old = self.w
        mean_x_old = self.mean_x
        mean_y_old = self.mean_y
        
        # Decay joint weight
        self.w = w_old * alpha + 1.0
        
        if is_x:
            # Update X
            if self.w_x == 0.0:
                self.w_x = 1.0
                self.mean_x = val
                self.var_x = 0.0
            else:
                self.w_x = self.w_x * alpha + 1.0
                d_x = val - mean_x_old
                self.mean_x = mean_x_old + d_x / self.w_x
                ss_x = self.var_x * (self.w_x - 1.0) * alpha + d_x * (val - self.mean_x)
                self.var_x = max(0.0, ss_x / self.w_x)
            
            # Decay Y weight
            self.w_y = self.w_y * alpha
            # Update covariance: Y didn't change (val_y = mean_y_old), so (val_y - mean_y_new) = 0
            # covariance update: cov = (cov_old * w_old * alpha + (val - mean_x_old) * (mean_y - mean_y_new)) / w_new
            self.cov = (self.cov * w_old * alpha) / self.w
        else:
            # Update Y
            if self.w_y == 0.0:
                self.w_y = 1.0
                self.mean_y = val
                self.var_y = 0.0
            else:
                self.w_y = self.w_y * alpha + 1.0
                d_y = val - mean_y_old
                self.mean_y = mean_y_old + d_y / self.w_y
                ss_y = self.var_y * (self.w_y - 1.0) * alpha + d_y * (val - self.mean_y)
                self.var_y = max(0.0, ss_y / self.w_y)
                
            # Decay X weight
            self.w_x = self.w_x * alpha
            self.cov = (self.cov * w_old * alpha) / self.w

        self.t_last = t

    def get_std_x(self):
        return np.sqrt(self.var_x)

    def get_std_y(self):
        return np.sqrt(self.var_y)

    def get_magnitude(self):
        return np.sqrt(self.mean_x**2 + self.mean_y**2)

    def get_radius(self):
        return np.sqrt(self.var_x + self.var_y)

    def get_pcc(self):
        std_x = self.get_std_x()
        std_y = self.get_std_y()
        if std_x * std_y == 0:
            return 0.0
        return self.cov / (std_x * std_y)

class KitsuneExtractor:
    def __init__(self):
        self.lambdas = [5.0, 3.0, 1.0, 0.1, 0.01]
        
        # Initialize dictionaries of trackers for each lambda j
        self.mi_trackers = [{} for _ in range(5)]
        self.h_trackers = [{} for _ in range(5)]
        self.hh_trackers = [{} for _ in range(5)]
        
        # We track connection last-times to compute jitter
        self.connection_last_t = {}
        self.hh_jit_trackers = [{} for _ in range(5)]
        self.hphp_trackers = [{} for _ in range(5)]

    def extract_features(self, t, src_mac, src_ip, src_port, dst_ip, dst_port, size):
        """
        Receives packet parameters and returns a 115-dimensional feature array.
        """
        # Form keys
        mi_key = f"{src_mac}_{src_ip}"
        h_key = dst_ip
        
        # HH key is undirected host pair
        hh_key = (min(src_ip, dst_ip), max(src_ip, dst_ip))
        is_hh_x = (src_ip <= dst_ip)
        
        # HpHp key is undirected src/dst socket pair
        sock_a = (src_ip, src_port)
        sock_b = (dst_ip, dst_port)
        hphp_key = (min(sock_a, sock_b), max(sock_a, sock_b))
        is_hphp_x = (sock_a <= sock_b)
        
        # Calculate jitter (time since last packet on this HH channel)
        last_t = self.connection_last_t.get(hh_key, None)
        jitter = 0.0
        if last_t is not None:
            jitter = max(0.0, t - last_t)
        self.connection_last_t[hh_key] = t
        
        # Arrays to hold stats for each lambda
        mi_features = []
        h_features = []
        hh_features = []
        hh_jit_features = []
        hphp_features = []
        
        for j, val_lambda in enumerate(self.lambdas):
            # 1. MI_dir
            if mi_key not in self.mi_trackers[j]:
                self.mi_trackers[j][mi_key] = Damped1D(val_lambda)
            t1 = self.mi_trackers[j][mi_key]
            t1.update(t, size)
            mi_features.extend([t1.w, t1.mean, t1.var])
            
            # 2. H
            if h_key not in self.h_trackers[j]:
                self.h_trackers[j][h_key] = Damped1D(val_lambda)
            t2 = self.h_trackers[j][h_key]
            t2.update(t, size)
            h_features.extend([t2.w, t2.mean, t2.var])
            
            # 3. HH
            if hh_key not in self.hh_trackers[j]:
                self.hh_trackers[j][hh_key] = Damped2D(val_lambda)
            t3 = self.hh_trackers[j][hh_key]
            t3.update(t, size, is_hh_x)
            hh_features.extend([
                t3.w, t3.mean_x, t3.get_std_x(), 
                t3.get_magnitude(), t3.get_radius(), 
                t3.cov, t3.get_pcc()
            ])
            
            # 4. HH_jit
            if hh_key not in self.hh_jit_trackers[j]:
                self.hh_jit_trackers[j][hh_key] = Damped1D(val_lambda)
            t4 = self.hh_jit_trackers[j][hh_key]
            # Update jitter tracker. Only update if jitter is valid (i.e. not the first packet)
            # Actually, kitsune updates jitter even if delta is 0
            t4.update(t, jitter)
            hh_jit_features.extend([t4.w, t4.mean, t4.var])
            
            # 5. HpHp
            if hphp_key not in self.hphp_trackers[j]:
                self.hphp_trackers[j][hphp_key] = Damped2D(val_lambda)
            t5 = self.hphp_trackers[j][hphp_key]
            t5.update(t, size, is_hphp_x)
            hphp_features.extend([
                t5.w, t5.mean_x, t5.get_std_x(), 
                t5.get_magnitude(), t5.get_radius(), 
                t5.cov, t5.get_pcc()
            ])
            
        # Combine all features in order into one array
        full_vector = mi_features + h_features + hh_features + hh_jit_features + hphp_features
        return np.array(full_vector)
