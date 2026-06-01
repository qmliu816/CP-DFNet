import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention import SelfAttentionLayer


def split_trend_residual(x, k=12):
    B, T, N, C = x.shape
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    # (B,N,C,T) -> (B*N*C,1,T)
    x_ = x.permute(0, 2, 3, 1).reshape(B * N * C, 1, T)
    trend_ = F.avg_pool1d(x_, kernel_size=k, stride=1, padding=k // 2)
    trend = trend_.reshape(B, N, C, T).permute(0, 3, 1, 2)
    residual = x - trend
    return residual, trend

class TemporalConv(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_feats, out_feats, kernel_size,
                              padding=padding, dilation=dilation)

    def forward(self, x):
        # x: (B,N,T,C) -> (B*N,C,T)
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).transpose(1, 2)
        y = self.conv(x)
        y = y[:, :, :T]
        # (B*N,out,T) -> (B,N,T,out)
        y = y.transpose(1, 2).view(B, N, T, -1)
        return y

class TemporalMixBlock(nn.Module):
    def __init__(self, hidden_feats, dropout=0.1):
        super().__init__()
        self.conv_d1 = TemporalConv(hidden_feats, hidden_feats, kernel_size=3, dilation=1)
        self.conv_d2 = TemporalConv(hidden_feats, hidden_feats, kernel_size=3, dilation=2)
        self.gate = nn.Linear(hidden_feats * 2, hidden_feats)
        self.norm = nn.LayerNorm(hidden_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,T,N,H) -> (B,N,T,H)
        B, T, N, H = x.shape
        x_ = x.permute(0, 2, 1, 3)

        y1 = self.conv_d1(x_)
        y2 = self.conv_d2(x_)
        y = torch.cat([y1, y2], dim=-1)  # (B,N,T,2H)
        y = self.gate(y)
        y = torch.relu(y)
        y = self.dropout(y)

        y = y.permute(0, 2, 1, 3)
        out = self.norm(x + y)
        return out

class SimpleGAT(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.1, edge_bias_scale=1.0):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.edge_bias_scale = edge_bias_scale

    def forward(self, x, adj_mask=None, edge_weight=None):
        B, T, N, H = x.shape
        h = self.fc(x)  # (B,T,N,H')

        h_i = h.unsqueeze(3).expand(-1, -1, -1, N, -1)
        h_j = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.attn_fc(a_input).squeeze(-1)  # (B,T,N,N)
        e = self.leaky_relu(e)

        if edge_weight is not None:
            e = e + self.edge_bias_scale * edge_weight.view(1, 1, N, N)

        if adj_mask is not None:
            mask = adj_mask.view(1, 1, N, N)
            e = e.masked_fill(mask == 0, float('-inf'))

        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        h_prime = torch.matmul(alpha, h)  # (B,T,N,H')
        return h_prime


class LocalBranchBlock(nn.Module):
    def __init__(self, hidden_feats, geo_adj, critical_adj=None,
                 fusion_alpha=0.7, dropout=0.1, device='cpu', edge_bias_scale=1.0):
        super().__init__()
        self.device = device

        self.register_buffer('geo_adj_buf', geo_adj.float())
        if critical_adj is not None:
            self.register_buffer('critical_adj_buf', critical_adj.float())
        else:
            self.critical_adj_buf = None

        init_alpha = torch.tensor(float(fusion_alpha)).clamp(1e-3, 1-1e-3)
        self.alpha_logit = nn.Parameter(torch.logit(init_alpha))

        self.tem_mix = TemporalMixBlock(hidden_feats, dropout=dropout)
        self.gat = SimpleGAT(hidden_feats, hidden_feats, dropout=dropout, edge_bias_scale=edge_bias_scale)

    @staticmethod
    def _row_normalize(adj):
        row = adj.sum(dim=1, keepdim=True)
        row = torch.where(row <= 0, torch.ones_like(row), row)
        return adj / row

    def forward(self, x):
        """
        x: (B,T,N,H)
        """
        B, T, N, H = x.shape

        h = self.tem_mix(x)  # (B,T,N,H)

        geo_adj = self.geo_adj_buf
        if self.critical_adj_buf is not None:
            alpha = torch.sigmoid(self.alpha_logit)
            fused_adj = alpha * self.critical_adj_buf + (1 - alpha) * geo_adj
        else:
            fused_adj = geo_adj

        fused_adj = self._row_normalize(fused_adj)
        edge_weight = torch.log(torch.clamp(fused_adj, min=1e-6))

        out = self.gat(h, adj_mask=(fused_adj > 0).float(), edge_weight=edge_weight)
        out = F.elu(out)

        return x + out

class MultiHopCriticalPathFinder:
    def __init__(self, coordinates, pollution_data,
                 correlation_threshold=0.6, distance_threshold=100,
                 max_hops=3, distance_decay=0.8,
                 max_paths_to_expand=4, gaussian_l=50.0, freq_gamma=1.0):

        self.coordinates = coordinates
        self.pollution_data = pollution_data
        self.correlation_threshold = correlation_threshold
        self.distance_threshold = distance_threshold
        self.max_hops = max_hops
        self.distance_decay = distance_decay
        self.max_paths_to_expand = max_paths_to_expand
        self.num_nodes = coordinates.shape[0]
        self.gaussian_l = gaussian_l
        self.freq_gamma = freq_gamma

    def find_critical_paths(self):
        direct_corr = self._compute_correlation_matrix()
        distance_matrix = self._compute_distance_matrix()
        multi_hop_paths = self._find_multi_hop_paths(direct_corr, distance_matrix)
        multi_hop_adj = self._build_multi_hop_adjacency(multi_hop_paths, direct_corr, distance_matrix)
        return multi_hop_adj, multi_hop_paths

    def _compute_correlation_matrix(self):
        N = self.num_nodes
        corr = np.zeros((N, N), dtype=np.float32)
        x = self.pollution_data  # (T, N)
        for i in range(N):
            for j in range(N):
                if i == j:
                    corr[i, j] = 1.0
                else:
                    a = x[:, i]
                    b = x[:, j]
                    mask = ~(np.isnan(a) | np.isnan(b))
                    if mask.sum() > 24:
                        v = np.corrcoef(a[mask], b[mask])[0, 1]
                        corr[i, j] = 0.0 if np.isnan(v) else v
        return corr

    def _compute_distance_matrix(self):
        # 哈弗辛距离（km）
        from math import radians, sin, cos, asin, sqrt
        coords = self.coordinates
        N = self.num_nodes
        D = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            lat1, lon1 = coords[i]
            for j in range(N):
                lat2, lon2 = coords[j]
                if i == j:
                    D[i, j] = 0.0
                else:
                    # haversine
                    rlat1, rlon1 = radians(lat1), radians(lon1)
                    rlat2, rlon2 = radians(lat2), radians(lon2)
                    dlat = rlat2 - rlat1
                    dlon = rlon2 - rlon1
                    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
                    c = 2 * asin(sqrt(a))
                    D[i, j] = 6371.0 * c
        return D

    def _one_hop_weight(self, corr_val, dist_val):
        if np.isnan(corr_val):
            return 0.0
        if corr_val <= self.correlation_threshold:
            return 0.0
        if dist_val >= self.distance_threshold:
            return 0.0
        l = float(self.gaussian_l)
        if l <= 0:
            # 退化为仅使用相关性
            return float(corr_val)
        return float(corr_val) * float(np.exp(- (dist_val ** 2) / (2.0 * l * l)))

    def _find_multi_hop_paths(self, direct_corr, distance_matrix):
        N = self.num_nodes
        all_paths = []

        B = getattr(self, "max_paths_to_expand", 4)

        for s in range(N):
            first_layer = []
            for t in range(N):
                if s == t:
                    continue
                w = self._one_hop_weight(direct_corr[s, t], distance_matrix[s, t])
                if w <= 0.0:
                    continue

                path = dict(
                    source=s,
                    target=t,
                    path=[s, t],
                    total_correlation=float(direct_corr[s, t]),
                    weight=float(w),
                    hops=1,
                )
                first_layer.append(path)

            if not first_layer:
                continue

            first_layer.sort(key=lambda p: p["weight"], reverse=True)
            beam = first_layer[:B]

            found_for_source = []
            for p in beam:
                if p["weight"] > self.correlation_threshold:
                    found_for_source.append(p)

            current_beam = beam
            while current_beam:
                cur_hops = current_beam[0]["hops"]
                if cur_hops >= self.max_hops:
                    break

                next_candidates = []
                for cur in current_beam:
                    u = cur["target"]
                    for v in range(N):
                        if v in cur["path"]:
                            continue

                        w_edge = self._one_hop_weight(direct_corr[u, v], distance_matrix[u, v])
                        if w_edge <= 0.0:
                            continue

                        new_w = cur["weight"] * float(w_edge)
                        new_corr = cur["total_correlation"] * float(direct_corr[u, v])
                        new_hops = cur["hops"] + 1

                        new_path = dict(
                            source=s,
                            target=v,
                            path=cur["path"] + [v],
                            total_correlation=new_corr,
                            weight=new_w,
                            hops=new_hops,
                        )
                        next_candidates.append(new_path)

                if not next_candidates:
                    break

                next_candidates.sort(key=lambda p: p["weight"], reverse=True)
                current_beam = next_candidates[:B]

                for p in current_beam:
                    if p["weight"] > self.correlation_threshold:
                        found_for_source.append(p)

            all_paths.extend(found_for_source)

        return all_paths

    def _build_multi_hop_adjacency(self, paths, direct_corr, distance_matrix):
        N = self.num_nodes

        w_direct = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                w_ij = self._one_hop_weight(direct_corr[i, j], distance_matrix[i, j])
                if w_ij > 0.0:
                    w_direct[i, j] = float(w_ij)

        freq_counts = np.zeros((N, N), dtype=np.float32)
        for p in paths:
            nodes = p.get("path", None)
            if nodes is None or len(nodes) < 2:
                continue
            for k in range(len(nodes) - 1):
                i = nodes[k]
                j = nodes[k + 1]
                if i == j:
                    continue
                freq_counts[i, j] += 1.0
                freq_counts[j, i] += 1.0

        c_max = freq_counts.max()
        if c_max > 0.0:
            f_ij = freq_counts / c_max
        else:
            f_ij = np.zeros_like(freq_counts)

        gamma = float(self.freq_gamma)
        w_crit = w_direct * np.exp(gamma * f_ij)

        adj = (w_crit + w_crit.T) / 2.0
        row = adj.sum(axis=1, keepdims=True)
        row[row <= 0.0] = 1.0
        adj = adj / row

        return torch.from_numpy(adj.astype(np.float32))


class GlobalEncoderSAL(nn.Module):
    def __init__(self, num_nodes, in_steps, out_steps,
                 input_dim, hidden_feats, num_layers=2, num_heads=4,
                 feed_forward_mult=4, dropout=0.1,
                 causal_time=True, use_node_emb=True, use_time_pe=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.hidden = hidden_feats
        self.use_node_emb = use_node_emb
        self.use_time_pe = use_time_pe

        ff_dim = feed_forward_mult * hidden_feats

        self.in_proj = nn.Linear(input_dim, hidden_feats)
        if self.use_time_pe:
            self.time_pe = nn.Parameter(torch.randn(1, in_steps, 1, hidden_feats))
        if self.use_node_emb:
            self.node_emb = nn.Parameter(torch.randn(1, 1, num_nodes, hidden_feats))


        self.time_layers = nn.ModuleList()
        self.space_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.time_layers.append(
                SelfAttentionLayer(
                    model_dim=hidden_feats,
                    feed_forward_dim=ff_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mask=causal_time
                )
            )

            self.space_layers.append(
                SelfAttentionLayer(
                    model_dim=hidden_feats,
                    feed_forward_dim=ff_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mask=False
                )
            )

        self.out_proj = nn.Linear(in_steps * hidden_feats, out_steps)

    def forward(self, x):
        """
        x: (B,T,N,C)
        """
        B, T, N, C = x.shape
        h = self.in_proj(x)

        if self.use_time_pe:
            h = h + self.time_pe[:, :T]
        if self.use_node_emb:
            h = h + self.node_emb

        for layer in self.time_layers:
            h = layer(h, dim=1)
        for layer in self.space_layers:
            h = layer(h, dim=2)

        y = h.transpose(1, 2).reshape(B, N, T * self.hidden)   # (B,N,T*H)
        y = self.out_proj(y).view(B, N, self.out_steps, 1)     # (B,N,L,1)
        return y.transpose(1, 2)                               # (B,L,N,1)


class Stem(nn.Module):
    def __init__(self, input_dim, hidden_feats, input_seq_len):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_feats)
        self.pos_emb = nn.Parameter(torch.randn(1, input_seq_len, 1, hidden_feats))

    def forward(self, x):
        # x: (B,T,N,C)
        B, T, N, C = x.shape
        h = self.in_proj(x)
        if self.pos_emb.size(1) >= T:
            h = h + self.pos_emb[:, :T]
        else:
            pos = self.pos_emb
            if pos.size(1) < T:
                repeat_times = (T + pos.size(1) - 1) // pos.size(1)
                pos = pos.repeat(1, repeat_times, 1, 1)
            h = h + pos[:, :T]
        return h

class EffectivePSOFusionHead(nn.Module):
    def __init__(self, population_size=15, max_iter=8,
                 inertia=0.7, c1=1.5, c2=1.5):
        super().__init__()
        self.population_size = population_size
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2

        self.lambda_logit = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, out_short, out_long):
        lam = torch.sigmoid(self.lambda_logit)
        fused = lam * out_short + (1.0 - lam) * out_long
        return fused, float(lam.detach().cpu())

    def _hybrid_initialize(self, device):
        n = self.population_size
        n_rand = n // 2
        n_strat = n - n_rand

        random_part = torch.rand(n_rand, device=device)

        # Stratified sampling over [0, 1]
        intervals = torch.linspace(0.0, 1.0, n_strat + 1, device=device)
        low = intervals[:-1]
        high = intervals[1:]
        stratified_part = low + (high - low) * torch.rand(n_strat, device=device)

        particles = torch.cat([random_part, stratified_part], dim=0)
        return particles.clamp(0.0, 1.0)

    @torch.no_grad()
    def _fitness(self, model, val_loader, device, lam):
        model.eval()
        total_loss = 0.0
        total_count = 0

        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            out_short, out_long, _, _ = model(X_batch)
            fused = lam * out_short + (1.0 - lam) * out_long

            loss = torch.mean((fused - Y_batch) ** 2)
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def optimize_once(self, model, val_loader, device):
        particles = self._hybrid_initialize(device)
        velocities = torch.zeros_like(particles)

        pbest = particles.clone()
        pbest_scores = torch.tensor(
            [self._fitness(model, val_loader, device, p.item()) for p in particles],
            device=device
        )

        best_idx = torch.argmin(pbest_scores)
        gbest = pbest[best_idx].clone()
        gbest_score = pbest_scores[best_idx].clone()

        for _ in range(self.max_iter):
            r1 = torch.rand_like(particles)
            r2 = torch.rand_like(particles)

            velocities = (
                self.inertia * velocities
                + self.c1 * r1 * (pbest - particles)
                + self.c2 * r2 * (gbest - particles)
            )

            particles = torch.clamp(particles + velocities, 0.0, 1.0)

            scores = torch.tensor(
                [self._fitness(model, val_loader, device, p.item()) for p in particles],
                device=device
            )

            improved = scores < pbest_scores
            pbest[improved] = particles[improved]
            pbest_scores[improved] = scores[improved]

            best_idx = torch.argmin(pbest_scores)
            if pbest_scores[best_idx] < gbest_score:
                gbest = pbest[best_idx].clone()
                gbest_score = pbest_scores[best_idx].clone()

        # Convert best lambda to logit and store it.
        eps = 1e-6
        best_lambda = torch.clamp(gbest, eps, 1.0 - eps)
        self.lambda_logit.data = torch.log(best_lambda / (1.0 - best_lambda)).to(
            self.lambda_logit.device
        )

        return float(best_lambda.detach().cpu()), float(gbest_score.detach().cpu())


class STMultiHorizonNet(nn.Module):
    def __init__(self, input_dim, hidden_feats, out_len, geo_adj,
                 input_seq_len=24, num_short_blocks=2, num_long_blocks=1,
                 dropout=0.1, use_critical_paths=False, coordinates=None, pollution_data=None,
                 fusion_alpha=0.7, correlation_threshold=0.6, distance_threshold=100,
                 max_hops=3, max_paths_to_expand=8,device='cpu', use_trend_residual_split=True):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_feats = hidden_feats
        self.out_len = out_len
        self.input_seq_len = input_seq_len
        self.use_trend_residual_split = use_trend_residual_split
        self.max_hops = max_hops

        self.critical_path_adj = None
        if use_critical_paths and coordinates is not None and pollution_data is not None:
            path_finder = MultiHopCriticalPathFinder(
                coordinates=coordinates, pollution_data=pollution_data,
                correlation_threshold=correlation_threshold,
                distance_threshold=distance_threshold, max_hops=max_hops,max_paths_to_expand=max_paths_to_expand
            )
            self.critical_path_adj, _ = path_finder.find_critical_paths()

        self.stem = Stem(input_dim, hidden_feats, input_seq_len)
        self.short_blocks = nn.ModuleList([
            LocalBranchBlock(hidden_feats=hidden_feats,
                             geo_adj=geo_adj,
                             critical_adj=self.critical_path_adj,
                             fusion_alpha=fusion_alpha,
                             dropout=dropout,
                             device=device,
                             edge_bias_scale=1.0)
            for _ in range(num_short_blocks)
        ])

        self.short_conv1 = nn.Conv2d(hidden_feats, hidden_feats, kernel_size=(1, 1))
        self.short_conv2 = nn.Conv2d(hidden_feats, 1, kernel_size=(1, 1))
        self.short_time_mlp = nn.Linear(self.input_seq_len, out_len)

        num_nodes = geo_adj.shape[0]
        self.long_encoder = GlobalEncoderSAL(
            num_nodes=num_nodes,
            in_steps=input_seq_len,
            out_steps=out_len,
            input_dim=input_dim,
            hidden_feats=hidden_feats,
            num_layers=num_long_blocks,
            num_heads=4,
            feed_forward_mult=4,
            dropout=dropout,
            causal_time=True,
            use_node_emb=True,
            use_time_pe=True
        )

        self.fusion = EffectivePSOFusionHead(population_size=15, max_iter=8)

    def forward(self, x, current_epoch=None, val_loader=None):
        B, T, N, C = x.shape

        if self.use_trend_residual_split:
            x_local, x_global = split_trend_residual(x, k=min(12, T))
        else:
            x_local, x_global = x, x

        h = self.stem(x_local)                # (B,T,N,H)
        for blk in self.short_blocks:
            h = blk(h)                         # (B,T,N,H)

        hs = self.short_conv1(h.permute(0, 3, 2, 1))  # (B,H,N,T)
        hs = self.short_conv2(hs).squeeze(1)          # (B,N,T)
        out_short = self.short_time_mlp(hs).permute(0, 2, 1)  # (B,out_len,N)

        out_long = self.long_encoder(x_global).squeeze(-1)    # (B,out_len,N)

        out_fused, alpha = self.fusion(out_short, out_long)

        return out_short, out_long, out_fused, alpha
