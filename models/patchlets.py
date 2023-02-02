import torch
import torch.nn as nn
import faiss
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as faiss_torch_knn_gpu
import models.pointnet2_utils as utils
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from scipy.spatial import cKDTree
import numpy as np
from pykeops.torch import Vi, Vj
import time
# import torch_cluster

def get_knn(x1, x2, k=16, res=None, method='faiss_gpu'):

    if method == 'faiss_gpu':
        distances, idxs = faiss_torch_knn_gpu(res, x1, x2, k=k)
    if method == 'faiss_cpu':
        distances, idxs = faiss.knn(np.ascontiguousarray(x1.detach().cpu().numpy(), dtype=np.float32),
                                    np.ascontiguousarray(x2.detach().cpu().numpy(), dtype=np.float32), k=k)
        distances, idxs = torch.tensor(distances, device=x1.device).cuda(), torch.tensor(idxs, device=x1.device).cuda()
    if method == 'spatial':
        tree = cKDTree(x2.detach().cpu().numpy())
        distances, idxs = tree.query(x1.detach().cpu().numpy(), k, workers=-1)
        distances, idxs = torch.tensor(distances).cuda(), torch.tensor(idxs).cuda()
    if method == 'keops': #supports batch operaations
        X_i = Vi(0, x1.shape[-1])
        X_j = Vj(1, x2.shape[-1])
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        KNN_fun = D_ij.Kmin_argKmin(k, dim=1)
        distances, idxs = KNN_fun(x1.contiguous(), x2.contiguous())
    # if method =='geometric':
    #     with torch.no_grad():
    #         b, n, d = x1.shape
    #         batch_x = torch.arange(b, device=x1.device).unsqueeze(1).repeat(1, n).reshape(-1)
    #         batch_y = batch_x.reshape(-1)
    #         idxs = torch_cluster.knn(x1.reshape(-1, 3), x2.reshape(-1, 3), k, batch_x, batch_y)[0].reshape(b, n, k)
    #         distances = torch.zeros_like(idxs)
    return distances, idxs


class PatchletsExtractor(nn.Module):
    def __init__(self, k=16, sample_mode='nn', npoints=None, add_centroid_jitter=None):
        super(PatchletsExtractor, self).__init__()
        #TODO consider implementing a radius threshold
        self.k = k
        self.sample_mode = sample_mode
        self.npoints = npoints
        self.add_centroid_jitter = add_centroid_jitter
        self.res = faiss.StandardGpuResources()
        self.res.setDefaultNullStreamAllDevices()

    def forward(self, point_seq, feat_seq=None):
        b, t, n, d = point_seq.shape
        n_original = n
        if feat_seq is None:
            feat_seq = point_seq
            d_feat = d
        else:
            d_feat = feat_seq.shape[-1]

        x1 = point_seq
        out_x = torch.empty_like(x1)
        x2 = torch.cat([point_seq[:, [0]], point_seq], dim=1)[:, :-1]

        if self.sample_mode == 'nn': #TODO imlement weighted random sample mode ?
            selected_point_idx = 0


        # Not supporting batches, including Ori's noise thingy
        patchlets = torch.empty(b, t, n, self.k, device=point_seq.device, dtype=torch.long)
        distances_i = torch.empty(b,  t, n, self.k, device=point_seq.device)
        idxs_i = torch.empty(b, t, n, self.k, device=point_seq.device, dtype=torch.long)
        patchlet_points = torch.empty(b, t, n, self.k, 3, device=point_seq.device)
        patchlet_feats = torch.empty(b, t, n, self.k, d_feat, device=point_seq.device)

        # loop over the data to reorder the indices to form the patchlets
        x_current = x1[:, 0]
        feat_seq_2 = torch.cat([feat_seq[:, [0]], feat_seq], dim=1)[:, :-1]
        for i in range(0, t):
            x_next = x2[:, i]
            distances, idxs = get_knn(x_current, x_next, k=self.k, res=self.res, method='keops')
            if self.sample_mode == 'nn':
                x_current = utils.index_points(x_next, idxs)[:, :, 0, :]
            elif self.sample_mode == 'rand':
                rand_idx = torch.randint(self.k, (b, n, 1, 3), device=x_next.device, dtype=torch.int64)
                x_current = torch.gather(utils.index_points(x_next, idxs).squeeze(), dim=2, index=rand_idx).squeeze()
            elif self.sample_mode == 'mean':
                x_current = utils.index_points(x_next, idxs).mean(2)

            # x1[:, i] = x_current
            out_x[:, i] = x_current
            if self.add_centroid_jitter is not None:
                x_current = x_current + self.add_centroid_jitter*torch.randn_like(x_current)

            distances_i[:, i], idxs_i[:, i] = distances, idxs
            patchlets[:, i] = idxs_i[:, i]
            patchlet_points[:, i] = utils.index_points(x_next, idxs).squeeze()
            patchlet_feats[:, i] = utils.index_points(feat_seq_2[:, i], idxs).squeeze()


        distances = distances_i
        idxs = idxs_i

        patchlet_feats = patchlet_feats.reshape(b*t, n, self.k, d_feat)
        patchlet_points = patchlet_points.reshape(b * t, n, self.k, 3)
        idxs = idxs.reshape(b*t, n, self.k)
        distances = distances.reshape(b*t, n, self.k)
        patchlets = patchlets.reshape(b*t, n, self.k)

        fps_idx = None
        # downsample
        if self.npoints is not None:
            fps_idx = utils.farthest_point_sample(point_seq[:, 0].contiguous(), self.npoints).to(torch.int64)
            patchlet_points = utils.index_points(patchlet_points, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints))
            patchlet_feats = utils.index_points(patchlet_feats, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints))
            distances = utils.index_points(distances, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints))
            idxs = utils.index_points(idxs, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints))
            patchlets =  utils.index_points(patchlets, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, self.npoints))
            n = self.npoints

        # reshape all to bxtxnxk
        distances, idxs = distances.reshape(b, t, n, self.k), idxs.reshape(b, t, n, self.k)
        patchlets, patchlet_points = patchlets.reshape(b, t, n, self.k), patchlet_points.reshape(b, t, n, self.k, d)
        patchlet_feats = patchlet_feats.reshape(b, t, n, self.k, d_feat)

        normalized_patchlet_points = patchlet_points - patchlet_points[:, 0, :, [0], :].unsqueeze(1) # normalize the patchlet around the center point of the first frame
        patchlet_feats = torch.cat([patchlet_feats, normalized_patchlet_points], -1)

        return {'idx': idxs, 'distances': distances, 'patchlets': patchlets,
                'patchlet_points': patchlet_points, 'patchlet_feats': patchlet_feats,
                'normalized_patchlet_points': normalized_patchlet_points, 'fps_idx': fps_idx,
                'x_current': out_x.reshape(b, t, n_original, 3)}


class PatchletTemporalConv(nn.Module):
    def __init__(self, in_channel, temporal_conv, k, mlp):
        super(PatchletTemporalConv, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, 1))
            # self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, k], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel

        self.temporal_conv = nn.Conv2d(out_channel, out_channel, [1, temporal_conv], 1, padding='same')
        self.bnt = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        x = torch.max(x, -1)[0] # pool neighbors to get patch representation
        # x = torch.mean(x, -1)  # pool neighbors to get patch representation
        x = F.relu(self.bnt(self.temporal_conv(x))) # convolve temporally to improve patch representation
        return x.permute(0, 3, 2, 1)

class PointMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointMLP, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x = F.relu(bn(conv(x)))

        return x.permute(0, 2, 3, 1)


class PointNet2PatchletsSA(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, k=16, temporal_conv=4):
        super(PointNet2PatchletsSA, self).__init__()
        self.k = k
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, self.k], 1, padding='same'))
            self.mlp_bns.append(nn.BatchNorm3d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        b, t, k, n = xyz.shape
        xyz = xyz.reshape(-1, k, n)
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 1, 3, 2)
            # points = points.reshape(-1, points.shape[-2], points.shape[-1])

        if self.group_all:
            new_xyz, new_points = utils.sample_and_group_all_4d(xyz.reshape(b, t, n, k), points)
        else:
            new_xyz, new_points = utils.sample_and_group_4d(self.npoint, self.radius, self.nsample,
                                                      xyz.reshape(b, t, n, k), points)

        # new_xyz: sampled points position data, [b*t, npoint, k]
        # new_points: sampled points data, [b*t, npoint, nsample, d+k]
        new_points = new_points.reshape(b, t, new_points.shape[-3], new_points.shape[-2], new_points.shape[-1])
        new_points = new_points.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0] #TODO test this with mean
        new_points = new_points.permute(0, 3, 1, 2)
        new_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)
        return new_xyz, new_points


class PointNet2Patchlets(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3, k=16):
        super(PointNet2Patchlets, self).__init__()
        self.n_frames = n_frames
        self.k = k
        self.patchlet_extractor = PatchletsExtractor(k=self.k, sample_mode='nn', npoints=512)
        self.patchlet_temporal_conv = PatchletTemporalConv(in_channel=in_channel, temporal_conv=8, k=k, mlp=[64, 64, 64])
        self.sa1 = PointNet2PatchletsSA(npoint=512, radius=0.2, nsample=32, in_channel=64+3,
                                        mlp=[64, 64, 128], group_all=False, k=8, temporal_conv=8)
        self.sa2 = PointNet2PatchletsSA(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                        mlp=[128, 128, 256], group_all=False, k=8, temporal_conv=4)
        self.sa3 = PointNet2PatchletsSA(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                        mlp=[256, 512, 1024], group_all=True, k=1, temporal_conv=4)

        self.temporal_pool = torch.nn.MaxPool3d([n_frames, 1, 1])
        self.temporal_pool_xyz = torch.nn.AvgPool3d([4, 1, 1])

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        # self.bnt = nn.BatchNorm1d(1024)
        # self.temporalconv1 = torch.nn.Conv1d(1024, 1024, int(n_frames/4), 1, padding='same')
        self.temporalconv2 = torch.nn.Conv1d(256, 256, n_frames, 1, padding='same')
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, xyz):
        b, t, d, n = xyz.shape
        # new_B = B*t

        patchlet_dict = self.patchlet_extractor(xyz.permute(0, 1, 3, 2))
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = self.patchlet_temporal_conv(patchlet_dict['normalized_patchlet_points'].permute(0, 4, 2, 1, 3)) # [b, d+k, npoint, t, nsample]

        l1_xyz, l1_points = self.sa1(xyz[:, :, :, 0, :].permute(0, 1, 3, 2), patchlet_feats.permute(0, 1, 3, 2))
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_xyz, l3_points = l3_xyz.squeeze(-1), l3_points.squeeze(-1)
        x = l3_points.permute(0, 2, 1)

        # x = F.relu(self.bnt(self.temporalconv1(x)))

        x = x.permute(0, 2, 1).reshape(b*t, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        # learn a temporal filter on all per-frame global representations
        x = F.relu(self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        x = self.fc3(x)

        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': l3_points}


class PointNet2Patchlets_v2(nn.Module):
    def __init__(self, num_class, n_frames=32, in_channel=3, k=16, sample_mode='nn', add_centroid_jitter=0.005):
        super(PointNet2Patchlets_v2, self).__init__()
        self.n_frames = n_frames
        self.k = k
        # self.point_mlp = PointMLP(in_channel=in_channel, mlp=[64, 64, 128])
        self.patchlet_extractor1 = PatchletsExtractor(k=self.k, sample_mode=sample_mode, npoints=512,
                                                      add_centroid_jitter=add_centroid_jitter)
        self.patchlet_temporal_conv1 = PatchletTemporalConv(in_channel=in_channel, temporal_conv=8, k=self.k, mlp=[64, 64, 128])
        self.patchlet_extractor2 = PatchletsExtractor(k=self.k, sample_mode=sample_mode, npoints=128,
                                                      add_centroid_jitter=add_centroid_jitter)
        self.patchlet_temporal_conv2 = PatchletTemporalConv(in_channel=128+3, temporal_conv=4, k=self.k, mlp=[128, 128, 256])
        self.patchlet_extractor3 = PatchletsExtractor(k=self.k, sample_mode=sample_mode, npoints=None,
                                                      add_centroid_jitter=add_centroid_jitter)
        self.patchlet_temporal_conv3 = PatchletTemporalConv(in_channel=256+3, temporal_conv=4, k=self.k,
                                                           mlp=[256, 512, 1024])

        # self.temporal_pool = torch.nn.MaxPool3d([n_frames, 1, 1])
        # self.temporal_pool = torch.nn.AvgPool2d(3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        # self.bnt = nn.BatchNorm1d(1024)
        # self.temporalconv1 = torch.nn.Conv1d(1024, 1024, int(n_frames/4), 1, padding='same')
        self.temporalconv2 = torch.nn.Conv1d(256, 256, n_frames, 1, padding='same')
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, xyz):
        b, t, d, n = xyz.shape

        patchlet_dict = self.patchlet_extractor1(xyz.permute(0, 1, 3, 2))
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['normalized_patchlet_points'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv1(patchlet_feats)  # [b, d+k, npoint, t, nsample]

        patchlet_dict = self.patchlet_extractor2(xyz[:, :, :, 0, :], patchlet_feats)
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv2(patchlet_feats)  # [b, d+k, npoint, t, nsample]

        patchlet_dict = self.patchlet_extractor3(xyz[:, :, :, 0, :], patchlet_feats)
        xyz = patchlet_dict['patchlet_points']
        patchlet_feats = patchlet_dict['patchlet_feats'].permute(0, 4, 2, 1, 3)
        patchlet_feats = self.patchlet_temporal_conv3(patchlet_feats)  # [b, d+k, npoint, t, nsample]


        xyz, patchlet_feats = xyz.squeeze(-1), patchlet_feats.squeeze(-1)
        x = torch.max(patchlet_feats, -2)[0]
        # x = torch.mean(patchlet_feats, -2)
        x = x.reshape(b*t, 1024)

        x = self.drop1(F.relu(self.bn1(self.fc1(x).reshape(b, t, 512).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 512))
        x = self.drop2(F.relu(self.bn2(self.fc2(x).reshape(b, t, 256).permute(0, 2, 1))).permute(0, 2, 1).reshape(-1, 256))
        # learn a temporal filter on all per-frame global representations
        x = F.relu(self.bn3(self.temporalconv2(x.reshape(b, t, 256).permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 256)))
        x = self.fc3(x)

        # x = self.temporal_pool(x.reshape(b, t, -1))

        x = F.log_softmax(x, -1)

        return {'pred': x.reshape(b, t, -1).permute([0, 2, 1]), 'features': patchlet_feats}