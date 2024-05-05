import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch.autograd import Function
import warnings

try:
    import models._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    print('------------ FAILED import models._ext as _ext')
    print('------------ NOT USING FPS')
    
    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    """_ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )"""

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    idx = idx.long()
    new_points = points[batch_indices, idx, :]
    return new_points


# This implementation is very slow and should not be used unless you can't get the CUDA version to compile
# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
#     distance = torch.ones(B, N, device=device)* 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
#     batch_indices = torch.arange(B, dtype=torch.long, device=device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx.to(torch.int64))

    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # dists = square_distance(new_xyz, xyz)  # B x npoint x N
    # idx = dists.argsort()[:, :, :nsample]

    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_4d(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, t, N, 3]
        points: input points data, [B, t, N, D]
    Return:
        new_xyz: sampled points position data, [B, t, npoint, nsample, 3]
        new_points: sampled points data, [B, t, npoint, nsample, 3+D]
    """
    B, t, N, C = xyz.shape
    S = npoint
    D = points.shape[-1]
    fps_idx = farthest_point_sample(xyz[:, 0], npoint)  # [B, npoint, C], use frist drame for fps
    xyz = xyz.reshape(-1, N, C)
    new_xyz = index_points(xyz, fps_idx.unsqueeze(1).repeat([1, t, 1]).reshape(-1, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B*t, S, 1, C)  # shouldnt this also be scaled to unit sphere?

    if points is not None:
        grouped_points = index_points(points.reshape(-1, N, D), idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    new_xyz = new_xyz.reshape(B, t, npoint, C)
    new_points = new_points.reshape(B, t, npoint, nsample, -1)

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all_4d(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, t, N, C = xyz.shape
    new_xyz = torch.zeros(B, t, 1, C).to(device)
    grouped_xyz = xyz.view(B, t, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, t, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, 1))
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
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 1, 3, 2)
            points = points.reshape(-1, points.shape[-2], points.shape[-1])

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [b*t, npoint, k]
        # new_points: sampled points data, [b*t, npoint, nsample, d+k]
        new_points = new_points.reshape(b, t, new_points.shape[-3], new_points.shape[-2], new_points.shape[-1])
        new_points = new_points.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]
        new_points = new_points.permute(0, 3, 1, 2)
        new_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)
        return new_xyz, new_points

class PointNetPP4DSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, temporal_conv=4):
        super(PointNetPP4DSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv3d(last_channel, out_channel, [1, temporal_conv, 1], 1, padding='same'))
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
            new_xyz, new_points = sample_and_group_all_4d(xyz.reshape(b, t, n, k), points)
        else:
            new_xyz, new_points = sample_and_group_4d(self.npoint, self.radius, self.nsample,
                                                      xyz.reshape(b, t, n, k), points)

        # new_xyz: sampled points position data, [b*t, npoint, k]
        # new_points: sampled points data, [b*t, npoint, nsample, d+k]
        new_points = new_points.reshape(b, t, new_points.shape[-3], new_points.shape[-2], new_points.shape[-1])
        new_points = new_points.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0]
        new_points = new_points.permute(0, 3, 1, 2)
        new_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel # + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv3d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm3d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

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
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 1, 3, 2)
            points = points.reshape(-1, points.shape[-2], points.shape[-1])

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.reshape(b, t, grouped_points.shape[-3], grouped_points.shape[-2], grouped_points.shape[-1])
            grouped_points = grouped_points.permute(0, 4, 2, 1, 3)  # [b, d+k, npoint, t, nsample]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, -1)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.reshape(b, t, new_xyz.shape[-2], new_xyz.shape[-1]).permute(0, 1, 3, 2)
        new_points_concat = torch.cat(new_points_list, dim=1)
        new_points_concat = new_points_concat.permute(0, 3, 1, 2)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetSetAbstractionOriginal(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstractionOriginal, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        xyz = xyz.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        
        #! fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        fps_inds = xyz
        
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


#farthest_point_sample = FurthestPointSampling.apply