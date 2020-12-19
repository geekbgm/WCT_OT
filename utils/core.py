"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import torch
import numpy as np
from PIL import Image


def cov(feat, device='gpu'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    return conv

def svd(feat, iden=False, device='gpu'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())

    if iden:
        conv += torch.eye(size[0]).to(device)
    u, e, v = torch.svd(conv, some=False)
    return u, e, v

def get_squeeze_feat(feat):
    _feat = feat.squeeze(0) #
    size = _feat.size(0)

    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):

    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break

    return r

    # 불필요한 채널을 제거하기위해 get_rank test시 줄여주는 경우가 많음    
    # svd말고 다른 것을 이용해서 실험 -
def wct_core2(cont_feat, styl_feat, weight=1, registers=None, device='gpu'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean


    _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    styl_feat = get_squeeze_feat(styl_feat)
    s_mean = torch.mean(styl_feat, 1)
    _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        
    k_s = get_rank(s_e, styl_feat.size()[0])
    s_d = s_e[0:k_s]
    s_cov = torch.mm(torch.mm(s_v[:,0:k_s], torch.diag(s_d) * weight), (s_v[:,0:k_s].t()))
 
    
    k_c = get_rank(c_e, cont_feat.size()[0])

    c_d1 = (c_e[0:k_c]).pow(0.5)
    c_d2 = (c_e[0:k_c]).pow(-0.5)
    c_cov1 = torch.mm(torch.mm(c_v[:,0:k_c], torch.diag(c_d1)), (c_v[:,0:k_c].t()))
    c_cov2 = torch.mm(torch.mm(c_v[:,0:k_c], torch.diag(c_d2)), (c_v[:,0:k_c].t()))

    A = torch.mm(torch.mm(c_cov1,s_cov),c_cov1)

    A_u, A_s, A_v = torch.svd(A, some=False)

    r_A = get_rank(A_s, A.size()[0])
    A_d = (A_s[0:r_A]).pow(0.5)

    A_1 = torch.mm(torch.mm(A_v[:,0:r_A], torch.diag(A_d)), (A_u[:,0:r_A].t())) # u와 v를 swithing 해봄 

    # TODO could be more fast
    step1 = torch.mm(c_cov2, cont_feat)
    step2 = torch.mm(A_1, step1)
    step3 = torch.mm(c_cov2,step2)

    targetFeature = step3 + s_mean.unsqueeze(1).expand_as(step3)
    
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature 

def wct_core_segment2(content_feat, style_feat, content_segment, style_segment,
                     label_set, label_indicator, weight=1, registers=None,
                     device='gpu'):

    def resize(feat, target):
        size = (target.size(2), target.size(1))
        if len(feat.shape) == 2:
            return np.asarray(Image.fromarray(feat).resize(size, Image.NEAREST))
        else:
            return np.asarray(Image.fromarray(feat, mode='RGB').resize(size, Image.NEAREST))

    def get_index(feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None

        return torch.LongTensor(mask[0]).to(device)

    squeeze_content_feat = content_feat.squeeze(0)
    squeeze_style_feat = style_feat.squeeze(0)

    content_feat_view = squeeze_content_feat.view(squeeze_content_feat.size(0), -1).clone()
    style_feat_view = squeeze_style_feat.view(squeeze_style_feat.size(0), -1).clone()

    resized_content_segment = resize(content_segment, squeeze_content_feat)
    resized_style_segment = resize(style_segment, squeeze_style_feat)

    target_feature = content_feat_view.clone()

    for label in label_set:
        if not label_indicator[label]:
            continue
        content_index = get_index(resized_content_segment, label)
        style_index = get_index(resized_style_segment, label)
        if content_index is None or style_index is None:
            continue
        masked_content_feat = torch.index_select(content_feat_view, 1, content_index)
        masked_style_feat = torch.index_select(style_feat_view, 1, style_index)

        _target_feature = wct_core2(masked_content_feat, masked_style_feat, weight, registers, device=device)
        
        if torch.__version__ >= '0.4.0':
            # XXX reported bug in the original repository
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, content_index,
                                           torch.transpose(_target_feature, 1, 0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
        else:
            target_feature.index_copy_(1, content_index, _target_feature)

    return target_feature


def feature_wct2(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=0.5, device='gpu'):
    if label_set is not None:
        target_feature = wct_core_segment2(content_feat, style_feat, content_segment, style_segment,
                                          label_set, label_indicator, weight, registers, device=device)
    else:
        target_feature = wct_core2(content_feat, style_feat, device=device)

    target_feature = target_feature.view_as(content_feat) #content feature의 size와 맞춘다.
    target_feature = alpha * target_feature + (1 - alpha) * content_feat # content feature와 convex combination

    return target_feature