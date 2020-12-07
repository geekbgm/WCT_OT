"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import torch
import numpy as np
from PIL import Image


#covariance 구하기
def cov(feat, iden=False, device='gpu'):

    size = feat.size()
    mean = torch.mean(feat, 1) # row별로 mean을 구한다. 
    mean = mean.unsqueeze(1).expand_as(feat)#feature size만큼 mean vector를 expand

    _feat = feat.clone()
    _feat -= mean #feature의 각 원소에 평균을 빼준다.

    cov = torch.mm(_feat, _feat.t()).div(size[1] - 1) #sample covariance
    
    # if iden:
    #     cov += torch.eye(size[0]).to(device)

    return cov

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


def wct_core(cont_feat, styl_feat, weight=1, registers=None, device='gpu'):

    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    if not registers:
        #content svd
        c_cov = cov(cont_feat, iden=False, device=device)
        
        #style svd
        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = torch.mean(styl_feat, 1) # style feature의 row mean vector
        s_cov = cov(styl_feat, iden=False, device=device)

        c_cov1 = (c_cov).pow(0.5)
        c_cov2 = (c_cov).pow(-0.5)

        step1 = torch.mm(c_cov1,s_cov)
        step2 = torch.mm(step1,c_cov1)
        step3 = (step2).pow(0.5)
        mapping = torch.mm(torch.mm(c_cov2,step3),c_cov2)
        cf_result = torch.mm(mapping, cont_feat)

   
    #content feature의 outproduct의 eigenvalue decomposition의 결과로 whitening 연산시작

    targetFeature = cf_result + s_mean.unsqueeze(1).expand_as(cf_result) #mean을 더해야한다.

    #targetFeature.clamp_(cont_min, cont_max) #content feature의 최대값과 최소값 사이로 제약을 건다.

    return targetFeature

def wct_core2(cont_feat, styl_feat, weight=1, registers=None, device='gpu'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    if not registers:
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = torch.mean(styl_feat, 1)
        _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        
        # 불필요한 채널을 제거하기위해 get_rank test시 줄여주는 경우가 많음
        # svd말고 다른 것을 이용해서 실험 -
        k_s = get_rank(s_e, styl_feat.size()[0])
        s_d = (s_e[0:k_s]).pow(0.5)
        EDE1 = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))

        if registers is not None:
            registers['EDE'] = EDE
            registers['s_mean'] = s_mean
            registers['c_v'] = c_v
            registers['c_e'] = c_e
    else:
        EDE1 = registers['EDE']
        s_mean = registers['s_mean']
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d1 = (c_e[0:k_c]).pow(0.25)
    c_d2 = (c_e[0:k_c]).pow(-0.5)
    EDE2 = torch.mm(torch.mm(c_v[:, 0:k_c], torch.diag(c_d1)), (c_v[:, 0:k_c].t()))
    EDE3 = torch.mm(torch.mm(c_v[:, 0:k_c], torch.diag(c_d2)), (c_v[:, 0:k_c].t()))
    
    # TODO could be more fast
    step1 = torch.mm(EDE3, cont_feat)
    step2 = torch.mm(EDE2, step1)
    step3 = torch.mm(EDE1,step2)
    step4 = torch.mm(EDE2,step3)
    step5 = torch.mm(EDE3,step4)

    targetFeature = step5 + s_mean.unsqueeze(1).expand_as(step5)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature

def wct_core_segment(content_feat, style_feat, content_segment, style_segment,
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

        #여기서부터 가우시안이 아니네.. 맞아?
        masked_content_feat = torch.index_select(content_feat_view, 1, content_index)
        masked_style_feat = torch.index_select(style_feat_view, 1, style_index)

        _target_feature = wct_core(content_feat, style_feat, weight, registers, device=device)
        
        # if torch.__version__ >= '0.4.0':
        #     # XXX reported bug in the original repository
        #     new_target_feature = torch.transpose(target_feature, 1, 0)
        #     new_target_feature.index_copy_(0, content_index,
        #                                    torch.transpose(_target_feature, 1, 0))
        #     target_feature = torch.transpose(new_target_feature, 1, 0)
        # else:
        #     target_feature.index_copy_(1, content_index, _target_feature)

    return target_feature

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

def feature_wct(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=1, device='gpu'):
    # if label_set is not None:
    #     target_feature = wct_core_segment(content_feat, style_feat, content_segment, style_segment,
    #                                       label_set, label_indicator, weight, registers, device=device)
    # else:
    target_feature = wct_core(content_feat, style_feat, device=device)

    target_feature = target_feature.view_as(content_feat) #content feature의 size와 맞춘다.
    #target_feature = alpha * target_feature + (1 - alpha) * content_feat # content feature와 convex combination

    return target_feature

def feature_wct2(content_feat, style_feat, content_segment=None, style_segment=None,
                label_set=None, label_indicator=None, weight=1, registers=None, alpha=1, device='gpu'):
    if label_set is not None:
        target_feature = wct_core_segment2(content_feat, style_feat, content_segment, style_segment,
                                          label_set, label_indicator, weight, registers, device=device)
    else:
        target_feature = wct_core2(content_feat, style_feat, device=device)

    target_feature = target_feature.view_as(content_feat) #content feature의 size와 맞춘다.
    target_feature = alpha * target_feature + (1 - alpha) * content_feat # content feature와 convex combination

    return target_feature