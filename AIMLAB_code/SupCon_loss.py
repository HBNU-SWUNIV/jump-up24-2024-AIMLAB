import torch
import torch.nn as nn
import torch.nn.functional as F

class LivePatchSpoofSampleLoss_ViT(nn.Module):
    def __init__(self, temperature=0.07, spoof_average_pool=True, sampling_rate=1.0):
        super(LivePatchSpoofSampleLoss_ViT, self).__init__()
        self.temperature = temperature
        self.spoof_average_pool = spoof_average_pool
        self.sampling_rate = sampling_rate

    def forward(self, features, labels):
        _, num_patches, feature_dim = features.size()

        # live feature
        live_features = features[labels == 1] 
        live_features = live_features.reshape(-1, feature_dim)      # [batch_size, patches, feature_dim] --> [batch_size*patches, feature_dim]
        live_features = F.normalize(live_features, p=2, dim=1)      # normalize

        # sampling patches
        if self.sampling_rate < 1.0:
            sampling_mask = torch.rand(live_features.shape[0]) < self.sampling_rate
            live_features = live_features[sampling_mask]

        # spoof feature (aggregation)
        spoof_features = features[labels == 0]
        if self.spoof_average_pool:
            spoof_features = spoof_features.mean(axis=1)            # [batch_size, feature_dim]
        else:
            spoof_features = spoof_features.max(axis=1)[0]             # [batch_size, feature_dim]
        spoof_features = F.normalize(spoof_features, p=2, dim=1)    # normalize        

        # calculate inner products
        live_similarity_matrix = torch.matmul(live_features, live_features.T)
        live_similarity_matrix = live_similarity_matrix / (self.temperature + 1e-8)
        #live_similarity_matrix = live_similarity_matrix * (1 - torch.eye(*live_similarity_matrix.size()).cuda())    # ignore self-similarity

        live_spoof_similarity_matrix = torch.matmul(live_features, spoof_features.T)
        live_spoof_similarity_matrix = live_spoof_similarity_matrix / (self.temperature + 1e-8)
        
        # for numerical stability
        sim_max, _ = torch.max(live_similarity_matrix, dim=1, keepdim=True)
        live_similarity_matrix = live_similarity_matrix - sim_max.detach()        
        live_spoof_similarity_matrix = live_spoof_similarity_matrix - sim_max.detach()  

        # calc log prob
        exp_live_similarity = torch.exp(live_similarity_matrix)
        exp_live_spoof_similarity = torch.exp(live_spoof_similarity_matrix)

        reweight = exp_live_similarity.shape[1] / exp_live_spoof_similarity.shape[1]
        denom = exp_live_similarity.sum(1, keepdim=True) + reweight * exp_live_spoof_similarity.sum(1, keepdim=True)
        log_prob = live_similarity_matrix - torch.log(denom + 1e-8)
        
        # normalize loss
        loss = -log_prob.mean() # log_prob.sum(1)/num_lives/num_lives 

        return loss