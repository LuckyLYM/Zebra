from numpy import average
import torch
from torch import nn
from utils.util import MergeLayer
import sys

class TemporalAttentionLayer(torch.nn.Module):

  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, output_dimension, n_head=2,dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head
    self.feat_dim = n_node_features
    self.time_dim = time_dim
    self.query_dim = n_node_features + time_dim
    self.key_dim = n_neighbors_features + time_dim + n_edge_features

    # ------ MLP model
    # self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    # self.fc2 = torch.nn.Linear(dim3, dim4)

    # output_dimension = n_node_features
    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)


    # query: source_node_embedding + time_feature
    # K & V: node_embedding + time_feature + edge_feature
    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)


  # self.aggregate(layer_id, source_embedding, combined_nodes_time_embedding, combined_neighbor_embeddings,combined_edge_time_embeddings,combined_edge_features,combined_mask)
  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):
    '''
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors, time_dim]
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors], true means no enighbors    'mask = neighbors_torch == 0'

    return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    '''





  
    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
    key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)

    # [660,10,172] * 3
    #print(f'neighbor feature {neighbors_features.shape}, edge_feature {edge_features.shape}, neighbors_time_features {neighbors_time_features.shape}, key {key.shape}')
    #sys.exit(1)

    # Reshape tensors so to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]


    #print(f'src_node_features {src_node_features_unrolled.shape}, query {query.shape}, key {key.shape}')
    #print(f'neighbors_padding_mask {neighbors_padding_mask.shape}')

    # *--- compute mask of which source nodes have no valid neighbors
    # neighbors_padding_mask, shape: [batch_size, n_neighbors]
    # torch.all(): test if all elements in input evaluate to True, dim=1 means by row
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True) #[600,10]=>[600,1]

    # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
    # force the attention to just 'attend' on this neighbor (which has the same features as all
    # the others since they are fake neighbors) and will produce an equivalent result to the
    # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
    #print(f'before {neighbors_padding_mask[0]}')
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    #print(f'after {neighbors_padding_mask[0]}')
    #print(f'invalid_neighborhood_mask {invalid_neighborhood_mask.shape}, neighbors_padding_mask[ {neighbors_padding_mask.shape}')


    # *------ aggregate neighborhood information
    # *------ the padding mask is used for dropout
    # *------ key_padding_mask will ignore certain keys
    # ------ key_padding_mask – if provided, specified padding elements in the key will be ignored by the attention. When given a binary mask and a value is True, the corresponding value on the attention layer will be ignored. When given a byte mask and a value is non-zero, the corresponding value on the attention layer will be ignored
    # ------ ignore the embeddings of fake 0 neighbors
    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,key_padding_mask=neighbors_padding_mask)

    #print(f'attn_output {attn_output.shape}, attn_output_weights {attn_output_weights.shape}')

    # ----- remove redundant dimension
    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    averaged_weights_before_fill= torch.mean(attn_output_weights,dim=0)

    # *----- use zero embedding for nodes without neighbors
    # Source nodes with no neighbors have an all zero attention output. The attention output is
    # then added or concatenated to the original source node features and then fed into an MLP.
    # This means that an all zero vector is not used.
    # set of target node embeddings to 0 if it has no neighbors
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0) #[600,344]
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0) #[600,10]
    
    #print(f'out {attn_output.shape}, weights {attn_output_weights.shape}, invalid_mask {invalid_neighborhood_mask.shape}, neighbor_mask {neighbors_padding_mask.shape}')


    
    averaged_weights_after_fill= torch.mean(attn_output_weights,dim=0)
    
    '''
    if attn_output_weights.shape[0]==6600:
      print(f'averaged_weights_before: {averaged_weights_before_fill.data}')
      print(f'averaged_weights_after: {averaged_weights_after_fill.data}')
    '''
    
    #print(f'weights {attn_output_weights.shape}')
    #print(f'attn_output {attn_output.shape}, attn_output_weights {attn_output_weights.shape}')

    # ----- apply MLP to the concatenation of atte_output and src_node_features
    # Skip connection with temporal attention over neighborhood and the features of the node itself
    attn_output = self.merger(attn_output, src_node_features)

    # output: source_embedding, _
    return attn_output, attn_output_weights

