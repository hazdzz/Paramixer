import torch
import torch.nn as nn
import torch.nn.init as init
from model import embedding, paramixer
from torch import Tensor


class SingleClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class) -> None:
        super(SingleClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.mlp_dim = mlp_dim
        self.num_class = num_class
        self.linear1 = nn.Linear(encoder_dim, mlp_dim)
        self.flatten_linear1 = nn.Linear(max_seq_len * encoder_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, num_class)
        self.leakyrelu = nn.LeakyReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.flatten_linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.xavier_normal_(self.linear2.weight, gain=1.0)

    def pooling(self, input: Tensor, mode: str) -> Tensor:
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.view(input.size(0), -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded: Tensor) -> Tensor:
        pooled = self.pooling(encoded, self.pooling_type)

        if self.pooling_type == 'FLATTEN':
            pooled1 = self.flatten_linear1(pooled)
        else:
            pooled1 = self.linear1(pooled)
        pooled1 = self.leakyrelu(pooled1)
        pooled2 = self.linear2(pooled1)
        classified = self.logsoftmax(pooled2)

        return classified


class DualClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class, 
                 interaction) -> None:
        super(DualClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.interaction = interaction
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.mlp_dim = mlp_dim
        self.linear1 = nn.Linear(encoder_dim * 2, mlp_dim)
        self.nli_linear1 = nn.Linear(encoder_dim * 4, mlp_dim)
        self.flatten_linear1 = nn.Linear(max_seq_len * encoder_dim * 2, mlp_dim)
        self.flatten_nli_linear1 = nn.Linear(max_seq_len * encoder_dim * 4, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, mlp_dim // 2)
        self.linear3 = nn.Linear(mlp_dim // 2, num_class)
        self.leakyrelu = nn.LeakyReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.nli_linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.flatten_linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.flatten_nli_linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.xavier_normal_(self.linear3.weight, gain=1.0)

    def pooling(self, input: Tensor, mode: str):
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.view(input.size(0), -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded_1: Tensor, encoded_2: Tensor) -> Tensor:
        pooled_1 = self.pooling(encoded_1, self.pooling_type)
        pooled_2 = self.pooling(encoded_2, self.pooling_type)
        if self.interaction == 'NLI':
            # NLI interaction style
            pooled = torch.cat([pooled_1, 
                                pooled_2, 
                                pooled_1 * pooled_2, 
                                pooled_1 - pooled_2], dim=-1)
            if self.pooling_type == 'FLATTEN':
                pooled_layer1 = self.flatten_nli_linear1(pooled)
            else:
                pooled_layer1 = self.nli_linear1(pooled)
        else:
            pooled = torch.cat([pooled_1, pooled_2], dim=-1)
            if self.pooling_type == 'FLATTEN':
                pooled_layer1 = self.flatten_linear1(pooled)
            else:
                pooled_layer1 = self.linear1(pooled)
        pooled_layer1 = self.leakyrelu(pooled_layer1)
        pooled_layer2 = self.linear2(pooled_layer1)
        pooled_layer2 = self.leakyrelu(pooled_layer2)
        pooled_layer3 = self.linear3(pooled_layer2)
        classified = self.logsoftmax(pooled_layer3)

        return classified


class ParamixerLRASingle(nn.Module):
    def __init__(self, args, device) -> None:
        super(ParamixerLRASingle, self).__init__()
        self.embedding = embedding.ParamixerEmbedding(args.pe_type, 
                                                      args.pooling_type, 
                                                      args.vocab_size, 
                                                      args.max_seq_len, 
                                                      args.embed_size, 
                                                      args.embed_drop_prob)
        self.paramixer = paramixer.Paramixer(args, device)
        self.classifier = SingleClassifier(args.pooling_type, 
                                           args.max_seq_len, 
                                           args.encoder_dim, 
                                           args.mlp_dim, 
                                           args.num_class
                                           )

    def forward(self, input: Tensor) -> Tensor:
        embeded = self.embedding(input)
        encoded = self.paramixer(embeded)
        classified = self.classifier(encoded)

        return classified


class ParamixerLRADual(nn.Module):
    def __init__(self, args, device) -> None:
        super(ParamixerLRADual, self).__init__()
        self.embedding = embedding.ParamixerEmbedding(args.pe_type, 
                                                      args.pooling_type, 
                                                      args.vocab_size, 
                                                      args.max_seq_len, 
                                                      args.embed_size, 
                                                      args.embed_drop_prob)
        self.paramixer = paramixer.Paramixer(args, device)
        self.classifier = DualClassifier(args.pooling_type, 
                                         args.max_seq_len, 
                                         args.encoder_dim, 
                                         args.mlp_dim, 
                                         args.num_class,
                                         args.interaction
                                         )

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        embeded1 = self.embedding(input1)
        embeded2 = self.embedding(input2)
        encoded1 = self.paramixer(embeded1)
        encoded2 = self.paramixer(embeded2)
        classified = self.classifier(encoded1, encoded2)

        return classified