from typing import Any
import torch
import torch.nn.functional as F

from torch import nn

import model


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(model.simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class SaintUpdated(nn.Module):
    def __init__(self, num_continuous, num_categories, dim, dim_out, depth, heads, attn_dropout, ff_dropout,
                 mlp_hidden_mults, cont_embeddings, scalingfactor, attentiontype, final_mlp_style, y_dim, categories):
        super(SaintUpdated, self).__init__()

        # Your initialization code here

        # Assign num_categories to self.num_categories
        self.num_categories = num_categories


    class SaintUpdated:
        def __init__(self, num_continuous, dim, num_categories):
            self.num_continuous = num_continuous
            self.dim = dim
            self.num_categories = num_categories
            # assign other parameters as needed

    # Creating an instance of the SAINT class
    saint_instance = SaintUpdated(num_categories=10, dim=8, num_continuous=5)
    print(saint_instance.num_categories)  # Access the attribute

    class MyClass:
        def __init__(self):
            super().__init__()  # Correctly indented within the constructor

    categories = 10  # Assign a positive number of categories

    # Check if 'categories' is positive
    assert categories > 0, 'number of each category must be positive'

    # assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

    # categories related calculations

    def my_function(self, categories=10):
        num_categories = categories
        # Rest of the function code

        # Rest of the function code
        # If categories is an integer representing the number of categories

        # self.num_categories = len(categories)
        self.num_unique_categories = categories  # Assuming categories is an integer count
        # self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        # Create an example tensor (replace this with your actual data)
        categories = torch.tensor([1, 2, 3])

        # Specify the padding length
        padding = (1, 0)

        # Define a value for padding
        num_special_tokens = 0
        categories_offset = F.pad(categories.clone().detach(), (1, 0), value=num_special_tokens)

        # categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = model.nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = model.nn.ModuleList(
                [model.simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = model.nn.ModuleList([model.simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continuous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = model.Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = model.RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = model.MLP(all_dimensions, act=mlp_act)
        self.embeds = model.nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = model.F.pad(
            model.model.torch.Tensor(self.num_categories).fill_(2).type(model.model.torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = model.F.pad(
            model.model.torch.Tensor(self.num_continuous).fill_(2).type(model.model.torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = model.nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = model.nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = model.nn.Embedding(2, self.dim)
        self.pos_encodings = model.nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = model.simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = model.simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous, model.model.np.ones(self.num_continuous).astype(int))

        self.mlpfory = model.simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = model.simple_MLP([dim * (self.num_continuous + self.num_categories),
                                        6 * dim * (self.num_continuous + self.num_categories) // 5,
                                        dim * (self.num_continuous + self.num_categories) // 2])
        self.pt_mlp2 = model.simple_MLP([dim * (self.num_continuous + self.num_categories),
                                         6 * dim * (self.num_continuous + self.num_categories) // 5,
                                         dim * (self.num_continuous + self.num_categories) // 2])

    def forward(self, x_categ, x_cont):

        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:, :self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories:, :])
        return cat_outs, con_outs