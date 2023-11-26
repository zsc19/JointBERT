import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)#抑制过拟合，随机删除一些神经元
        self.linear = nn.Linear(input_dim, num_intent_labels)#最后输出的维度必须是意图标签的个数

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)#抑制过拟合，随机删除一些神经元
        self.linear = nn.Linear(input_dim + num_intent_labels, input_dim)#最后输出的维度必须是意图标签的个数
        self.layerNorm = nn.LayerNorm(input_dim)
        self.self_attention = SelfAttention(input_dim, input_dim, input_dim, 0.5)


    def forward(self, intent_logits, sequence_output):
        n = len(sequence_output[1])


        # Step 1: 将intent logits复制n次，创建维度为 (batch_size, n, intent_dim) 的张量
        intent_replicas = intent_logits.unsqueeze(1).expand(-1, n, -1)

        # Step 2: 将每个token的特征和intent复制进行拼接，输出维度为 (batch_size, seq_len, input_dim + intent_dim)
        concat_inputs = torch.cat([sequence_output, intent_replicas], dim=-1)
        # print(sequence_output.device, intent_replicas.device)

        # Step 3: 经过线性层进行维度变化，输出维度为 (batch_size, seq_len, d)
        # linear_layer = nn.Linear(concat_inputs.size(-1), 768)
        transformed_inputs = self.linear(concat_inputs)

        # Step 4: 经过层正则化和self-attention操作，输出维度仍为 (batch_size, seq_len, d)
        normalized_inputs = self.layerNorm(transformed_inputs)
        # self_attention = torch.matmul(normalized_inputs, normalized_inputs.transpose(-2, -1))
        # attended_inputs = torch.matmul(self_attention, normalized_inputs)
        attended_inputs = self.self_attention(normalized_inputs)

        # Step 5: 将经过self-attention后的向量与每个token相加，输出维度仍为 (batch_size, seq_len, d)
        output = attended_inputs + transformed_inputs

        return output
        # return None

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    # def forward(self, input_x, seq_lens):
    def forward(self, input_x):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x
